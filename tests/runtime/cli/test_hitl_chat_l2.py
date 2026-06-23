"""L2: CLI 承認 UI（chat の _collect_decisions / _prompt_decision / 承認往復）を fake で検証する。

`ConversationClient` をフェイクへ、入力ヘルパ `_ainput` をキュー入力へ差し替えて、承認待ち発生 →
approve で継続 / reject で継続 / 複数個別選択 / EOF → reject（安全側・NFR-7）を確認する。
本ファイルは CLI の承認 UI ロジックに焦点を当てるため、フェイク client が承認往復の応答を制御する
（実 apply_approvals バグの影響を受けない）。httpx / websockets / rich 未導入環境では importorskip。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

pytest.importorskip("httpx")
pytest.importorskip("websockets")
pytest.importorskip("rich")

from rich.console import Console  # noqa: E402

from oai_agentspec.runtime.cli import chat as chat_module  # noqa: E402
from oai_agentspec.runtime.cli.client import (  # noqa: E402
    ApprovalRequired,
    PendingApproval,
    SendResult,
    StreamDone,
    StreamToken,
)

pytestmark = pytest.mark.integration


def _patch_ainput(monkeypatch: pytest.MonkeyPatch, inputs: list[str]) -> None:
    """_ainput をキュー入力へ差し替える（使い切ると None=EOF）。"""
    it = iter(inputs)

    async def _fake_ainput(_console: Any, _prompt: str) -> str | None:
        try:
            return next(it)
        except StopIteration:
            return None

    monkeypatch.setattr(chat_module, "_ainput", _fake_ainput)


# ----------------------------------------------------------------------
# _prompt_decision: approve / reject / 却下理由 / EOF→reject
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_prompt_decision_approve(monkeypatch: pytest.MonkeyPatch) -> None:
    """approve（a）を選ぶと decision=approve を返す。"""
    _patch_ainput(monkeypatch, ["a"])
    approval = PendingApproval(tool_name="danger", call_id="c1")
    decision = await chat_module._prompt_decision(Console(), approval)
    assert decision == {"call_id": "c1", "decision": "approve", "rejection_message": None}


@pytest.mark.asyncio
async def test_prompt_decision_y_n_convention(monkeypatch: pytest.MonkeyPatch) -> None:
    """主表記 y=承認 / n=却下（[y/N]）が受理される。"""
    approval = PendingApproval(tool_name="danger", call_id="c1")
    _patch_ainput(monkeypatch, ["y"])
    yes = await chat_module._prompt_decision(Console(), approval)
    assert yes["decision"] == "approve"
    _patch_ainput(monkeypatch, ["n"])  # 却下理由はスキップ（Enter 相当の空入力）
    no = await chat_module._prompt_decision(Console(), approval)
    assert no["decision"] == "reject"


@pytest.mark.asyncio
async def test_prompt_decision_reject_with_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    """reject（r）を選び理由を入力すると rejection_message に反映される。"""
    _patch_ainput(monkeypatch, ["r", "危険なので却下"])
    approval = PendingApproval(tool_name="danger", call_id="c1")
    decision = await chat_module._prompt_decision(Console(), approval)
    assert decision["decision"] == "reject"
    assert decision["rejection_message"] == "危険なので却下"


@pytest.mark.asyncio
async def test_prompt_decision_eof_is_reject(monkeypatch: pytest.MonkeyPatch) -> None:
    """入力 EOF（None）は安全側に倒し reject 扱いとする（NFR-7）。"""
    _patch_ainput(monkeypatch, [])  # 即 EOF
    approval = PendingApproval(tool_name="danger", call_id="c1")
    decision = await chat_module._prompt_decision(Console(), approval)
    assert decision["decision"] == "reject"


@pytest.mark.asyncio
async def test_prompt_decision_unknown_input_is_reject(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """認識できない入力は reject へ倒す（fail-closed・NFR-7）。"""
    _patch_ainput(monkeypatch, ["xyz", ""])  # 未知選択 → 却下理由スキップ
    approval = PendingApproval(tool_name="danger", call_id="c1")
    decision = await chat_module._prompt_decision(Console(), approval)
    assert decision["decision"] == "reject"
    assert decision["rejection_message"] is None


# ----------------------------------------------------------------------
# _collect_decisions: 複数承認待ちを call_id ごとに個別選択（FR-9）
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_collect_decisions_multiple_individual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """複数承認待ちを call_id ごとに個別に approve/reject 選択する（FR-9）。"""
    # c1=approve、c2=reject（理由なし）。
    _patch_ainput(monkeypatch, ["a", "r", ""])
    pending = [
        PendingApproval(tool_name="danger", call_id="c1"),
        PendingApproval(tool_name="danger", call_id="c2"),
    ]
    decisions = await chat_module._collect_decisions(Console(), pending)
    assert [d["call_id"] for d in decisions] == ["c1", "c2"]
    assert decisions[0]["decision"] == "approve"
    assert decisions[1]["decision"] == "reject"


# ----------------------------------------------------------------------
# _turn_non_streaming: 承認待ち → approve で継続 / reject で継続（FR-9・段階解決）
# ----------------------------------------------------------------------
class _FakeApprovalClient:
    """承認往復を制御するフェイク client（send → pending → resolve → final）。"""

    def __init__(self, *, final_output: str = "final") -> None:
        self._final = final_output
        self.resolved: list[list[dict[str, Any]]] = []

    async def send(self, agent_name: str | None, text: str, *, conversation_id: str) -> SendResult:
        """1 件の承認待ちを返す。"""
        return SendResult(
            status="pending",
            pending=[PendingApproval(tool_name="danger", call_id="c1")],
        )

    async def resolve_approvals(
        self, conversation_id: str, decisions: list[dict[str, Any]]
    ) -> SendResult:
        """承認/却下を記録し最終応答を返す（全解決とみなす）。"""
        self.resolved.append(decisions)
        return SendResult(status="final", output=self._final)


@pytest.mark.asyncio
async def test_turn_non_streaming_approve_continues(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """非ストリーミングで承認待ち → approve で継続し最終応答を表示する（FR-9）。"""
    _patch_ainput(monkeypatch, ["a"])
    client = _FakeApprovalClient(final_output="承認後の応答")
    await chat_module._turn_non_streaming(client, Console(), "bot", "go", conversation_id="conv-1")
    assert client.resolved[0][0]["decision"] == "approve"
    assert "承認後の応答" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_turn_non_streaming_reject_continues(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """非ストリーミングで承認待ち → reject で継続し最終応答を表示する（FR-5/FR-9）。"""
    _patch_ainput(monkeypatch, ["r", ""])
    client = _FakeApprovalClient(final_output="却下後の応答")
    await chat_module._turn_non_streaming(client, Console(), "bot", "go", conversation_id="conv-1")
    assert client.resolved[0][0]["decision"] == "reject"
    assert "却下後の応答" in capsys.readouterr().out


# ----------------------------------------------------------------------
# _turn_streaming: WS approval_handler 経由の承認往復（FR-9）
# ----------------------------------------------------------------------
class _FakeStreamClient:
    """WS の承認往復を制御するフェイク client（stream → approval_required → handler → done）。"""

    def __init__(self) -> None:
        self.handled_decisions: list[dict[str, Any]] = []

    async def stream(
        self,
        agent_name: str | None,
        text: str,
        *,
        conversation_id: str,
        approval_handler: Any = None,
    ) -> AsyncIterator[Any]:
        """承認待ちを 1 件 yield し、handler の decisions を記録してから再開 token/done を流す。"""
        pending = [PendingApproval(tool_name="danger", call_id="c1")]
        yield ApprovalRequired(pending=pending)
        if approval_handler is not None:
            self.handled_decisions = await approval_handler(pending)
        yield StreamToken(text="再開")
        yield StreamDone(output="再開")


@pytest.mark.asyncio
async def test_turn_streaming_approval_roundtrip(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """ストリーミングで承認待ち → approval_handler で approve → 再開 token を表示する（FR-9）。"""
    _patch_ainput(monkeypatch, ["a"])
    client = _FakeStreamClient()
    await chat_module._turn_streaming(client, Console(), "bot", "go", conversation_id="conv-1")
    assert client.handled_decisions[0]["decision"] == "approve"
    assert "再開" in capsys.readouterr().out
