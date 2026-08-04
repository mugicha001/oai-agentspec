"""L2: ConversationService の HITL（ツール実行承認）を実 Agent + FakeModel で検証する。

承認必須ツール（`function_tool(..., needs_approval=True)`）を載せたエージェントを registry に
登録し、FakeModel が当該ツールへ ToolCall を返す応答 →（再開後）テキスト応答の順をキューする。
Runner が needs_approval ツールの ToolCall を検知して `result.interruptions` を生成し、
ConversationService が承認待ち（pending）を返す。承認/却下の解決で再開し最終応答を返す経路と、
承認前の非実行保証（NFR-7）を確認する。実 LLM は呼ばない。

承認の最小粒度は call_id。`tool_call_response(..., call_id=...)` / `multi_tool_call_response`
で call_id を可変にし、複数承認待ち（FR-7）を再現する。

`apply_approvals` は承認待ち item を `state.get_interruptions()` で引く（SDK の `RunState` は
`interruptions` 属性を持たず `get_interruptions()` を公開する）。resolve_approvals /
stream_resolve はこの経路で valid な call_id を正しく引き当てて approve/reject/段階解決する。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from oai_agentspec import AgentRegistry, AgentSpec
from oai_agentspec.runtime.conversation import (
    ApprovalDecision,
    ApprovalRequired,
    ConversationError,
    ConversationErrorCode,
    ConversationService,
    PendingApproval,
    StreamDelta,
    StreamDone,
)
from oai_agentspec.runtime.deterministic import (
    multi_tool_call_response,
    text_response,
    tool_call_response,
)

from _helpers.approval import QueuedFakeModel, ToolRecorder, make_approval_tool

pytestmark = pytest.mark.integration


def _service_with_tool(
    model: QueuedFakeModel,
    tool: Any,
    *,
    name: str = "bot",
    session_policy: Any = None,
) -> ConversationService:
    """承認必須ツールを載せた FakeModel エージェントの ConversationService を作る。"""
    reg = AgentRegistry()
    reg.register(AgentSpec(name=name, instructions="b", model=model, tools=[tool]))
    return ConversationService(reg, session_policy=session_policy)


# ----------------------------------------------------------------------
# 承認待ちの検知（send / stream）と承認前の非実行（FR-1/FR-2/FR-3/NFR-7）
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_send_returns_pending_and_tool_not_executed() -> None:
    """承認必須ツール呼び出しで send が pending を返し、ツールは未実行（FR-1/FR-3/NFR-7）。"""
    recorder = ToolRecorder()
    tool = make_approval_tool(recorder, name="danger")
    model = QueuedFakeModel().queue(tool_call_response("danger", '{"x": "v"}', call_id="c1"))
    svc = _service_with_tool(model, tool)
    cid = await svc.create_conversation()

    result = await svc.send("bot", "go", conversation_id=cid)

    assert result.status == "pending"
    assert result.output is None
    assert result.pending == [PendingApproval(tool_name="danger", call_id="c1")]
    # 承認前はツールが一切実行されていない（NFR-7・実行記録ゼロ）。
    assert recorder.executed == []


@pytest.mark.asyncio
async def test_stream_yields_approval_required_without_done() -> None:
    """stream が ApprovalRequired を yield し StreamDone を出さない・ツール未実行（FR-2/NFR-7）。"""
    recorder = ToolRecorder()
    tool = make_approval_tool(recorder, name="danger")
    model = QueuedFakeModel().queue(tool_call_response("danger", '{"x": "v"}', call_id="s1"))
    svc = _service_with_tool(model, tool)
    cid = await svc.create_conversation()

    events = [e async for e in svc.stream("bot", "go", conversation_id=cid)]

    # StreamDone は出ず、終端は ApprovalRequired（専用イベント・StreamEvent Union 非混入）。
    assert not any(isinstance(e, StreamDone) for e in events)
    approval = next(e for e in events if isinstance(e, ApprovalRequired))
    assert approval.approvals == [PendingApproval(tool_name="danger", call_id="s1")]
    assert recorder.executed == []


@pytest.mark.asyncio
async def test_stream_without_approval_tool_yields_only_delta_done() -> None:
    """承認必須ツール非宣言の stream は StreamDelta*→StreamDone のみ（既存挙動不変・NFR-6）。"""
    recorder = ToolRecorder()
    tool = make_approval_tool(recorder, name="danger")
    # ツール呼び出しを返さず最初からテキストを返す（承認待ちが発生しないターン）。
    model = QueuedFakeModel().queue(text_response("plain answer"))
    svc = _service_with_tool(model, tool)
    cid = await svc.create_conversation()

    events = [e async for e in svc.stream("bot", "hi", conversation_id=cid)]

    assert not any(isinstance(e, ApprovalRequired) for e in events)
    assert isinstance(events[-1], StreamDone)
    assert events[-1].final_output == "plain answer"


@pytest.mark.asyncio
async def test_send_without_approval_tool_returns_final() -> None:
    """承認必須ツール非宣言の send は従来どおり final を返す（回帰・NFR-6）。"""
    recorder = ToolRecorder()
    tool = make_approval_tool(recorder, name="danger")
    model = QueuedFakeModel().queue(text_response("plain final"))
    svc = _service_with_tool(model, tool)
    cid = await svc.create_conversation()

    result = await svc.send("bot", "hi", conversation_id=cid)

    assert result.status == "final"
    assert result.output == "plain final"
    assert result.pending == []


# ----------------------------------------------------------------------
# 承認待ち中の新ターン抑止（P1・安全性）
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_send_while_pending_returns_existing_without_new_run() -> None:
    """承認待ち未解決のまま再度 send すると新ターンを開始せず既存承認待ちを返す（P1）。"""
    recorder = ToolRecorder()
    tool = make_approval_tool(recorder, name="danger")
    model = QueuedFakeModel().queue(tool_call_response("danger", '{"x": "v"}', call_id="c1"))
    svc = _service_with_tool(model, tool)
    cid = await svc.create_conversation()

    first = await svc.send("bot", "go", conversation_id=cid)
    assert first.status == "pending"
    calls_after_first = len(model.inputs)

    # 解決せずに別メッセージを送る → 新たな Runner.run を回さず既存承認待ちを返す。
    second = await svc.send("bot", "another message", conversation_id=cid)
    assert second.status == "pending"
    assert second.pending == [PendingApproval(tool_name="danger", call_id="c1")]
    # モデルは再呼び出しされない（session に新履歴が追記されない）。
    assert len(model.inputs) == calls_after_first
    # ツールは依然として未実行（NFR-7）。
    assert recorder.executed == []


@pytest.mark.asyncio
async def test_stream_while_pending_yields_existing_without_new_run() -> None:
    """承認待ち未解決のまま再度 stream すると新ターンを開始せず ApprovalRequired を再提示（P1）。"""
    recorder = ToolRecorder()
    tool = make_approval_tool(recorder, name="danger")
    model = QueuedFakeModel().queue(tool_call_response("danger", '{"x": "v"}', call_id="c1"))
    svc = _service_with_tool(model, tool)
    cid = await svc.create_conversation()

    await svc.send("bot", "go", conversation_id=cid)
    calls_after_first = len(model.inputs)

    events = [e async for e in svc.stream("bot", "more text", conversation_id=cid)]

    # StreamDelta/StreamDone は出ず、既存の承認待ちが ApprovalRequired で再提示される。
    assert not any(isinstance(e, (StreamDelta, StreamDone)) for e in events)
    approval = next(e for e in events if isinstance(e, ApprovalRequired))
    assert approval.approvals == [PendingApproval(tool_name="danger", call_id="c1")]
    # モデル再呼び出しなし・ツール未実行。
    assert len(model.inputs) == calls_after_first
    assert recorder.executed == []


# ----------------------------------------------------------------------
# 承認待ち取得（冪等・FR-3）
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_pending_approvals_idempotent() -> None:
    """pending_approvals は承認待ち一覧を冪等に返す（複数回呼んでも同一・FR-3）。"""
    recorder = ToolRecorder()
    tool = make_approval_tool(recorder, name="danger")
    model = QueuedFakeModel().queue(tool_call_response("danger", '{"x": "v"}', call_id="c1"))
    svc = _service_with_tool(model, tool)
    cid = await svc.create_conversation()
    await svc.send("bot", "go", conversation_id=cid)

    first = await svc.pending_approvals(cid)
    second = await svc.pending_approvals(cid)

    assert first == [PendingApproval(tool_name="danger", call_id="c1")]
    assert first == second


@pytest.mark.asyncio
async def test_pending_approvals_empty_when_no_interruption() -> None:
    """承認待ちが無い会話の pending_approvals は空リスト。"""
    recorder = ToolRecorder()
    tool = make_approval_tool(recorder, name="danger")
    model = QueuedFakeModel().queue(text_response("done"))
    svc = _service_with_tool(model, tool)
    cid = await svc.create_conversation()
    await svc.send("bot", "hi", conversation_id=cid)

    assert await svc.pending_approvals(cid) == []


# ----------------------------------------------------------------------
# 承認系エラー（検証先行・state 不変・FR-4 / NFR-5）
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_resolve_without_pending_raises_no_pending_approval() -> None:
    """承認待ちが無い会話への resolve は NO_PENDING_APPROVAL（FR-4・NFR-5）。"""
    recorder = ToolRecorder()
    tool = make_approval_tool(recorder, name="danger")
    model = QueuedFakeModel().queue(text_response("done"))
    svc = _service_with_tool(model, tool)
    cid = await svc.create_conversation()
    await svc.send("bot", "hi", conversation_id=cid)

    with pytest.raises(ConversationError) as exc:
        await svc.resolve_approvals(cid, [{"call_id": "x", "decision": "approve"}])
    assert exc.value.code == ConversationErrorCode.NO_PENDING_APPROVAL


@pytest.mark.asyncio
async def test_resolve_unknown_call_id_raises_and_keeps_state() -> None:
    """未知 call_id の resolve は UNKNOWN_APPROVAL を返し state は変化しない（FR-4・検証先行）。"""
    recorder = ToolRecorder()
    tool = make_approval_tool(recorder, name="danger")
    model = QueuedFakeModel().queue(tool_call_response("danger", '{"x": "v"}', call_id="c1"))
    svc = _service_with_tool(model, tool)
    cid = await svc.create_conversation()
    await svc.send("bot", "go", conversation_id=cid)

    with pytest.raises(ConversationError) as exc:
        await svc.resolve_approvals(cid, [{"call_id": "ghost", "decision": "approve"}])
    assert exc.value.code == ConversationErrorCode.UNKNOWN_APPROVAL
    # state 不変: 元の承認待ちが残り、ツールは未実行のまま（FR-4・NFR-7）。
    assert await svc.pending_approvals(cid) == [PendingApproval(tool_name="danger", call_id="c1")]
    assert recorder.executed == []


@pytest.mark.asyncio
async def test_resolve_mixed_batch_with_unknown_does_not_apply_any() -> None:
    """混在バッチ（正常 + 未知）は正常分も含め state を変えず UNKNOWN_APPROVAL（FR-4・2 パス）。"""
    recorder = ToolRecorder()
    tool = make_approval_tool(recorder, name="danger")
    model = QueuedFakeModel().queue(tool_call_response("danger", '{"x": "v"}', call_id="c1"))
    svc = _service_with_tool(model, tool)
    cid = await svc.create_conversation()
    await svc.send("bot", "go", conversation_id=cid)

    with pytest.raises(ConversationError) as exc:
        await svc.resolve_approvals(
            cid,
            [
                {"call_id": "c1", "decision": "approve"},
                {"call_id": "ghost", "decision": "approve"},
            ],
        )
    assert exc.value.code == ConversationErrorCode.UNKNOWN_APPROVAL
    # 正常分（c1）も適用されない: 再開せず承認待ち維持・ツール未実行（部分適用を避ける・FR-4）。
    assert await svc.pending_approvals(cid) == [PendingApproval(tool_name="danger", call_id="c1")]
    assert recorder.executed == []


@pytest.mark.asyncio
async def test_resolve_unknown_conversation_raises() -> None:
    """未登録 conversation_id への resolve は UNKNOWN_CONVERSATION。"""
    recorder = ToolRecorder()
    tool = make_approval_tool(recorder, name="danger")
    svc = _service_with_tool(QueuedFakeModel(), tool)

    with pytest.raises(ConversationError) as exc:
        await svc.resolve_approvals("missing", [{"call_id": "x", "decision": "approve"}])
    assert exc.value.code == ConversationErrorCode.UNKNOWN_CONVERSATION


# ----------------------------------------------------------------------
# 承認後の再開（approve → 実行 → final）/ 却下（reject → 非実行 → final）
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_approve_executes_tool_and_returns_final() -> None:
    """approve で承認待ちを解決 → ツール実行（実行記録あり）→ final（FR-6・NFR-7）。"""
    recorder = ToolRecorder()
    tool = make_approval_tool(recorder, name="danger")
    model = (
        QueuedFakeModel()
        .queue(tool_call_response("danger", '{"x": "v"}', call_id="c1"))
        .queue(text_response("all done"))
    )
    svc = _service_with_tool(model, tool)
    cid = await svc.create_conversation()
    await svc.send("bot", "go", conversation_id=cid)

    result = await svc.resolve_approvals(cid, [{"call_id": "c1", "decision": "approve"}])

    assert result.status == "final"
    assert result.output == "all done"
    # approve 後に初めて実行記録が残る（NFR-7・承認前ゼロ → approve 後実行）。
    assert recorder.executed == ["v"]
    # 中断状態はクリアされ承認待ちは空になる。
    assert await svc.pending_approvals(cid) == []


@dataclass
class _RaiseOnceModel(QueuedFakeModel):
    """指定回目の get_response で 1 度だけ例外を投げる FakeModel（再開一時失敗の再現）。

    Attributes:
        raise_on_call: この回数目の get_response 呼び出しで RuntimeError を投げる（1 始まり）。
        calls: これまでの get_response 呼び出し回数（内部カウンタ）。
    """

    raise_on_call: int = 0
    calls: int = 0

    async def get_response(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        """指定回で 1 度だけ例外を投げ、それ以外はキュー応答を返す。"""
        self.calls += 1
        if self.calls == self.raise_on_call:
            raise RuntimeError("transient resume failure")
        return await super().get_response(*args, **kwargs)


@pytest.mark.asyncio
async def test_resume_failure_then_retry_resumes_and_finalizes() -> None:
    """全 approve 後の再開が一時失敗 → 空 decisions で再 resolve すると再開して final（P2-1）。

    再開失敗時に state（解決適用済み）が保持され、entry の残承認待ちが state の実状（全解決=空）へ
    同期・再永続されるため、空 decisions の再 resolve が resume をやり直せる（会話が詰まらない）。
    """
    recorder = ToolRecorder()
    tool = make_approval_tool(recorder, name="danger")
    # call1=初回 send（tool call）/ call2=最初の resume（失敗）/ call3=再試行 resume（text）。
    model = _RaiseOnceModel(raise_on_call=2)
    model.queue(tool_call_response("danger", '{"x": "v"}', call_id="c1"))
    model.queue(text_response("recovered"))
    svc = _service_with_tool(model, tool)
    cid = await svc.create_conversation()
    await svc.send("bot", "go", conversation_id=cid)

    # 全 approve → 再開が一時エラーで失敗（構造化エラー・EXECUTION_ERROR）。
    with pytest.raises(ConversationError) as exc:
        await svc.resolve_approvals(cid, [{"call_id": "c1", "decision": "approve"}])
    assert exc.value.code == ConversationErrorCode.EXECUTION_ERROR

    # この時点で承認待ちは「全解決・未再開」（state 由来で空）。同じ approve は既解決で弾かれない。
    assert await svc.pending_approvals(cid) == []

    # 空 decisions の再 resolve で再開をやり直せる → ツール実行・final（詰まらない・P2-1）。
    result = await svc.resolve_approvals(cid, [])

    assert result.status == "final"
    assert result.output == "recovered"
    assert recorder.executed == ["v"]
    assert await svc.pending_approvals(cid) == []


@pytest.mark.asyncio
async def test_reject_does_not_execute_and_continues() -> None:
    """reject でツール未実行・会話継続・final（FR-5・NFR-7）。"""
    recorder = ToolRecorder()
    tool = make_approval_tool(recorder, name="danger")
    model = (
        QueuedFakeModel()
        .queue(tool_call_response("danger", '{"x": "v"}', call_id="c1"))
        .queue(text_response("continued after reject"))
    )
    svc = _service_with_tool(model, tool)
    cid = await svc.create_conversation()
    await svc.send("bot", "go", conversation_id=cid)

    result = await svc.resolve_approvals(cid, [{"call_id": "c1", "decision": "reject"}])

    assert result.status == "final"
    assert result.output == "continued after reject"
    # reject ではツールは一切実行されない（NFR-7）。
    assert recorder.executed == []


@pytest.mark.asyncio
async def test_resolve_with_approval_decision_object_approves_and_executes() -> None:
    """型付き入力 ApprovalDecision(approve=True) で承認 → ツール実行・final（dict と等価）。"""
    recorder = ToolRecorder()
    tool = make_approval_tool(recorder, name="danger")
    model = (
        QueuedFakeModel()
        .queue(tool_call_response("danger", '{"x": "v"}', call_id="c1"))
        .queue(text_response("all done"))
    )
    svc = _service_with_tool(model, tool)
    cid = await svc.create_conversation()
    await svc.send("bot", "go", conversation_id=cid)

    result = await svc.resolve_approvals(cid, [ApprovalDecision(call_id="c1", approve=True)])

    assert result.status == "final"
    assert result.output == "all done"
    # 型付き approve でも dict 経路と同じくツールが実行される（NFR-7）。
    assert recorder.executed == ["v"]
    assert await svc.pending_approvals(cid) == []


@pytest.mark.asyncio
async def test_resolve_with_approval_decision_object_rejects_without_executing() -> None:
    """型付き入力 ApprovalDecision(approve=False) で却下 → 非実行・final（fail-closed）。"""
    recorder = ToolRecorder()
    tool = make_approval_tool(recorder, name="danger")
    model = (
        QueuedFakeModel()
        .queue(tool_call_response("danger", '{"x": "v"}', call_id="c1"))
        .queue(text_response("continued after reject"))
    )
    svc = _service_with_tool(model, tool)
    cid = await svc.create_conversation()
    await svc.send("bot", "go", conversation_id=cid)

    result = await svc.resolve_approvals(
        cid, [ApprovalDecision(call_id="c1", approve=False, rejection_message="do not run")]
    )

    assert result.status == "final"
    assert result.output == "continued after reject"
    # 型付き reject でもツールは一切実行されない（NFR-7）。
    assert recorder.executed == []


@pytest.mark.asyncio
async def test_reject_with_message_reflected_in_continuation_input() -> None:
    """reject に却下理由を付けると再開入力（次ターン）へ反映される（FR-5）。"""
    recorder = ToolRecorder()
    tool = make_approval_tool(recorder, name="danger")
    model = (
        QueuedFakeModel()
        .queue(tool_call_response("danger", '{"x": "v"}', call_id="c1"))
        .queue(text_response("ok"))
    )
    svc = _service_with_tool(model, tool)
    cid = await svc.create_conversation()
    await svc.send("bot", "go", conversation_id=cid)

    await svc.resolve_approvals(
        cid,
        [{"call_id": "c1", "decision": "reject", "rejection_message": "do not run"}],
    )

    # 再開 run の input（2 回目の get_response）に却下理由が文字列として現れる。
    resumed_input = model.inputs[-1]
    assert "do not run" in repr(resumed_input)
    assert recorder.executed == []


@pytest.mark.asyncio
async def test_fail_closed_unknown_decision_is_treated_as_reject() -> None:
    """decision 未知値は reject 扱い（fail-closed・ツール非実行・NFR-7）。"""
    recorder = ToolRecorder()
    tool = make_approval_tool(recorder, name="danger")
    model = (
        QueuedFakeModel()
        .queue(tool_call_response("danger", '{"x": "v"}', call_id="c1"))
        .queue(text_response("done"))
    )
    svc = _service_with_tool(model, tool)
    cid = await svc.create_conversation()
    await svc.send("bot", "go", conversation_id=cid)

    result = await svc.resolve_approvals(cid, [{"call_id": "c1", "decision": "maybe"}])

    assert result.status == "final"
    # "approve" 明示一致以外は reject（非実行）へ倒す。
    assert recorder.executed == []


# ----------------------------------------------------------------------
# 複数承認待ち（段階解決・FR-7）
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_multiple_pending_approvals_listed_by_call_id() -> None:
    """1 ターンで複数の承認必須ツール呼び出しは call_id 単位の一覧になる（FR-7）。"""
    recorder = ToolRecorder()
    tool = make_approval_tool(recorder, name="danger")
    model = QueuedFakeModel().queue(
        multi_tool_call_response(
            [
                ("danger", '{"x": "a"}', "c1"),
                ("danger", '{"x": "b"}', "c2"),
            ]
        )
    )
    svc = _service_with_tool(model, tool)
    cid = await svc.create_conversation()

    result = await svc.send("bot", "go", conversation_id=cid)

    assert result.status == "pending"
    call_ids = {p.call_id for p in result.pending}
    assert call_ids == {"c1", "c2"}
    assert recorder.executed == []


@pytest.mark.asyncio
async def test_partial_resolution_keeps_remaining_pending() -> None:
    """一部のみ decision を与えると未指定 call_id は残り再開しない（段階解決・FR-7）。"""
    recorder = ToolRecorder()
    tool = make_approval_tool(recorder, name="danger")
    model = (
        QueuedFakeModel()
        .queue(
            multi_tool_call_response(
                [
                    ("danger", '{"x": "a"}', "c1"),
                    ("danger", '{"x": "b"}', "c2"),
                ]
            )
        )
        .queue(text_response("both resolved"))
    )
    svc = _service_with_tool(model, tool)
    cid = await svc.create_conversation()
    await svc.send("bot", "go", conversation_id=cid)

    # c1 のみ approve（c2 は未指定 = 未解決のまま残す）。
    partial = await svc.resolve_approvals(cid, [{"call_id": "c1", "decision": "approve"}])
    assert partial.status == "pending"
    assert {p.call_id for p in partial.pending} == {"c2"}


@pytest.mark.asyncio
async def test_multiple_approve_one_reject_executes_only_approved() -> None:
    """複数承認待ちで一部 approve・一部 reject → approve 分のみ実行（FR-7・NFR-7）。"""
    recorder = ToolRecorder()
    tool = make_approval_tool(recorder, name="danger")
    model = (
        QueuedFakeModel()
        .queue(
            multi_tool_call_response(
                [
                    ("danger", '{"x": "a"}', "c1"),
                    ("danger", '{"x": "b"}', "c2"),
                ]
            )
        )
        .queue(text_response("resolved"))
    )
    svc = _service_with_tool(model, tool)
    cid = await svc.create_conversation()
    await svc.send("bot", "go", conversation_id=cid)

    result = await svc.resolve_approvals(
        cid,
        [
            {"call_id": "c1", "decision": "approve"},
            {"call_id": "c2", "decision": "reject"},
        ],
    )

    assert result.status == "final"
    # approve した c1（x=a）のみ実行され、reject した c2（x=b）は実行されない。
    assert recorder.executed == ["a"]


# ----------------------------------------------------------------------
# ストリーミング再開（stream_resolve）
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_stream_resolve_approve_resumes_and_done() -> None:
    """stream_resolve(approve) で再開し StreamDelta*→StreamDone・ツール実行（FR-6）。"""
    recorder = ToolRecorder()
    tool = make_approval_tool(recorder, name="danger")
    model = (
        QueuedFakeModel()
        .queue(tool_call_response("danger", '{"x": "v"}', call_id="s1"))
        .queue(text_response("streamed resume"))
    )
    svc = _service_with_tool(model, tool)
    cid = await svc.create_conversation()
    # 先に承認待ちを発生させる。
    _ = [e async for e in svc.stream("bot", "go", conversation_id=cid)]

    events = [e async for e in svc.stream_resolve(cid, [{"call_id": "s1", "decision": "approve"}])]

    assert isinstance(events[-1], StreamDone)
    assert events[-1].final_output == "streamed resume"
    assert recorder.executed == ["v"]
    assert any(isinstance(e, StreamDelta) for e in events)
