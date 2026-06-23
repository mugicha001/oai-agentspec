"""L2: CLI 対話ループ（chat・rich 2層UI）と argparse main を fake クライアントで検証する。

実ネットワーク・実端末を使わず、`ConversationClient` をフェイクへ、入力ヘルパ `_ainput` を
キュー化した値へ差し替えて、セッション選択画面 -> 会話（新規 / 復元）-> /back・/quit を
確認する。httpx / websockets 未導入環境では importorskip でスキップする。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

pytest.importorskip("httpx")
pytest.importorskip("websockets")
pytest.importorskip("rich")

from oai_agentspec.runtime.cli import chat as chat_module  # noqa: E402
from oai_agentspec.runtime.cli.client import (  # noqa: E402
    PendingApproval,
    SendResult,
    SessionMeta,
    StreamDone,
    StreamToken,
)
from oai_agentspec.runtime.cli.main import build_parser, main  # noqa: E402

pytestmark = pytest.mark.integration


class FakeClient:
    """`ConversationClient` 互換のフェイク（async context・REST/WS をメモリで模倣）。"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """送受信ログと応答データを初期化する。"""
        self.entry: str | None = "bot"
        self.sessions: list[SessionMeta] = []
        self.history: list[dict[str, Any]] = []
        self.sent: list[tuple[str | None, str, str]] = []
        self.created: list[str | None] = []
        self.history_requested: tuple[str, int | None] | None = None
        # 入口ドレイン用の承認待ち（空なら no-op）。resolve で消費される。
        self.pending: list[PendingApproval] = []
        self.resolved: list[tuple[str, list[dict[str, Any]]]] = []
        self.get_approvals_calls = 0

    async def __aenter__(self) -> FakeClient:
        """コンテキストに入る。"""
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """コンテキストを抜ける。"""
        return None

    async def get_entry(self) -> str | None:
        """エントリエージェント名を返す。"""
        return self.entry

    async def list_sessions(self) -> list[SessionMeta]:
        """永続化済み session メタを返す。"""
        return self.sessions

    async def get_history(
        self, session_id: str, *, limit: int | None = None
    ) -> list[dict[str, Any]]:
        """復元時の過去履歴を返す（取得要求を記録）。"""
        self.history_requested = (session_id, limit)
        return self.history

    async def create_conversation(self, *, session_id: str | None = None) -> str:
        """会話を作成し ID を返す。"""
        self.created.append(session_id)
        return "conv-1"

    async def get_approvals(self, conversation_id: str) -> list[PendingApproval]:
        """現在の承認待ち一覧を返す（入口ドレイン用・呼び出し回数を記録）。"""
        self.get_approvals_calls += 1
        return list(self.pending)

    async def resolve_approvals(
        self, conversation_id: str, decisions: list[dict[str, Any]]
    ) -> SendResult:
        """承認/却下を適用し承認待ちを消費して final を返す（段階解決は本フェイクでは一括）。"""
        self.resolved.append((conversation_id, decisions))
        self.pending = []
        return SendResult(status="final", output="resumed")

    async def send(self, agent_name: str | None, text: str, *, conversation_id: str) -> SendResult:
        """非ストリーム応答（最終応答）を返す。"""
        self.sent.append((agent_name, text, conversation_id))
        return SendResult(status="final", output=f"echo:{text}")

    async def stream(
        self,
        agent_name: str | None,
        text: str,
        *,
        conversation_id: str,
        approval_handler: Any = None,
    ) -> AsyncIterator[StreamToken | StreamDone]:
        """token を 2 つ流して done で終端する（承認待ちなし・approval_handler は未使用）。"""
        self.sent.append((agent_name, text, conversation_id))
        yield StreamToken(text="ab")
        yield StreamToken(text="cd")
        yield StreamDone(output="abcd")


def _patch(monkeypatch: pytest.MonkeyPatch, fake: FakeClient, inputs: list[str]) -> None:
    """ConversationClient をフェイクへ、_ainput をキュー入力へ差し替える。

    入力キューを使い切ると None（EOF 相当）を返す。
    """
    monkeypatch.setattr(chat_module, "ConversationClient", lambda *a, **k: fake)
    it = iter(inputs)

    async def _fake_ainput(_console: Any, _prompt: str) -> str | None:
        try:
            return next(it)
        except StopIteration:
            return None

    monkeypatch.setattr(chat_module, "_ainput", _fake_ainput)


@pytest.mark.asyncio
async def test_run_chat_new_non_streaming(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """新規会話を選び 1 ターン送信して /quit で終了する（エントリ起点・agent_name=None）。"""
    fake = FakeClient()
    _patch(monkeypatch, fake, ["n", "hello", "/quit"])
    rc = await chat_module.run_chat(base_url="http://x", stream=False)
    assert rc == 0
    # エントリ起点のため agent_name は None で送信される。
    assert fake.sent == [(None, "hello", "conv-1")]
    # 新規会話は sess- 接頭辞の session_id を採番して作成される。
    assert len(fake.created) == 1 and fake.created[0].startswith("sess-")
    assert "echo:hello" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_run_chat_new_streaming(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """ストリーミングで token を逐次表示し /exit で終了する。"""
    fake = FakeClient()
    _patch(monkeypatch, fake, ["n", "go", "/exit"])
    rc = await chat_module.run_chat(base_url="http://x", stream=True)
    assert rc == 0
    assert "abcd" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_run_chat_restore_by_number(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """過去 session 一覧から番号で復元を選び、その session_id で会話作成・履歴取得する（D5）。"""
    fake = FakeClient()
    fake.sessions = [
        SessionMeta(session_id="named-1", updated_at="t1", turn_count=1, preview="p1"),
        SessionMeta(session_id="named-2", updated_at="t2", turn_count=3, preview="p2"),
    ]
    fake.history = [
        {"role": "user", "content": "前回の質問"},
        {"role": "assistant", "content": [{"type": "output_text", "text": "前回の応答"}]},
    ]
    _patch(monkeypatch, fake, ["2", "hi", "/quit"])
    rc = await chat_module.run_chat(base_url="http://x", stream=False)
    assert rc == 0
    # 2 番目（named-2）で会話作成し、その履歴を直近 10 件で取得する。
    assert fake.created == ["named-2"]
    assert fake.history_requested == ("named-2", 10)
    out = capsys.readouterr().out
    # 復元履歴（過去の発話）が表示される。
    assert "前回の質問" in out
    assert "前回の応答" in out


@pytest.mark.asyncio
async def test_run_chat_restore_drains_pending_before_input(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """復元会話の入口で承認待ちをドレイン提示→approve してから入力が処理される（P2-2）。

    永続承認待ちのある session を復元すると、入力ループ前に get_approvals でドレインされ、承認 UI
    が出る。最初の _ainput で approve（a）を入力 → 解決・再開。続くユーザー入力（"first message"）が
    黙って捨てられず send で処理される（先頭入力の取りこぼし防止）。
    """
    fake = FakeClient()
    fake.sessions = [
        SessionMeta(session_id="named-1", updated_at="t1", turn_count=1, preview="p1"),
    ]
    # 復元時に承認待ちが 1 件ある（入口でドレインされる）。
    fake.pending = [PendingApproval(tool_name="danger", call_id="c1")]
    # 入力: 1=復元選択 / a=承認（ドレイン UI への応答）/ first message=通常入力 / /quit。
    _patch(monkeypatch, fake, ["1", "a", "first message", "/quit"])

    rc = await chat_module.run_chat(base_url="http://x", stream=False)
    assert rc == 0

    # 入口で get_approvals が呼ばれ、approve が resolve_approvals へ委譲された（ドレイン）。
    assert fake.get_approvals_calls >= 1
    assert len(fake.resolved) == 1
    assert fake.resolved[0][1] == [
        {"call_id": "c1", "decision": "approve", "rejection_message": None}
    ]
    # ドレイン後、先頭ユーザー入力（"first message"）が捨てられず send で処理される。
    assert (None, "first message", "conv-1") in fake.sent
    out = capsys.readouterr().out
    assert "承認待ち" in out  # 承認 UI が提示された
    assert "resumed" in out  # 再開後の応答が表示された
    assert "echo:first message" in out  # 先頭入力が処理された


@pytest.mark.asyncio
async def test_run_chat_back_returns_to_picker(monkeypatch: pytest.MonkeyPatch) -> None:
    """/back でセッション選択へ戻り、その後 q で終了する。"""
    fake = FakeClient()
    _patch(monkeypatch, fake, ["n", "/back", "q"])
    rc = await chat_module.run_chat(base_url="http://x", stream=False)
    assert rc == 0
    # 新規会話を 1 度作成（/back 後は会話を作らず picker で終了）。
    assert len(fake.created) == 1


@pytest.mark.asyncio
async def test_run_chat_quit_at_picker(monkeypatch: pytest.MonkeyPatch) -> None:
    """セッション選択画面で q を選ぶと会話を作らず終了する。"""
    fake = FakeClient()
    _patch(monkeypatch, fake, ["q"])
    rc = await chat_module.run_chat(base_url="http://x", stream=False)
    assert rc == 0
    assert fake.created == []


@pytest.mark.asyncio
async def test_run_chat_eof_at_picker_quits(monkeypatch: pytest.MonkeyPatch) -> None:
    """入力 EOF（None）でセッション選択画面から安全に終了する。"""
    fake = FakeClient()
    _patch(monkeypatch, fake, [])
    rc = await chat_module.run_chat(base_url="http://x", stream=False)
    assert rc == 0
    assert fake.created == []


@pytest.mark.asyncio
async def test_run_chat_entry_none_returns_1(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """エントリエージェントが無い（registry 空）なら 1 を返す。"""
    fake = FakeClient()
    fake.entry = None
    _patch(monkeypatch, fake, [])
    rc = await chat_module.run_chat(base_url="http://x", stream=False)
    assert rc == 1
    assert "エントリエージェントが見つかりません" in capsys.readouterr().out


def test_build_parser_defaults() -> None:
    """chat サブコマンドの既定（base-url / stream=True）を確認する。"""
    parser = build_parser()
    args = parser.parse_args(["chat"])
    assert args.command == "chat"
    assert args.base_url == "http://localhost:8000"
    assert args.stream is True
    # --agent / --session-id は廃止された。
    assert not hasattr(args, "agent")
    assert not hasattr(args, "session_id")


def test_build_parser_no_stream() -> None:
    """--no-stream で stream=False になる。"""
    parser = build_parser()
    args = parser.parse_args(["chat", "--no-stream"])
    assert args.stream is False


def test_main_no_command_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    """サブコマンド無しはヘルプを表示して 0 を返す。"""
    rc = main([])
    assert rc == 0
    assert "chat" in capsys.readouterr().out


def test_main_runs_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    """main chat が run_chat を呼ぶ（引数が伝播する）。"""
    captured: dict[str, Any] = {}

    async def _fake_run_chat(**kwargs: Any) -> int:
        captured.update(kwargs)
        return 0

    monkeypatch.setattr("oai_agentspec.runtime.cli.chat.run_chat", _fake_run_chat)
    rc = main(["chat", "--no-stream"])
    assert rc == 0
    assert captured["base_url"] == "http://localhost:8000"
    assert captured["stream"] is False
