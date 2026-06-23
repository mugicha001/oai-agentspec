"""L2: FastAPI 会話サーバ（REST + WebSocket）を TestClient + FakeModel で検証する。

REST（エージェント一覧 / 会話作成 / 非ストリーム会話）・WebSocket（token → done）・
構造化エラー（REST status / WS error）を確認する。実 LLM は呼ばない。fastapi 未インストール
環境では importorskip でスキップする。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

pytest.importorskip("fastapi")

from agents.items import ModelResponse  # noqa: E402
from agents.models.interface import Model  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from oai_agentspec import AgentRegistry, AgentSpec  # noqa: E402
from oai_agentspec.runtime.conversation import ConversationService, SessionPolicy  # noqa: E402
from oai_agentspec.runtime.serve import create_app  # noqa: E402

from _helpers.responses import text_response  # noqa: E402

pytestmark = pytest.mark.integration


class StreamingFakeModel(Model):
    """カンネドテキストを返し、ストリーミング時は delta + completed を流す Model。"""

    def __init__(self) -> None:
        """空のレスポンスキューで生成する。"""
        self._responses: list[ModelResponse] = []

    def queue_text(self, text: str) -> StreamingFakeModel:
        """テキスト応答をキューに積む（自身を返す）。"""
        self._responses.append(text_response(text))
        return self

    async def get_response(
        self,
        system_instructions: str | None = None,
        input: Any = None,  # noqa: A002 - SDK シグネチャに追従
        *args: Any,
        **kwargs: Any,
    ) -> ModelResponse:
        """キュー先頭の応答を返す（空なら空テキスト）。"""
        if self._responses:
            return self._responses.pop(0)
        return text_response("")

    async def stream_response(  # type: ignore[override]
        self,
        system_instructions: str | None = None,
        input: Any = None,  # noqa: A002 - SDK シグネチャに追従
        *args: Any,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """最終テキストを delta + completed イベントで流す（run_streamed 対応）。"""
        from oai_agentspec._adapters import _completed_event, _text_delta_events, _text_of

        response = await self.get_response(system_instructions, input, *args, **kwargs)
        text = _text_of(response)
        seq = 0
        for event in _text_delta_events(text):
            yield event
            seq = event.sequence_number + 1
        yield _completed_event(response.output, seq)


def _client(model: Model | None = None) -> TestClient:
    """1 つの FakeModel エージェントを登録した app の TestClient を作る。"""
    reg = AgentRegistry()
    reg.register(AgentSpec(name="bot", instructions="b", model=model or StreamingFakeModel()))
    app = create_app(ConversationService(reg))
    return TestClient(app)


class _ApiKeyErrorModel(Model):
    """get_response / stream_response で api_key 不備例外を送出する Model。

    `model=None`（SDK デフォルトモデル）は正当な用法のため実行前には拒否しない。モデル
    不備は実行時の SDK 例外として現れる経路を、決定論的に再現する（ヒューリスティックで
    `model_not_configured` へ分類される）。
    """

    async def get_response(self, system_instructions=None, input=None, *a, **k) -> ModelResponse:  # type: ignore[override]  # noqa: A002,E501
        raise RuntimeError("missing api_key")

    async def stream_response(
        self, system_instructions=None, input=None, *a, **k
    ) -> AsyncIterator[Any]:  # type: ignore[override]  # noqa: A002,E501
        raise RuntimeError("missing api_key")
        yield  # pragma: no cover - 到達しない（型のため）


def _client_no_model() -> TestClient:
    """実行時 api_key エラーで model_not_configured を起こすエージェントの TestClient を作る。"""
    reg = AgentRegistry()
    reg.register(AgentSpec(name="bot", instructions="b", model=_ApiKeyErrorModel()))
    app = create_app(ConversationService(reg))
    return TestClient(app)


def test_list_agents() -> None:
    """GET /agents が登録名一覧とエントリエージェントを返す。"""
    client = _client()
    resp = client.get("/agents")
    assert resp.status_code == 200
    assert resp.json() == {"agents": ["bot"], "entry": "bot"}


def test_create_conversation() -> None:
    """POST /conversations が conversation_id を返す（201）。"""
    client = _client()
    resp = client.post("/conversations", json={})
    assert resp.status_code == 201
    assert resp.json()["conversation_id"]


def test_create_conversation_with_session_id(tmp_path: Any) -> None:
    """session_id 指定で会話作成できる（永続化先は tmp。cwd を汚さない）。"""
    client = _client_with_tmp_policy(tmp_path)
    resp = client.post("/conversations", json={"conversation_id": "c1", "session_id": "s1"})
    assert resp.status_code == 201
    assert resp.json()["conversation_id"] == "c1"


def _client_with_tmp_policy(tmp_path: Any, model: Model | None = None) -> TestClient:
    """tmp ディレクトリを永続化先にした app の TestClient を作る（D5 用）。"""
    reg = AgentRegistry()
    reg.register(AgentSpec(name="bot", instructions="b", model=model or StreamingFakeModel()))
    policy = SessionPolicy(base_dir=tmp_path, db_name="conversations.db")
    app = create_app(ConversationService(reg, session_policy=policy))
    return TestClient(app)


def test_list_sessions_empty(tmp_path: Any) -> None:
    """永続化会話が無ければ GET /sessions は空。"""
    client = _client_with_tmp_policy(tmp_path)
    resp = client.get("/sessions")
    assert resp.status_code == 200
    assert resp.json() == {"sessions": []}


def test_list_sessions_after_persisted_conversation(tmp_path: Any) -> None:
    """session_id 明示の会話後に GET /sessions が当該 session をメタ付きで列挙する（D5）。"""
    client = _client_with_tmp_policy(tmp_path, StreamingFakeModel().queue_text("ok"))
    client.post("/conversations", json={"conversation_id": "c1", "session_id": "named-1"})
    client.post("/conversations/c1/messages", json={"agent_name": "bot", "text": "hi"})
    resp = client.get("/sessions")
    assert resp.status_code == 200
    sessions = resp.json()["sessions"]
    ids = [s["session_id"] for s in sessions]
    assert "named-1" in ids
    named = next(s for s in sessions if s["session_id"] == "named-1")
    assert named["turn_count"] == 1
    assert named["preview"] == "hi"


def test_session_history_returns_items(tmp_path: Any) -> None:
    """GET /sessions/{id}/history が過去履歴アイテムを時系列で返す（D5・復元表示用）。"""
    client = _client_with_tmp_policy(tmp_path, StreamingFakeModel().queue_text("やあ"))
    client.post("/conversations", json={"conversation_id": "c1", "session_id": "hist-1"})
    client.post("/conversations/c1/messages", json={"agent_name": "bot", "text": "hello"})
    resp = client.get("/sessions/hist-1/history")
    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"] == "hist-1"
    roles = [item.get("role") for item in body["items"]]
    assert "user" in roles
    assert "assistant" in roles


def test_send_uses_entry_agent_when_omitted(tmp_path: Any) -> None:
    """agent_name 省略時はエントリエージェント起点で会話する。"""
    client = _client_with_tmp_policy(tmp_path, StreamingFakeModel().queue_text("entry ok"))
    cid = client.post("/conversations", json={}).json()["conversation_id"]
    resp = client.post(f"/conversations/{cid}/messages", json={"text": "hi"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["output"] == "entry ok"
    assert body["status"] == "final"
    assert body["pending"] is None


def test_send_non_streaming() -> None:
    """POST /conversations/{id}/messages が最終応答を返す。"""
    client = _client(StreamingFakeModel().queue_text("hi there"))
    cid = client.post("/conversations", json={}).json()["conversation_id"]
    resp = client.post(
        f"/conversations/{cid}/messages",
        json={"agent_name": "bot", "text": "hello"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["output"] == "hi there"
    assert body["status"] == "final"
    assert body["pending"] is None


def test_send_unknown_agent_returns_404() -> None:
    """不正エージェント名は 404 + 構造化エラーボディを返す。"""
    client = _client()
    cid = client.post("/conversations", json={}).json()["conversation_id"]
    resp = client.post(
        f"/conversations/{cid}/messages",
        json={"agent_name": "nope", "text": "x"},
    )
    assert resp.status_code == 404
    body = resp.json()
    assert body["code"] == "unknown_agent"
    assert body["message"]


def test_send_unknown_conversation_returns_404() -> None:
    """不正 conversation_id は 404 + 構造化エラーボディを返す。"""
    client = _client()
    resp = client.post(
        "/conversations/missing/messages",
        json={"agent_name": "bot", "text": "x"},
    )
    assert resp.status_code == 404
    assert resp.json()["code"] == "unknown_conversation"


def test_ws_streaming_token_then_done() -> None:
    """WebSocket で turn を送ると token が逐次流れ done で終端する。"""
    client = _client(StreamingFakeModel().queue_text("streaming output text"))
    cid = client.post("/conversations", json={}).json()["conversation_id"]
    tokens: list[str] = []
    done: dict[str, Any] | None = None
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "turn", "agent_name": "bot", "conversation_id": cid, "text": "go"})
        while True:
            msg = ws.receive_json()
            if msg["type"] == "token":
                tokens.append(msg["text"])
            elif msg["type"] == "done":
                done = msg
                break
    assert "".join(tokens) == "streaming output text"
    assert done is not None
    assert done["output"] == "streaming output text"


def test_ws_unknown_agent_sends_error() -> None:
    """WebSocket で不正エージェント名を送ると error メッセージが返る。"""
    client = _client()
    cid = client.post("/conversations", json={}).json()["conversation_id"]
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "turn", "agent_name": "nope", "conversation_id": cid, "text": "x"})
        msg = ws.receive_json()
    assert msg["type"] == "error"
    assert msg["code"] == "unknown_agent"


def test_ws_unknown_conversation_sends_error() -> None:
    """WebSocket で不正 conversation_id を送ると error メッセージが返る。"""
    client = _client()
    with client.websocket_connect("/ws") as ws:
        ws.send_json(
            {"type": "turn", "agent_name": "bot", "conversation_id": "missing", "text": "x"}
        )
        msg = ws.receive_json()
    assert msg["type"] == "error"
    assert msg["code"] == "unknown_conversation"


def test_ws_unsupported_message_type_sends_error() -> None:
    """未対応の WS メッセージ種別は error メッセージで返す。"""
    client = _client()
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "bogus"})
        msg = ws.receive_json()
    assert msg["type"] == "error"


def test_send_model_not_configured_returns_503() -> None:
    """モデル未注入の会話送信は 503 + model_not_configured を返す。"""
    client = _client_no_model()
    cid = client.post("/conversations", json={}).json()["conversation_id"]
    resp = client.post(
        f"/conversations/{cid}/messages",
        json={"agent_name": "bot", "text": "hi"},
    )
    assert resp.status_code == 503
    assert resp.json()["code"] == "model_not_configured"


def test_create_duplicate_conversation_returns_409() -> None:
    """同一 conversation_id の重複作成は 409 + conversation_already_exists を返す。"""
    client = _client()
    client.post("/conversations", json={"conversation_id": "dup"})
    resp = client.post("/conversations", json={"conversation_id": "dup"})
    assert resp.status_code == 409
    assert resp.json()["code"] == "conversation_already_exists"


def test_ws_model_not_configured_sends_error() -> None:
    """WS のモデル未注入は error(code=model_not_configured) を返す。"""
    client = _client_no_model()
    cid = client.post("/conversations", json={}).json()["conversation_id"]
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "turn", "agent_name": "bot", "conversation_id": cid, "text": "x"})
        msg = ws.receive_json()
    assert msg["type"] == "error"
    assert msg["code"] == "model_not_configured"


def test_ws_non_dict_message_sends_error() -> None:
    """非 dict の JSON メッセージは AttributeError で抜けず error で返す。"""
    client = _client()
    with client.websocket_connect("/ws") as ws:
        ws.send_json(["not", "a", "dict"])
        msg = ws.receive_json()
    assert msg["type"] == "error"


def test_ws_invalid_json_sends_error() -> None:
    """不正 JSON テキストは error で返して閉じる（クラッシュしない）。"""
    client = _client()
    with client.websocket_connect("/ws") as ws:
        ws.send_text("{ this is not json")
        msg = ws.receive_json()
    assert msg["type"] == "error"
