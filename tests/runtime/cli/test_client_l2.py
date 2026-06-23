"""L2: CLI クライアントの REST 部分を段階2 FastAPI app に対して検証する。

実 uvicorn は起動せず、httpx の ASGI トランスポートで段階2 の app へ in-process 接続して
agents 一覧 / 会話作成 / 非ストリーム送信・構造化エラーを確認する。WS は段階2 で検証済みの
ため本段階では最小（クライアント側の URL 導出と error 変換のユニット）に留める。
httpx / websockets / fastapi 未導入環境では importorskip でスキップする。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

pytest.importorskip("httpx")
pytest.importorskip("websockets")
pytest.importorskip("fastapi")

import httpx  # noqa: E402
from agents.items import ModelResponse  # noqa: E402
from agents.models.interface import Model  # noqa: E402

from oai_agentspec import AgentRegistry, AgentSpec  # noqa: E402
from oai_agentspec.runtime.cli.client import (  # noqa: E402
    ConversationClient,
    ConversationClientError,
    _ws_url,
)
from oai_agentspec.runtime.conversation import ConversationService, SessionPolicy  # noqa: E402
from oai_agentspec.runtime.serve import create_app  # noqa: E402

from _helpers.responses import text_response  # noqa: E402

pytestmark = pytest.mark.integration


class _FakeModel(Model):
    """カンネドテキストを返すだけの Model（ストリーミングは未使用）。"""

    def __init__(self, text: str = "hi") -> None:
        """応答テキストを設定する。"""
        self._text = text

    async def get_response(
        self,
        system_instructions: str | None = None,
        input: Any = None,  # noqa: A002 - SDK シグネチャに追従
        *args: Any,
        **kwargs: Any,
    ) -> ModelResponse:
        """設定済みテキストを返す。"""
        return text_response(self._text)

    async def stream_response(  # type: ignore[override]
        self,
        *args: Any,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """未使用（REST テストのみ）。"""
        raise NotImplementedError
        yield  # pragma: no cover


def _app() -> Any:
    """1 つの FakeModel エージェントを登録した段階2 app を作る。"""
    reg = AgentRegistry()
    reg.register(AgentSpec(name="bot", instructions="b", model=_FakeModel("hi there")))
    return create_app(ConversationService(reg))


def _client_for(app: Any) -> ConversationClient:
    """段階2 app へ in-process 接続する CLI クライアントを作る。"""
    client = ConversationClient("http://testserver")
    # 内部 httpx クライアントを ASGI トランスポートへ差し替え（実ネットワーク不使用）。
    client._client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    )
    return client


@pytest.mark.asyncio
async def test_list_agents() -> None:
    """list_agents がサーバの登録名一覧を返す。"""
    async with _client_for(_app()) as client:
        assert await client.list_agents() == ["bot"]


@pytest.mark.asyncio
async def test_list_sessions(tmp_path: Any) -> None:
    """list_sessions が永続化済み session を返す（D5・tmp 永続化先）。"""
    reg = AgentRegistry()
    reg.register(AgentSpec(name="bot", instructions="b", model=_FakeModel("ok")))
    policy = SessionPolicy(base_dir=tmp_path, db_name="conversations.db")
    app = create_app(ConversationService(reg, session_policy=policy))
    async with _client_for(app) as client:
        # 初期は空。
        assert await client.list_sessions() == []
        # session_id 明示の会話で 1 ターン送ると永続化され列挙対象になる。
        cid = await client.create_conversation(session_id="named-1")
        await client.send("bot", "hi", conversation_id=cid)
        sessions = await client.list_sessions()
        assert "named-1" in [s.session_id for s in sessions]


@pytest.mark.asyncio
async def test_get_entry() -> None:
    """get_entry がサーバのエントリエージェント名を返す。"""
    async with _client_for(_app()) as client:
        assert await client.get_entry() == "bot"


@pytest.mark.asyncio
async def test_get_history_returns_items(tmp_path: Any) -> None:
    """get_history が復元対象 session の履歴アイテムを返す（D5）。"""
    reg = AgentRegistry()
    reg.register(AgentSpec(name="bot", instructions="b", model=_FakeModel("やあ")))
    policy = SessionPolicy(base_dir=tmp_path, db_name="conversations.db")
    app = create_app(ConversationService(reg, session_policy=policy))
    async with _client_for(app) as client:
        cid = await client.create_conversation(session_id="hist-1")
        await client.send("bot", "hello", conversation_id=cid)
        items = await client.get_history("hist-1", limit=10)
        roles = [item.get("role") for item in items]
        assert "user" in roles
        assert "assistant" in roles


@pytest.mark.asyncio
async def test_create_conversation() -> None:
    """create_conversation が conversation_id を返す。"""
    async with _client_for(_app()) as client:
        cid = await client.create_conversation()
        assert cid


@pytest.mark.asyncio
async def test_create_conversation_with_session_id(tmp_path: Any) -> None:
    """session_id 指定の会話作成ができる（永続化先は tmp。cwd を汚さない）。"""
    reg = AgentRegistry()
    reg.register(AgentSpec(name="bot", instructions="b", model=_FakeModel("hi")))
    policy = SessionPolicy(base_dir=tmp_path, db_name="conversations.db")
    app = create_app(ConversationService(reg, session_policy=policy))
    async with _client_for(app) as client:
        cid = await client.create_conversation(session_id="s1")
        assert cid


@pytest.mark.asyncio
async def test_send_non_streaming() -> None:
    """send が最終応答テキストを返す。"""
    async with _client_for(_app()) as client:
        cid = await client.create_conversation()
        out = await client.send("bot", "hello", conversation_id=cid)
        assert out.status == "final"
        assert out.output == "hi there"


@pytest.mark.asyncio
async def test_send_unknown_agent_raises_structured_error() -> None:
    """不正エージェント名はサーバ 404 を構造化エラーへ変換する。"""
    async with _client_for(_app()) as client:
        cid = await client.create_conversation()
        with pytest.raises(ConversationClientError) as exc:
            await client.send("nope", "x", conversation_id=cid)
        assert exc.value.code == "unknown_agent"


@pytest.mark.asyncio
async def test_send_unknown_conversation_raises_structured_error() -> None:
    """不正 conversation_id はサーバ 404 を構造化エラーへ変換する。"""
    async with _client_for(_app()) as client:
        with pytest.raises(ConversationClientError) as exc:
            await client.send("bot", "x", conversation_id="missing")
        assert exc.value.code == "unknown_conversation"


@pytest.mark.asyncio
async def test_connection_error_is_friendly() -> None:
    """サーバ未起動（接続不能）は分かりやすいエラーへ変換する。"""
    # 実在しないポートへ接続させて接続失敗を誘発する。
    async with ConversationClient("http://127.0.0.1:0") as client:
        with pytest.raises(ConversationClientError) as exc:
            await client.list_agents()
        assert "接続に失敗" in exc.value.message


def test_ws_url_derivation() -> None:
    """REST base_url から WS URL（ws(s)://.../ws）を導出する。"""
    assert _ws_url("http://localhost:8000") == "ws://localhost:8000/ws"
    assert _ws_url("https://example.com/") == "wss://example.com/ws"
