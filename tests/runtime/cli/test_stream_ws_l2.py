"""L2: CLI クライアントの WebSocket ストリーミングを実サーバへ接続して検証する。

段階2 の FastAPI app を loopback 上の uvicorn で短時間起動し、`ConversationClient.stream`
が token を逐次 yield し done で終端すること・サーバ error を構造化エラーへ変換することを
確認する。conftest のネットワークガードは loopback（127.0.0.1）を許可するため実通信できる。
httpx / websockets / uvicorn / fastapi 未導入環境では importorskip でスキップする。
"""

from __future__ import annotations

import asyncio
import socket
from collections.abc import AsyncIterator
from typing import Any

import pytest

pytest.importorskip("httpx")
pytest.importorskip("websockets")
pytest.importorskip("uvicorn")
pytest.importorskip("fastapi")

import uvicorn  # noqa: E402
from agents.items import ModelResponse  # noqa: E402
from agents.models.interface import Model  # noqa: E402

from oai_agentspec import AgentRegistry, AgentSpec  # noqa: E402
from oai_agentspec.runtime.cli.client import (  # noqa: E402
    ConversationClient,
    ConversationClientError,
    StreamDone,
    StreamToken,
)
from oai_agentspec.runtime.conversation import ConversationService  # noqa: E402
from oai_agentspec.runtime.deterministic import text_response  # noqa: E402
from oai_agentspec.runtime.serve import create_app  # noqa: E402

pytestmark = pytest.mark.integration


class _StreamingFakeModel(Model):
    """最終テキストを delta + completed で流す Model（run_streamed 対応）。"""

    def __init__(self, text: str) -> None:
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
        system_instructions: str | None = None,
        input: Any = None,  # noqa: A002 - SDK シグネチャに追従
        *args: Any,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """delta + completed イベントを流す。"""
        from oai_agentspec._adapters import _completed_event, _text_delta_events, _text_of

        response = await self.get_response(system_instructions, input, *args, **kwargs)
        text = _text_of(response)
        seq = 0
        for event in _text_delta_events(text, item_id="msg_stream_fake"):
            yield event
            seq = event.sequence_number + 1
        yield _completed_event(
            response.output, seq, response_id="resp_stream_fake", model="stream-fake"
        )


def _free_port() -> int:
    """loopback の空きポートを 1 つ確保して返す。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _build_app() -> Any:
    """ストリーミング対応エージェントを登録した段階2 app を作る。"""
    reg = AgentRegistry()
    reg.register(AgentSpec(name="bot", instructions="b", model=_StreamingFakeModel("hello world")))
    return create_app(ConversationService(reg))


async def _serve(app: Any, port: int) -> tuple[uvicorn.Server, asyncio.Task[None]]:
    """uvicorn サーバを起動し、起動完了まで待ってから (server, task) を返す。"""
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    for _ in range(100):
        if server.started:
            break
        await asyncio.sleep(0.02)
    return server, task


@pytest.mark.asyncio
async def test_client_stream_tokens_then_done() -> None:
    """ConversationClient.stream が token を逐次 yield し done で終端する。"""
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    server, task = await _serve(_build_app(), port)
    try:
        async with ConversationClient(base_url) as client:
            cid = await client.create_conversation()
            tokens: list[str] = []
            done: StreamDone | None = None
            async for event in client.stream("bot", "go", conversation_id=cid):
                if isinstance(event, StreamToken):
                    tokens.append(event.text)
                elif isinstance(event, StreamDone):
                    done = event
            assert "".join(tokens) == "hello world"
            assert done is not None
            assert done.output == "hello world"
    finally:
        server.should_exit = True
        await task


@pytest.mark.asyncio
async def test_client_stream_unknown_agent_raises() -> None:
    """サーバ error メッセージは ConversationClientError に変換される。"""
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    server, task = await _serve(_build_app(), port)
    try:
        async with ConversationClient(base_url) as client:
            cid = await client.create_conversation()
            with pytest.raises(ConversationClientError) as exc:
                async for _ in client.stream("nope", "x", conversation_id=cid):
                    pass
            assert exc.value.code == "unknown_agent"
    finally:
        server.should_exit = True
        await task
