"""L2: CLI クライアントの HITL（get_approvals / resolve_approvals / send pending / WS 承認往復）。

REST は httpx の ASGI トランスポートで段階2 app へ in-process 接続し、承認待ち取得・send の
SendResult 解釈・resolve のエラーを確認する。WS の承認往復（approval_handler 経由）は loopback の
uvicorn を短時間起動して実通信で確認する（既存 test_stream_ws_l2 のループパターン流用）。
実 LLM は呼ばない。
"""

from __future__ import annotations

import asyncio
import socket
from typing import Any

import pytest

pytest.importorskip("httpx")
pytest.importorskip("websockets")
pytest.importorskip("uvicorn")
pytest.importorskip("fastapi")

import httpx  # noqa: E402
import uvicorn  # noqa: E402

from oai_agentspec import AgentRegistry, AgentSpec  # noqa: E402
from oai_agentspec.runtime.cli.client import (  # noqa: E402
    ApprovalRequired,
    ConversationClient,
    ConversationClientError,
    PendingApproval,
    StreamDone,
)
from oai_agentspec.runtime.conversation import ConversationService, SessionPolicy  # noqa: E402
from oai_agentspec.runtime.deterministic import text_response, tool_call_response  # noqa: E402
from oai_agentspec.runtime.serve import create_app  # noqa: E402

from _helpers.approval import QueuedFakeModel, ToolRecorder, make_approval_tool  # noqa: E402

pytestmark = pytest.mark.integration


def _app(model: QueuedFakeModel, tool: Any, tmp_path: Any) -> Any:
    """承認必須ツールを載せた段階2 app を作る（永続化先は tmp）。"""
    reg = AgentRegistry()
    reg.register(AgentSpec(name="bot", instructions="b", model=model, tools=[tool]))
    policy = SessionPolicy(base_dir=tmp_path, db_name="conversations.db")
    return create_app(ConversationService(reg, session_policy=policy))


def _client_for(app: Any) -> ConversationClient:
    """段階2 app へ in-process 接続する CLI クライアントを作る。"""
    client = ConversationClient("http://testserver")
    client._client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    )
    return client


# ----------------------------------------------------------------------
# REST（in-process ASGI）
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_send_returns_pending_send_result(tmp_path: Any) -> None:
    """send が承認待ちを SendResult(status="pending") として解釈する（D-Disc）。"""
    recorder = ToolRecorder()
    tool = make_approval_tool(recorder, name="danger")
    model = QueuedFakeModel().queue(tool_call_response("danger", '{"x": "v"}', call_id="c1"))
    async with _client_for(_app(model, tool, tmp_path)) as client:
        cid = await client.create_conversation(session_id="s1")
        result = await client.send("bot", "go", conversation_id=cid)
        assert result.status == "pending"
        assert result.output is None
        assert result.pending == [PendingApproval(tool_name="danger", call_id="c1")]


@pytest.mark.asyncio
async def test_get_approvals_returns_pending(tmp_path: Any) -> None:
    """get_approvals が承認待ち一覧を返す（D-RestGet・冪等）。"""
    recorder = ToolRecorder()
    tool = make_approval_tool(recorder, name="danger")
    model = QueuedFakeModel().queue(tool_call_response("danger", '{"x": "v"}', call_id="c1"))
    async with _client_for(_app(model, tool, tmp_path)) as client:
        cid = await client.create_conversation(session_id="s1")
        await client.send("bot", "go", conversation_id=cid)
        pending = await client.get_approvals(cid)
        assert pending == [PendingApproval(tool_name="danger", call_id="c1")]


@pytest.mark.asyncio
async def test_resolve_unknown_call_id_raises_client_error(tmp_path: Any) -> None:
    """未知 call_id の resolve_approvals は構造化エラー（unknown_approval）になる。"""
    recorder = ToolRecorder()
    tool = make_approval_tool(recorder, name="danger")
    model = QueuedFakeModel().queue(tool_call_response("danger", '{"x": "v"}', call_id="c1"))
    async with _client_for(_app(model, tool, tmp_path)) as client:
        cid = await client.create_conversation(session_id="s1")
        await client.send("bot", "go", conversation_id=cid)
        with pytest.raises(ConversationClientError) as exc:
            await client.resolve_approvals(cid, [{"call_id": "ghost", "decision": "approve"}])
        assert exc.value.code == "unknown_approval"


@pytest.mark.asyncio
async def test_resolve_approve_returns_final(tmp_path: Any) -> None:
    """resolve_approvals(approve) で再開し SendResult(status="final") を返す（FR-6）。"""
    recorder = ToolRecorder()
    tool = make_approval_tool(recorder, name="danger")
    model = (
        QueuedFakeModel()
        .queue(tool_call_response("danger", '{"x": "v"}', call_id="c1"))
        .queue(text_response("resumed"))
    )
    async with _client_for(_app(model, tool, tmp_path)) as client:
        cid = await client.create_conversation(session_id="s1")
        await client.send("bot", "go", conversation_id=cid)
        result = await client.resolve_approvals(cid, [{"call_id": "c1", "decision": "approve"}])
        assert result.status == "final"
        assert result.output == "resumed"


# ----------------------------------------------------------------------
# WS（loopback uvicorn・承認往復）
# ----------------------------------------------------------------------
def _free_port() -> int:
    """loopback の空きポートを 1 つ確保して返す。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


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
async def test_ws_stream_yields_approval_required(tmp_path: Any) -> None:
    """WS stream が承認待ちを ApprovalRequired として yield する（handler なしで終端・FR-2）。"""
    recorder = ToolRecorder()
    tool = make_approval_tool(recorder, name="danger")
    model = QueuedFakeModel().queue(tool_call_response("danger", '{"x": "v"}', call_id="c1"))
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    server, task = await _serve(_app(model, tool, tmp_path), port)
    try:
        async with ConversationClient(base_url) as client:
            cid = await client.create_conversation(session_id="s1")
            events = [e async for e in client.stream("bot", "go", conversation_id=cid)]
            required = [e for e in events if isinstance(e, ApprovalRequired)]
            assert len(required) == 1
            assert required[0].pending == [PendingApproval(tool_name="danger", call_id="c1")]
            assert not any(isinstance(e, StreamDone) for e in events)
    finally:
        server.should_exit = True
        await task


@pytest.mark.asyncio
async def test_ws_approval_handler_roundtrip_resumes(tmp_path: Any) -> None:
    """WS approval_handler で approve を返すと再開し token→done を受け取る（FR-9・FR-2）。"""
    recorder = ToolRecorder()
    tool = make_approval_tool(recorder, name="danger")
    model = (
        QueuedFakeModel()
        .queue(tool_call_response("danger", '{"x": "v"}', call_id="c1"))
        .queue(text_response("ws resumed"))
    )
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    server, task = await _serve(_app(model, tool, tmp_path), port)

    async def _approve(pending: list[PendingApproval]) -> list[dict[str, Any]]:
        return [{"call_id": p.call_id, "decision": "approve"} for p in pending]

    try:
        async with ConversationClient(base_url) as client:
            cid = await client.create_conversation(session_id="s1")
            done: StreamDone | None = None
            async for event in client.stream(
                "bot", "go", conversation_id=cid, approval_handler=_approve
            ):
                if isinstance(event, StreamDone):
                    done = event
            assert done is not None
            assert done.output == "ws resumed"
            assert recorder.executed == ["v"]
    finally:
        server.should_exit = True
        await task
