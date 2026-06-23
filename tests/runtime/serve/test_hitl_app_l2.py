"""L2: FastAPI 会話サーバの HITL（承認）を TestClient + FakeModel で検証する。

承認必須ツールを載せたエージェントで、REST の messages 応答が status="pending" + 承認待ち一覧を
返すこと（FR-3）、GET /approvals で承認待ちを取得できること（D-RestGet）、POST /approvals で
承認/却下できること（D-WsMsg）、承認系エラーの HTTP status（404/409）、decision の Literal
バリデーション（422）、WS で approval_required → approval → token → done の往復（FR-2/FR-8）を
確認する。実 LLM は呼ばない。
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from oai_agentspec import AgentRegistry, AgentSpec  # noqa: E402
from oai_agentspec.runtime.conversation import ConversationService, SessionPolicy  # noqa: E402
from oai_agentspec.runtime.serve import create_app  # noqa: E402

from _helpers.approval import QueuedFakeModel, ToolRecorder, make_approval_tool  # noqa: E402
from _helpers.responses import text_response, tool_call_response  # noqa: E402

pytestmark = pytest.mark.integration


def _client(model: QueuedFakeModel, tool: Any, tmp_path: Any) -> TestClient:
    """承認必須ツールを載せた app の TestClient を作る（永続化先は tmp）。"""
    reg = AgentRegistry()
    reg.register(AgentSpec(name="bot", instructions="b", model=model, tools=[tool]))
    policy = SessionPolicy(base_dir=tmp_path, db_name="conversations.db")
    return TestClient(create_app(ConversationService(reg, session_policy=policy)))


def _pending_client(tmp_path: Any, recorder: ToolRecorder) -> tuple[TestClient, str]:
    """承認待ちを発生させた状態の (client, conversation_id) を返す。"""
    tool = make_approval_tool(recorder, name="danger")
    model = (
        QueuedFakeModel()
        .queue(tool_call_response("danger", '{"x": "v"}', call_id="c1"))
        .queue(text_response("resumed"))
    )
    client = _client(model, tool, tmp_path)
    cid = client.post("/conversations", json={"conversation_id": "c1", "session_id": "s1"}).json()[
        "conversation_id"
    ]
    resp = client.post(f"/conversations/{cid}/messages", json={"text": "go"})
    assert resp.status_code == 200
    return client, cid


# ----------------------------------------------------------------------
# REST: 承認待ち応答 / 取得 / エラー
# ----------------------------------------------------------------------
def test_send_pending_body_shape(tmp_path: Any) -> None:
    """messages 応答そのものが status="pending" / output=None / pending 一覧を持つ（D-Disc）。"""
    recorder = ToolRecorder()
    tool = make_approval_tool(recorder, name="danger")
    model = QueuedFakeModel().queue(tool_call_response("danger", '{"x": "v"}', call_id="c1"))
    client = _client(model, tool, tmp_path)
    cid = client.post("/conversations", json={"conversation_id": "c1", "session_id": "s1"}).json()[
        "conversation_id"
    ]
    resp = client.post(f"/conversations/{cid}/messages", json={"text": "go"})
    body = resp.json()
    assert body["status"] == "pending"
    assert body["output"] is None
    assert body["pending"] == [{"tool_name": "danger", "call_id": "c1"}]


def test_get_approvals_returns_pending(tmp_path: Any) -> None:
    """GET /conversations/{id}/approvals が承認待ち一覧を返す（D-RestGet・冪等）。"""
    recorder = ToolRecorder()
    client, cid = _pending_client(tmp_path, recorder)
    resp = client.get(f"/conversations/{cid}/approvals")
    assert resp.status_code == 200
    assert resp.json()["pending"] == [{"tool_name": "danger", "call_id": "c1"}]


def test_resolve_unknown_call_id_returns_404(tmp_path: Any) -> None:
    """未知 call_id の approve は 404 + unknown_approval（D-Err）。"""
    recorder = ToolRecorder()
    client, cid = _pending_client(tmp_path, recorder)
    resp = client.post(
        f"/conversations/{cid}/approvals",
        json={"decisions": [{"call_id": "ghost", "decision": "approve"}]},
    )
    assert resp.status_code == 404
    assert resp.json()["code"] == "unknown_approval"


def test_resolve_without_pending_returns_409(tmp_path: Any) -> None:
    """承認待ちが無い会話への approve は 409 + no_pending_approval（D-Err）。"""
    recorder = ToolRecorder()
    tool = make_approval_tool(recorder, name="danger")
    model = QueuedFakeModel().queue(text_response("done"))
    client = _client(model, tool, tmp_path)
    cid = client.post("/conversations", json={}).json()["conversation_id"]
    client.post(f"/conversations/{cid}/messages", json={"text": "hi"})
    resp = client.post(
        f"/conversations/{cid}/approvals",
        json={"decisions": [{"call_id": "x", "decision": "approve"}]},
    )
    assert resp.status_code == 409
    assert resp.json()["code"] == "no_pending_approval"


def test_resolve_invalid_decision_returns_422(tmp_path: Any) -> None:
    """decision が Literal 外（不正値）なら 422（Pydantic バリデーション・fail-closed）。"""
    recorder = ToolRecorder()
    client, cid = _pending_client(tmp_path, recorder)
    resp = client.post(
        f"/conversations/{cid}/approvals",
        json={"decisions": [{"call_id": "c1", "decision": "maybe"}]},
    )
    assert resp.status_code == 422


def test_get_approvals_empty_when_no_pending(tmp_path: Any) -> None:
    """承認待ちが無い会話の GET /approvals は空一覧（NFR-6・回帰）。"""
    recorder = ToolRecorder()
    tool = make_approval_tool(recorder, name="danger")
    model = QueuedFakeModel().queue(text_response("done"))
    client = _client(model, tool, tmp_path)
    cid = client.post("/conversations", json={}).json()["conversation_id"]
    client.post(f"/conversations/{cid}/messages", json={"text": "hi"})
    resp = client.get(f"/conversations/{cid}/approvals")
    assert resp.status_code == 200
    assert resp.json()["pending"] == []


def test_resolve_approve_returns_final(tmp_path: Any) -> None:
    """approve で再開し status="final" + 最終応答・ツール実行（FR-6・NFR-7）。"""
    recorder = ToolRecorder()
    client, cid = _pending_client(tmp_path, recorder)
    resp = client.post(
        f"/conversations/{cid}/approvals",
        json={"decisions": [{"call_id": "c1", "decision": "approve"}]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "final"
    assert body["output"] == "resumed"
    assert recorder.executed == ["v"]


def test_resolve_reject_returns_final_without_execution(tmp_path: Any) -> None:
    """reject で再開し status="final"・ツール未実行（FR-5・NFR-7）。"""
    recorder = ToolRecorder()
    client, cid = _pending_client(tmp_path, recorder)
    resp = client.post(
        f"/conversations/{cid}/approvals",
        json={"decisions": [{"call_id": "c1", "decision": "reject"}]},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "final"
    assert recorder.executed == []


# ----------------------------------------------------------------------
# WS: turn → approval_required（done なし）→ approval → token → done
# ----------------------------------------------------------------------
def test_ws_turn_yields_approval_required_without_done(tmp_path: Any) -> None:
    """WS の承認必須ターンは approval_required を送り done を送らない（FR-2）。"""
    recorder = ToolRecorder()
    tool = make_approval_tool(recorder, name="danger")
    model = QueuedFakeModel().queue(tool_call_response("danger", '{"x": "v"}', call_id="c1"))
    client = _client(model, tool, tmp_path)
    cid = client.post("/conversations", json={"conversation_id": "c1", "session_id": "s1"}).json()[
        "conversation_id"
    ]

    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "turn", "conversation_id": cid, "text": "go"})
        msg = ws.receive_json()
        # 承認待ちターンの先頭メッセージは approval_required（done ではない）。
        assert msg["type"] == "approval_required"
        assert msg["pending"] == [{"tool_name": "danger", "call_id": "c1"}]
    assert recorder.executed == []


def test_ws_approval_then_token_done(tmp_path: Any) -> None:
    """WS で approval_required → approval(approve) → token → done と再開する（FR-2/FR-8）。"""
    recorder = ToolRecorder()
    tool = make_approval_tool(recorder, name="danger")
    model = (
        QueuedFakeModel()
        .queue(tool_call_response("danger", '{"x": "v"}', call_id="c1"))
        .queue(text_response("ws resumed"))
    )
    client = _client(model, tool, tmp_path)
    cid = client.post("/conversations", json={"conversation_id": "c1", "session_id": "s1"}).json()[
        "conversation_id"
    ]

    done: dict[str, Any] | None = None
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "turn", "conversation_id": cid, "text": "go"})
        first = ws.receive_json()
        assert first["type"] == "approval_required"
        ws.send_json(
            {
                "type": "approval",
                "decisions": [{"call_id": "c1", "decision": "approve"}],
            }
        )
        while True:
            msg = ws.receive_json()
            if msg["type"] == "done":
                done = msg
                break
    assert done is not None
    assert recorder.executed == ["v"]
