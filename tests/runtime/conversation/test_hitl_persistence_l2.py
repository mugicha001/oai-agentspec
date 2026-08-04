"""L2: HITL 承認待ちの永続化と別インスタンスからの復元（FR-10・方針B）。

tmp の SessionPolicy で session_id を明示し永続会話にする。承認必須ツールで中断 →
conversations.db の専用テーブル（oai_agentspec_pending_approvals・主キー session_id）へ保存 → 別の
ConversationService インスタンス（サーバ再起動相当）で**同 session_id・新 conversation_id** を
作り pending_approvals すると復元され承認待ちが復活すること（実際の復元フロー）を確認する。
非エントリの明示エージェントが作った承認待ちが正しいエージェントで再開されること、揮発
（persist=False / session_id 無指定）では復元されないことも確認する。

実 LLM は呼ばない。承認/却下による再開（resolve）の経路は別ファイルで検証する。本ファイルは
保存・復元（中断状態の往復・正しいエージェント復元・揮発の非復元）に焦点を当てる。
"""

from __future__ import annotations

from typing import Any

import pytest

from oai_agentspec import AgentRegistry, AgentSpec
from oai_agentspec._adapters import list_pending_approvals
from oai_agentspec.runtime.conversation import (
    ConversationService,
    PendingApproval,
    SessionPolicy,
)
from oai_agentspec.runtime.deterministic import text_response, tool_call_response

from _helpers.approval import QueuedFakeModel, ToolRecorder, make_approval_tool

pytestmark = pytest.mark.integration


def _registry(model: QueuedFakeModel, tool: Any) -> AgentRegistry:
    """承認必須ツールを載せた FakeModel エージェントの registry を作る。"""
    reg = AgentRegistry()
    reg.register(AgentSpec(name="bot", instructions="b", model=model, tools=[tool]))
    return reg


@pytest.mark.asyncio
async def test_pending_persisted_to_db_on_interruption(tmp_path: Any) -> None:
    """永続会話で中断すると専用テーブルへ承認待ちが保存される（FR-10・方針B・session_id キー）。"""
    recorder = ToolRecorder()
    tool = make_approval_tool(recorder, name="danger")
    model = QueuedFakeModel().queue(tool_call_response("danger", '{"x": "v"}', call_id="c1"))
    policy = SessionPolicy(base_dir=tmp_path, db_name="conversations.db")
    svc = ConversationService(_registry(model, tool), session_policy=policy)
    cid = await svc.create_conversation(conversation_id="conv-1", session_id="sess-1")

    result = await svc.send("bot", "go", conversation_id=cid)
    assert result.status == "pending"

    # conversations.db の専用テーブルに session_id キーで中断状態が保存される。
    rows = list_pending_approvals(str(tmp_path / "conversations.db"))
    saved = [r for r in rows if r["session_id"] == "sess-1"]
    assert len(saved) == 1
    assert saved[0]["agent_name"] == "bot"


@pytest.mark.asyncio
async def test_restore_pending_in_new_service_instance(tmp_path: Any) -> None:
    """別インスタンス（再起動相当）で同 session_id・新 conversation_id の承認待ちが復元される。

    CLI の実復元フローと同じく、再起動後は同 session_id に新しい conversation_id を振って復元する。
    承認待ちは session_id キーで保存されるため、新 conversation_id でも pending_approvals で復活し、
    resolve_approvals で再開できる（FR-10）。
    """
    recorder = ToolRecorder()
    tool = make_approval_tool(recorder, name="danger")
    model = QueuedFakeModel().queue(tool_call_response("danger", '{"x": "v"}', call_id="c1"))
    policy = SessionPolicy(base_dir=tmp_path, db_name="conversations.db")
    svc1 = ConversationService(_registry(model, tool), session_policy=policy)
    await svc1.create_conversation(conversation_id="conv-old", session_id="sess-1")
    await svc1.send("bot", "go", conversation_id="conv-old")

    # 新しいサービス（再起動相当）で同 session_id に新 conversation_id を振って復元する。
    recorder2 = ToolRecorder()
    model2 = QueuedFakeModel().queue(text_response("resumed"))
    svc2 = ConversationService(
        _registry(model2, make_approval_tool(recorder2, name="danger")),
        session_policy=policy,
    )
    await svc2.create_conversation(conversation_id="conv-new", session_id="sess-1")

    # 同 session_id の承認待ちが新 conversation_id 上で復元される。
    restored = await svc2.pending_approvals("conv-new")
    assert restored == [PendingApproval(tool_name="danger", call_id="c1")]

    # 復元した承認待ちを approve すると再開してツールが実行され final が返る（FR-10 再開）。
    result = await svc2.resolve_approvals("conv-new", [{"call_id": "c1", "decision": "approve"}])
    assert result.status == "final"
    assert recorder2.executed == ["v"]


@pytest.mark.asyncio
async def test_restore_uses_persisted_agent_name_not_entry(tmp_path: Any) -> None:
    """非エントリの明示エージェントが作った承認待ちが、復元時も正しいエージェントで再開される。

    registry に 2 体（エントリ=alpha / 非エントリ=beta）を登録。beta のツールで中断 → 別インスタンス
    で同 session_id・新 conversation_id を作り復元する。新 entry.agent_name は None でエントリ
    （alpha）に倒れうるが、永続レコードの agent_name（beta）が initial_agent に使われ beta のツール
    が実行されることを確認する（P2-agent）。
    """
    rec_alpha = ToolRecorder()
    rec_beta = ToolRecorder()
    model = QueuedFakeModel().queue(tool_call_response("beta_danger", '{"x": "v"}', call_id="c1"))
    reg = AgentRegistry()
    reg.register(
        AgentSpec(
            name="alpha",
            instructions="a",
            model=QueuedFakeModel(),
            tools=[make_approval_tool(rec_alpha, name="alpha_danger")],
        )
    )
    reg.register(
        AgentSpec(
            name="beta",
            instructions="b",
            model=model,
            tools=[make_approval_tool(rec_beta, name="beta_danger")],
        )
    )
    policy = SessionPolicy(base_dir=tmp_path, db_name="conversations.db")
    svc1 = ConversationService(reg, session_policy=policy)
    await svc1.create_conversation(conversation_id="conv-old", session_id="sess-b")
    # 非エントリ（beta）の明示エージェントで中断を起こす。
    pending = await svc1.send("beta", "go", conversation_id="conv-old")
    assert pending.status == "pending"

    # 別インスタンス（再起動相当）で同 session_id・新 conversation_id を作り復元。
    rec_alpha2 = ToolRecorder()
    rec_beta2 = ToolRecorder()
    reg2 = AgentRegistry()
    reg2.register(
        AgentSpec(
            name="alpha",
            instructions="a",
            model=QueuedFakeModel(),
            tools=[make_approval_tool(rec_alpha2, name="alpha_danger")],
        )
    )
    reg2.register(
        AgentSpec(
            name="beta",
            instructions="b",
            model=QueuedFakeModel().queue(text_response("resumed")),
            tools=[make_approval_tool(rec_beta2, name="beta_danger")],
        )
    )
    svc2 = ConversationService(reg2, session_policy=policy)
    await svc2.create_conversation(conversation_id="conv-new", session_id="sess-b")

    restored = await svc2.pending_approvals("conv-new")
    assert restored == [PendingApproval(tool_name="beta_danger", call_id="c1")]

    # 復元・再開で beta のツールが実行され、エントリ（alpha）のツールは実行されない。
    result = await svc2.resolve_approvals("conv-new", [{"call_id": "c1", "decision": "approve"}])
    assert result.status == "final"
    assert rec_beta2.executed == ["v"]
    assert rec_alpha2.executed == []


@pytest.mark.asyncio
async def test_ephemeral_conversation_not_persisted(tmp_path: Any) -> None:
    """揮発会話（session_id 無指定）は専用テーブルへ保存されず復元されない（FR-10）。"""
    recorder = ToolRecorder()
    tool = make_approval_tool(recorder, name="danger")
    model = QueuedFakeModel().queue(tool_call_response("danger", '{"x": "v"}', call_id="c1"))
    policy = SessionPolicy(base_dir=tmp_path, db_name="conversations.db")
    svc = ConversationService(_registry(model, tool), session_policy=policy)
    # session_id 無指定 = 揮発（persist=False）。
    cid = await svc.create_conversation(conversation_id="conv-vol")
    result = await svc.send("bot", "go", conversation_id=cid)
    assert result.status == "pending"

    # 専用テーブルには書かれない（メモリ保持のみ）。揮発会話の session_id は conversation_id。
    rows = list_pending_approvals(str(tmp_path / "conversations.db"))
    assert all(r["session_id"] != "conv-vol" for r in rows)


@pytest.mark.asyncio
async def test_no_interruption_no_persisted_pending(tmp_path: Any) -> None:
    """承認待ちが無い永続会話では専用テーブルに中断状態が残らない（NFR-6 回帰）。"""
    recorder = ToolRecorder()
    tool = make_approval_tool(recorder, name="danger")
    model = QueuedFakeModel().queue(text_response("done"))
    policy = SessionPolicy(base_dir=tmp_path, db_name="conversations.db")
    svc = ConversationService(_registry(model, tool), session_policy=policy)
    await svc.create_conversation(conversation_id="conv-1", session_id="sess-1")
    await svc.send("bot", "hi", conversation_id="conv-1")

    rows = list_pending_approvals(str(tmp_path / "conversations.db"))
    assert all(r["session_id"] != "sess-1" for r in rows)
