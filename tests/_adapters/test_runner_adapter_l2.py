"""L2: DefaultRunnerAdapter（runner シーム本番実装）を実 Agent + FakeModel で検証する。

registry による AGENT 名解決、session / max_turns / run_config の Runner.run への
受け渡しを確認する（NFR-7・SDK 結合点の局在化）。
"""

from __future__ import annotations

import pytest
from agents import RunConfig, Runner, SQLiteSession

from oai_agentspec import AgentRegistry, AgentSpec
from oai_agentspec._adapters import DefaultRunnerAdapter

from _helpers.fake_model import FakeModel

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_runner_adapter_resolves_agent_name_via_registry() -> None:
    """registry 非 None のとき agent 名を SDK Agent へ解決して Runner.run する。"""
    model = FakeModel().queue_text("resolved-out")
    reg = AgentRegistry()
    reg.register(AgentSpec(name="worker", instructions="w", model=model))

    adapter = DefaultRunnerAdapter(reg)
    result = await adapter.run("worker", "in")
    # 名前 "worker" が registry で Agent へ解決され FakeModel が回った証跡。
    assert result.final_output == "resolved-out"
    assert len(model.calls) == 1


@pytest.mark.asyncio
async def test_runner_adapter_accepts_resolved_agent_when_no_registry() -> None:
    """registry None のとき agent はそのまま SDK Agent として扱われる。"""
    model = FakeModel().queue_text("direct-out")
    reg = AgentRegistry()
    reg.register(AgentSpec(name="a", instructions="a", model=model))
    agent = reg.get("a")

    adapter = DefaultRunnerAdapter(None)
    result = await adapter.run(agent, "x")
    assert result.final_output == "direct-out"


@pytest.mark.asyncio
async def test_runner_adapter_passes_session_and_max_turns(tmp_path: object) -> None:
    """session / max_turns / run_config が Runner.run へ渡される（例外なく実行できる）。"""
    model = FakeModel().queue_text("ok")
    reg = AgentRegistry()
    reg.register(AgentSpec(name="a", instructions="a", model=model))
    agent = reg.get("a")

    session = SQLiteSession("wf-test", ":memory:")
    run_config = RunConfig()
    adapter = DefaultRunnerAdapter(None)
    result = await adapter.run(agent, "hello", session=session, run_config=run_config, max_turns=3)
    assert result.final_output == "ok"
    # session に履歴が積まれている（session 経路が実際に通った証跡）。
    items = await session.get_items()
    assert items


@pytest.mark.asyncio
async def test_workflow_agent_node_session_via_facade() -> None:
    """session 指定の WorkflowGraph が AGENT ノードで session を使って実行できる（経路C）。"""
    from oai_agentspec._adapters import build_agent
    from oai_agentspec.workflow import END, START, WorkflowGraph

    model = FakeModel().queue_text("done")
    reg = AgentRegistry()
    reg.register(AgentSpec(name="step", instructions="s", model=model))

    session = SQLiteSession("wf-sess", ":memory:")
    wf = WorkflowGraph(name="wf_sess", run_defaults={"session": session})
    wf.add_agent_node("step", agent="step")
    wf.add_edge(START, "step")
    wf.add_edge("step", END)
    spec = wf.as_agent_spec("wf_sess_agent", registry=reg)
    agent = build_agent(spec)

    result = await Runner.run(agent, input="go")
    assert result.final_output == "done"
