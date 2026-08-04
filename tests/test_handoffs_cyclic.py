"""L2: FakeModel + Runner で循環ハンドオフの 2 ホップ実委譲とサブエージェントを検証。"""

from __future__ import annotations

import pytest
from agents import Runner

from oai_agentspec import AgentRegistry, AgentSpec, HandoffGraph
from oai_agentspec.runtime.deterministic import text_response, tool_call_response

from _helpers.fake_model import FakeModel


@pytest.mark.asyncio
async def test_cyclic_handoff_two_hop_delegation() -> None:
    # a が b へ handoff し、b が応答を返す 2 ホップ。
    model = FakeModel()
    model.responses.append(tool_call_response("transfer_to_b"))
    model.responses.append(text_response("handled by b"))

    reg = AgentRegistry()
    reg.register(AgentSpec(name="a", instructions="agent a", model=model, handoffs=["b"]))
    reg.register(AgentSpec(name="b", instructions="agent b", model=model, handoffs=["a"]))

    graph = HandoffGraph(entry="a")
    graph.edge("a", "b")
    graph.edge("b", "a")
    graph.apply(reg)

    a = reg.get("a")
    b = reg.get("b")
    # 構築時の identity（循環）
    assert b.handoffs[0] is a

    result = await Runner.run(a, input="please route")
    assert result.final_output == "handled by b"


@pytest.mark.asyncio
async def test_sub_agent_as_tool_returns_control() -> None:
    # orchestrator が researcher を as_tool 呼び出し → 結果を受けて自分で最終応答。
    model = FakeModel()
    model.responses.append(tool_call_response("researcher", arguments='{"input": "topic"}'))
    model.responses.append(text_response("final from orchestrator"))
    sub_model = FakeModel()
    sub_model.responses.append(text_response("research result"))

    reg = AgentRegistry()
    reg.register(AgentSpec(name="researcher", instructions="research", model=sub_model))
    reg.register(
        AgentSpec(
            name="orch",
            instructions="orchestrate",
            model=model,
            sub_agents=["researcher"],
        )
    )

    orch = reg.get("orch")
    assert orch.tools[0].name == "researcher"

    result = await Runner.run(orch, input="do research")
    assert result.final_output == "final from orchestrator"
    # サブが実際に呼ばれた（制御が戻ってメインが続行）
    assert sub_model.calls
