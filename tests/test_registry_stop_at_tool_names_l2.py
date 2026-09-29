"""L2: `AgentRegistry` 経由の `stop_at_tool_names` 検証（実 Agent・実 SDK 型）。

`get()` 経路で、`build_agent` の形状検査（ValueError・失敗時の構築キャッシュ巻き戻し）と、
`_wire` 末尾の名前突合（sub_agent の as_tool 名確定後・不一致は RuntimeWarning）を検証する。
MCP / Sandbox の突合除外、DI 差し替え builder 経路での形状検査の再実行、workflow facade の
非回帰もここで固定する。突合ヘルパ単体の分岐は `tests/_adapters/test_builders_l2.py` が担う。
"""

from __future__ import annotations

import warnings

import pytest
from agents import Agent, function_tool

from oai_agentspec import AgentRegistry, AgentSpec, FacadeMode, HandoffGraph
from oai_agentspec.spec import SandboxAgentSpec
from oai_agentspec.workflow import END, START, WorkflowGraph

from _helpers.fake_model import FakeModel

pytestmark = pytest.mark.integration


@function_tool
def get_order(order_id: str) -> str:
    """注文を取得する。"""
    return order_id


def _stop_at(*names: object) -> dict[str, object]:
    """extra に積む dict 形の tool_use_behavior を返す。"""
    return {"tool_use_behavior": {"stop_at_tool_names": list(names)}}


def _get_without_warning(registry: AgentRegistry, name: str) -> Agent:
    """`get()` が警告を 1 件も出さないことを記録件数 0 で確認して Agent を返す。"""
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        agent = registry.get(name)
    assert [str(r.message) for r in records] == []
    return agent


def _registry_with_researcher(**router_kwargs: object) -> AgentRegistry:
    """sub_agent `researcher` を持つ router を登録した registry を返す。"""
    registry = AgentRegistry()
    registry.register(AgentSpec(name="researcher", instructions="r", model=FakeModel()))
    registry.register(
        AgentSpec(
            name="router",
            instructions="i",
            model=FakeModel(),
            sub_agents=["researcher"],
            **router_kwargs,
        )
    )
    return registry


# ----------------------------------------------------------------------
# 形状検査（build_agent 経由）
# ----------------------------------------------------------------------


def test_get_rejects_string_value_and_leaves_no_built_residue() -> None:
    """文字列値は get() で ValueError になり、構築キャッシュに残骸（sub_agent 含む）を残さない。"""
    registry = _registry_with_researcher(
        extra={"tool_use_behavior": {"stop_at_tool_names": "researcher"}}
    )
    with pytest.raises(ValueError) as excinfo:
        registry.get("router")
    assert str(excinfo.value) == (
        "agent 'router': tool_use_behavior の stop_at_tool_names は str の list / tuple "
        "である必要がありますが 'str' が渡されました: 'researcher'"
    )
    assert registry._built == {}


def test_get_rejects_non_str_element_with_type_and_name() -> None:
    """要素に FunctionTool を渡すと、型名と name を含む ValueError になる。"""
    registry = AgentRegistry()
    registry.register(
        AgentSpec(
            name="router",
            instructions="i",
            model=FakeModel(),
            tools=[get_order],
            extra=_stop_at(get_order),
        )
    )
    with pytest.raises(ValueError) as excinfo:
        registry.get("router")
    message = str(excinfo.value)
    assert message == (
        "agent 'router': tool_use_behavior の stop_at_tool_names[0] は str である必要が"
        "ありますが 'FunctionTool' が渡されました（name='get_order'）"
    )


# ----------------------------------------------------------------------
# 名前突合（_wire 末尾）
# ----------------------------------------------------------------------


def test_get_accepts_sub_agent_as_tool_default_name() -> None:
    """sub_agent の as_tool 既定名（エージェント名由来）で停止指定すると警告なしで構築される。"""
    registry = _registry_with_researcher(extra=_stop_at("researcher"))
    agent = _get_without_warning(registry, "router")
    assert [t.name for t in agent.tools] == ["researcher"]


def test_get_accepts_sub_agent_tools_override_name() -> None:
    """sub_agent_tools で上書きした as_tool 名で停止指定すると警告なしで構築される。"""
    registry = _registry_with_researcher(
        sub_agent_tools={"researcher": ("ask_researcher", "調査")},
        extra=_stop_at("ask_researcher"),
    )
    agent = _get_without_warning(registry, "router")
    assert [t.name for t in agent.tools] == ["ask_researcher"]


def test_get_warns_when_sub_agent_registered_name_is_used() -> None:
    """as_tool 名を上書きした sub_agent の登録名で停止指定すると RuntimeWarning（構築は成功）。"""
    registry = _registry_with_researcher(
        sub_agent_tools={"researcher": ("ask_researcher", "調査")},
        extra=_stop_at("researcher"),
    )
    with pytest.warns(RuntimeWarning) as records:
        agent = registry.get("router")
    assert isinstance(agent, Agent)
    assert len(records) == 1
    assert str(records[0].message) == (
        "agent 'router': tool_use_behavior の stop_at_tool_names に、このエージェントの "
        "function tool の実行時名（name / qualified_name）と一致しない名前があります: "
        "['researcher']（候補: ['ask_researcher']）。この名前では停止しません"
    )


def test_get_skips_resolution_for_agent_with_mcp_servers() -> None:
    """mcp_servers を持つ spec は不一致名でも警告なしで構築される。"""
    registry = AgentRegistry()
    registry.register(
        AgentSpec(
            name="router",
            instructions="i",
            model=FakeModel(),
            tools=[get_order],
            mcp_servers=[object()],
            extra=_stop_at("mcp_only_tool"),
        )
    )
    agent = _get_without_warning(registry, "router")
    assert agent.mcp_servers


def test_get_skips_resolution_for_sandbox_agent() -> None:
    """SandboxAgentSpec は不一致名（capability 由来想定）でも警告なしで構築される。"""
    registry = AgentRegistry()
    registry.register(
        SandboxAgentSpec(
            name="sbx", instructions="i", model=FakeModel(), extra=_stop_at("sandbox_only_tool")
        )
    )
    _get_without_warning(registry, "sbx")


# ----------------------------------------------------------------------
# DI 差し替え builder（build_agent を通らない経路）
# ----------------------------------------------------------------------


class _PassthroughBuilder:
    """build_agent を呼ばず、extra の tool_use_behavior を素通しで Agent に渡す builder。"""

    def build(self, spec: AgentSpec) -> Agent:
        return Agent(
            name=spec.name,
            instructions=spec.instructions,
            tools=list(spec.tools),
            tool_use_behavior=spec.extra["tool_use_behavior"],
        )


def test_get_with_di_builder_reruns_shape_check_in_wire() -> None:
    """DI builder が形状検査を通さなくても、_wire 側の再検査で ValueError になる。"""
    registry = AgentRegistry(agent_builder=_PassthroughBuilder())
    registry.register(
        AgentSpec(
            name="router",
            instructions="i",
            extra={"tool_use_behavior": {"stop_at_tool_names": "refund"}},
        )
    )
    with pytest.raises(ValueError) as excinfo:
        registry.get("router")
    assert str(excinfo.value) == (
        "agent 'router': tool_use_behavior の stop_at_tool_names は str の list / tuple "
        "である必要がありますが 'str' が渡されました: 'refund'"
    )
    assert registry._built == {}


# ----------------------------------------------------------------------
# workflow facade の非回帰（AC g）
# ----------------------------------------------------------------------


def _facade_workflow() -> WorkflowGraph:
    wf = WorkflowGraph(name="hf")
    wf.add_function_node("a", fn=lambda msg, ctx: msg)
    wf.add_edge(START, "a")
    wf.add_edge("a", END)
    return wf


@pytest.mark.parametrize("mode", [FacadeMode.LLM_INPUT, FacadeMode.DETERMINISTIC])
def test_connect_as_facade_builds_without_warning(mode: FacadeMode) -> None:
    """connect_as_facade のファサードは実 registry の get() で警告なしに構築される。"""
    registry = AgentRegistry()
    registry.register(AgentSpec(name="triage", instructions="t", model=FakeModel()))
    graph = HandoffGraph(entry="triage")
    model = FakeModel() if mode is FacadeMode.LLM_INPUT else None
    _facade_workflow().connect_as_facade(
        registry, graph, "hf_agent", "triage", mode=mode, model=model
    )
    graph.apply(registry)
    _get_without_warning(registry, "triage")
    facade = _get_without_warning(registry, "hf_agent")
    assert facade.tool_use_behavior == "stop_on_first_tool"
