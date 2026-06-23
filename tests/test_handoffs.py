"""L1: HandoffGraph の反映・mermaid・factory 起点エラー検証。"""

from __future__ import annotations

import pytest

from oai_agentspec import AgentRegistry, AgentSpec, HandoffGraph, from_specs

from _helpers.fake_builder import FakeAgentBuilder


def make_registry() -> AgentRegistry:
    return AgentRegistry(agent_builder=FakeAgentBuilder())


def test_apply_via_public_api() -> None:
    reg = make_registry()
    reg.register(AgentSpec(name="triage", instructions="t"))
    reg.register(AgentSpec(name="billing", instructions="b"))
    graph = HandoffGraph(entry="triage")
    graph.edge("triage", "billing")
    graph.apply(reg)
    assert reg._specs["triage"].handoffs == ["billing"]  # noqa: SLF001
    entry = graph.entry_agent(reg)
    assert entry.handoffs[0] is reg.get("billing")


def test_apply_factory_source_errors() -> None:
    reg = make_registry()
    reg.register_factory("custom", lambda r: object())
    reg.register(AgentSpec(name="billing", instructions="b"))
    graph = HandoffGraph()
    graph.edge("custom", "billing")
    with pytest.raises(KeyError, match="custom"):
        graph.apply(reg)


def test_mermaid() -> None:
    graph = HandoffGraph(entry="triage")
    graph.edge("triage", "billing", description="請求")
    out = graph.mermaid()
    assert "flowchart TD" in out
    assert "start([start]) --> triage" in out
    assert "triage -->|請求| billing" in out


def test_extend_and_outgoing() -> None:
    graph = HandoffGraph()
    graph.extend([("a", "b"), ("a", "c")])
    assert graph.outgoing("a") == ["b", "c"]


def test_from_specs() -> None:
    specs = [
        AgentSpec(name="a", instructions="a", handoffs=["b"]),
        AgentSpec(name="b", instructions="b"),
    ]
    graph = from_specs(specs, entry="a")
    assert graph.entry == "a"
    assert graph.outgoing("a") == ["b"]


def test_entry_agent_requires_entry() -> None:
    graph = HandoffGraph()
    with pytest.raises(ValueError, match="no entry"):
        graph.entry_agent(make_registry())
