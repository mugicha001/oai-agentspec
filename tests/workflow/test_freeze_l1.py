"""L1（agents 非依存）: ``WorkflowGraph.freeze()`` と ``WorkflowFrozenError`` の検証。

freeze 後の 5 add_* メソッドが ``WorkflowFrozenError`` を raise すること、read-only API
（``validate`` / ``mermaid`` / ``_interpret`` / ``as_agent_spec`` / ``as_facade_spec``）が
影響を受けないこと、freeze の冪等性、``WorkflowFrozenError`` が ``RuntimeError`` を継承して
``IntegrityError`` 系統から分離されていることを検証する。
"""

from __future__ import annotations

import pytest

from oai_agentspec import AgentRegistry, AgentSpec, IntegrityError
from oai_agentspec.workflow import END, START, WorkflowFrozenError, WorkflowGraph

from _helpers.fake_builder import FakeAgentBuilder

pytestmark = pytest.mark.unit


def _make_registry(*names: str) -> AgentRegistry:
    reg = AgentRegistry(agent_builder=FakeAgentBuilder())
    for name in names:
        reg.register(AgentSpec(name=name, instructions=name))
    return reg


def _build_valid_wf() -> WorkflowGraph:
    """validate / mermaid テストで使うミニマルな有効グラフ。"""
    wf = WorkflowGraph(name="wf")
    wf.add_function_node("f", fn=lambda msg, ctx: msg)
    wf.add_edge(START, "f")
    wf.add_edge("f", END)
    return wf


# ----------------------------------------------------------------------
# 例外階層
# ----------------------------------------------------------------------
def test_workflow_frozen_error_is_runtime_error() -> None:
    """``WorkflowFrozenError`` は ``RuntimeError`` を継承し、``IntegrityError`` 系統と分離。"""
    assert issubclass(WorkflowFrozenError, RuntimeError)
    assert not issubclass(WorkflowFrozenError, IntegrityError)


# ----------------------------------------------------------------------
# freeze の冪等性
# ----------------------------------------------------------------------
def test_freeze_is_idempotent() -> None:
    """``freeze()`` を 2 度呼んでも 2 回目は no-op として成功する。"""
    wf = _build_valid_wf()
    wf.freeze()
    wf.freeze()  # 2 回目も例外無く成功。
    # 2 回 freeze 後でも frozen 状態は維持される。
    with pytest.raises(WorkflowFrozenError):
        wf.add_function_node("g", fn=lambda msg, ctx: msg)


# ----------------------------------------------------------------------
# 5 add_* メソッドが freeze 後に raise
# ----------------------------------------------------------------------
def test_add_agent_node_raises_after_freeze() -> None:
    """freeze 後の ``add_agent_node`` は ``WorkflowFrozenError`` を raise する。"""
    wf = _build_valid_wf()
    wf.freeze()
    with pytest.raises(WorkflowFrozenError, match="add_agent_node"):
        wf.add_agent_node("new", agent="a")


def test_add_function_node_raises_after_freeze() -> None:
    """freeze 後の ``add_function_node`` は ``WorkflowFrozenError`` を raise する。"""
    wf = _build_valid_wf()
    wf.freeze()
    with pytest.raises(WorkflowFrozenError, match="add_function_node"):
        wf.add_function_node("new", fn=lambda msg, ctx: msg)


def test_add_edge_raises_after_freeze() -> None:
    """freeze 後の ``add_edge`` は ``WorkflowFrozenError`` を raise する。"""
    wf = _build_valid_wf()
    wf.freeze()
    with pytest.raises(WorkflowFrozenError, match="add_edge"):
        wf.add_edge("f", END)


def test_add_conditional_edges_raises_after_freeze() -> None:
    """freeze 後の ``add_conditional_edges`` は ``WorkflowFrozenError`` を raise する。"""
    wf = _build_valid_wf()
    wf.freeze()
    with pytest.raises(WorkflowFrozenError, match="add_conditional_edges"):
        wf.add_conditional_edges("f", lambda msg, ctx: "x", {"x": END})


def test_add_fan_in_edge_raises_after_freeze() -> None:
    """freeze 後の ``add_fan_in_edge`` は ``WorkflowFrozenError`` を raise する。"""
    wf = _build_valid_wf()
    wf.freeze()
    with pytest.raises(WorkflowFrozenError, match="add_fan_in_edge"):
        wf.add_fan_in_edge(["f"], "merge")


# ----------------------------------------------------------------------
# read-only API は freeze 後も成功
# ----------------------------------------------------------------------
def test_validate_succeeds_after_freeze() -> None:
    """``validate`` は read-only のため freeze 後も成功する。"""
    reg = _make_registry()
    wf = _build_valid_wf()
    wf.freeze()
    wf.validate(reg)  # 例外無し。


def test_mermaid_succeeds_after_freeze() -> None:
    """``mermaid`` は read-only のため freeze 後も成功する。"""
    wf = _build_valid_wf()
    wf.freeze()
    out = wf.mermaid()
    assert "flowchart TD" in out


def test_as_agent_spec_succeeds_after_freeze() -> None:
    """``as_agent_spec`` は read-only のため freeze 後も成功する。"""
    wf = _build_valid_wf()
    wf.freeze()
    spec = wf.as_agent_spec("wf_agent")
    assert spec.name == "wf_agent"


def test_as_facade_spec_succeeds_after_freeze() -> None:
    """``as_facade_spec`` は read-only のため freeze 後も成功する（DETERMINISTIC は LLM 0 回）。"""
    from oai_agentspec.workflow import FacadeMode

    wf = _build_valid_wf()
    wf.freeze()
    spec = wf.as_facade_spec("facade", mode=FacadeMode.DETERMINISTIC)
    assert spec.name == "facade"


# ----------------------------------------------------------------------
# 違反操作名の包含
# ----------------------------------------------------------------------
def test_error_messages_contain_operation_name() -> None:
    """各 ``WorkflowFrozenError`` のメッセージに違反操作名が含まれる。"""
    wf = _build_valid_wf()
    wf.freeze()
    with pytest.raises(WorkflowFrozenError) as exc_info:
        wf.add_function_node("name", fn=lambda msg, ctx: msg)
    assert "add_function_node" in str(exc_info.value)
    assert "name" in str(exc_info.value)


# ----------------------------------------------------------------------
# dict snapshot（外部参照経由の mutation 遮断・MappingProxyType / tuple）
# ----------------------------------------------------------------------
def test_freeze_blocks_nodes_direct_mutation() -> None:
    """``freeze()`` 後、``wf.nodes[key] = ...`` 直接代入は ``TypeError`` で遮断される。"""
    wf = _build_valid_wf()
    wf.freeze()
    with pytest.raises(TypeError):
        wf.nodes["evil"] = None  # type: ignore[index]


def test_freeze_blocks_nodes_clear() -> None:
    """``freeze()`` 後、``wf.nodes.clear()`` も遮断される。"""
    wf = _build_valid_wf()
    wf.freeze()
    with pytest.raises(AttributeError):
        wf.nodes.clear()  # type: ignore[attr-defined]


def test_freeze_blocks_edges_list_mutation() -> None:
    """``freeze()`` 後、``wf.edges[src].append(...)`` は tuple 化により遮断される。"""
    wf = _build_valid_wf()
    wf.freeze()
    # "f" の下流（END）が tuple 化されているので append は AttributeError。
    with pytest.raises(AttributeError):
        wf.edges["f"].append("evil")  # type: ignore[attr-defined]


def test_freeze_blocks_edges_clear() -> None:
    """``freeze()`` 後、``wf.edges.clear()`` は ``MappingProxyType`` で遮断される。"""
    wf = _build_valid_wf()
    wf.freeze()
    with pytest.raises(AttributeError):
        wf.edges.clear()  # type: ignore[attr-defined]


def test_freeze_blocks_conditional_edges_mutation() -> None:
    """``freeze()`` 後、``wf.conditional_edges`` の直接代入も遮断される。"""
    wf = _build_valid_wf()
    wf.freeze()
    with pytest.raises(TypeError):
        wf.conditional_edges["x"] = None  # type: ignore[index]


def test_freeze_blocks_fan_in_edges_mutation() -> None:
    """``freeze()`` 後、``wf.fan_in_edges`` の直接代入も遮断される。"""
    wf = _build_valid_wf()
    wf.freeze()
    with pytest.raises(TypeError):
        wf.fan_in_edges["x"] = None  # type: ignore[index]


def test_freeze_preserves_read_only_api_after_snapshot() -> None:
    """snapshot 後も ``validate`` / ``mermaid`` / ``as_agent_spec`` が動作する。"""
    wf = _build_valid_wf()
    wf.freeze()
    # validate（必要 registry なし: agent ノード未登録なら fake registry のみで通る）。
    wf.validate(_make_registry())
    # mermaid: tuple/MappingProxy 経由でも iterate 可能。
    diagram = wf.mermaid()
    assert "START" in diagram
    assert "END" in diagram
    # as_agent_spec: registry なしでも spec 生成可能。
    spec = wf.as_agent_spec("frozen_agent")
    assert spec.name == "frozen_agent"


def test_freeze_snapshot_is_idempotent() -> None:
    """``freeze()`` 2 回目は no-op で snapshot を再生成しない（identity 維持）。"""
    wf = _build_valid_wf()
    wf.freeze()
    snapshot_nodes = wf.nodes
    snapshot_edges = wf.edges
    wf.freeze()  # 2 回目
    assert wf.nodes is snapshot_nodes
    assert wf.edges is snapshot_edges
