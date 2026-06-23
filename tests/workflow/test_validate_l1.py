"""L1（agents 非依存）: WorkflowGraph.validate / mermaid を node/edge 方式で検証する。

validate は参照ミス（registry 未登録 AGENT・未登録ノード名）・単一エントリ・到達性
（START から各ノード / END）・conditional mapping 欠落・fan-in 全ソース解決および合流先が
FUNCTION であること・recursion_limit の存在を集約報告し ValueError を投げる（FR-6/NFR-5）。
registry は FakeAgentBuilder ベースで構築する（SDK 非依存）。
"""

from __future__ import annotations

import pytest

from oai_agentspec import AgentRegistry, AgentSpec
from oai_agentspec.workflow import END, START, WorkflowGraph

from _helpers.fake_builder import FakeAgentBuilder

pytestmark = pytest.mark.unit


def make_registry(*names: str) -> AgentRegistry:
    reg = AgentRegistry(agent_builder=FakeAgentBuilder())
    for name in names:
        reg.register(AgentSpec(name=name, instructions=name))
    return reg


def test_validate_passes_for_well_formed_graph() -> None:
    """全参照解決・到達可能なグラフは例外を出さない。"""
    reg = make_registry("classifier")
    wf = WorkflowGraph(name="ok")
    wf.add_agent_node("classify", agent="classifier")
    wf.add_function_node("format", fn=lambda msg, ctx: msg)
    wf.add_edge(START, "classify")
    wf.add_edge("classify", "format")
    wf.add_edge("format", END)
    wf.validate(reg)  # 例外が出なければ成功。


def test_validate_passes_for_fan_in_graph() -> None:
    """fan-out / fan-in（合流先 FUNCTION）の整ったグラフは例外を出さない。"""
    reg = make_registry("l", "r")
    wf = WorkflowGraph(name="ok_fan")
    wf.add_agent_node("left", agent="l")
    wf.add_agent_node("right", agent="r")
    wf.add_function_node("merge", fn=lambda msg, ctx: msg)
    wf.add_edge(START, "left")
    wf.add_edge("left", "right")
    wf.add_fan_in_edge(["left", "right"], "merge")
    wf.add_edge("merge", END)
    wf.validate(reg)


def test_validate_detects_missing_agent_reference() -> None:
    """AGENT ノードの registry 未登録参照を検出する。"""
    reg = make_registry()  # 空 registry。
    wf = WorkflowGraph(name="bad_agent")
    wf.add_agent_node("classify", agent="ghost")
    wf.add_edge(START, "classify")
    wf.add_edge("classify", END)
    with pytest.raises(ValueError, match="agent 参照 'ghost' が registry 未登録"):
        wf.validate(reg)


def test_validate_detects_unreachable_node() -> None:
    """START から到達不能なノードを検出する。"""
    reg = make_registry("a")
    wf = WorkflowGraph(name="unreach")
    wf.add_agent_node("a", agent="a")
    wf.add_function_node("orphan", fn=lambda msg, ctx: msg)  # 到達不能。
    wf.add_edge(START, "a")
    wf.add_edge("a", END)
    with pytest.raises(ValueError, match="'orphan' が START から到達不能"):
        wf.validate(reg)


def test_validate_detects_no_path_to_end() -> None:
    """START から END へ到達できないグラフを検出する。"""
    reg = make_registry()
    wf = WorkflowGraph(name="no_end")
    wf.add_function_node("a", fn=lambda msg, ctx: msg)
    wf.add_edge(START, "a")  # END へ繋がるエッジが無い。
    with pytest.raises(ValueError, match="START から END へ到達できません"):
        wf.validate(reg)


def test_validate_detects_missing_conditional_target() -> None:
    """conditional mapping の全分岐先の未登録ノードを検出する。"""
    reg = make_registry("router")
    wf = WorkflowGraph(name="bad_cond")
    wf.add_agent_node("route", agent="router")
    wf.add_function_node("known", fn=lambda msg, ctx: msg)
    wf.add_edge(START, "route")
    wf.add_conditional_edges(
        "route",
        lambda msg, ctx: msg,
        {"a": "known", "b": "missing"},
    )
    wf.add_edge("known", END)
    with pytest.raises(ValueError) as exc:
        wf.validate(reg)
    assert "'missing'" in str(exc.value)


def test_validate_detects_missing_edge_target() -> None:
    """通常エッジが参照する未登録ノードを検出する。"""
    reg = make_registry()
    wf = WorkflowGraph(name="bad_edge")
    wf.add_function_node("a", fn=lambda msg, ctx: msg)
    wf.add_edge(START, "a")
    wf.add_edge("a", "nowhere")
    with pytest.raises(ValueError, match="エッジの終点 node 'nowhere'"):
        wf.validate(reg)


def test_validate_detects_missing_edge_source() -> None:
    """通常エッジの始点が未登録ノードのとき検出する。"""
    reg = make_registry()
    wf = WorkflowGraph(name="bad_edge_src")
    wf.add_function_node("a", fn=lambda msg, ctx: msg)
    wf.add_edge(START, "a")
    wf.add_edge("a", END)
    wf.edges["ghost_src"] = ["a"]  # 始点が未登録のエッジを直接注入。
    with pytest.raises(ValueError, match="エッジの始点 node 'ghost_src'"):
        wf.validate(reg)


def test_validate_detects_missing_conditional_source() -> None:
    """conditional の始点が未登録ノードのとき検出する。"""
    reg = make_registry()
    wf = WorkflowGraph(name="bad_cond_src")
    wf.add_function_node("a", fn=lambda msg, ctx: msg)
    wf.add_edge(START, "a")
    wf.add_edge("a", END)
    wf.add_conditional_edges("ghost_cond", lambda msg, ctx: "x", {"x": END})
    with pytest.raises(ValueError, match="条件エッジの始点 node 'ghost_cond'"):
        wf.validate(reg)


def test_validate_detects_missing_fan_in_dst() -> None:
    """fan-in の合流先ノードが未登録のとき検出する。"""
    reg = make_registry()
    wf = WorkflowGraph(name="bad_fan_dst")
    wf.add_function_node("a", fn=lambda msg, ctx: msg)
    wf.add_edge(START, "a")
    wf.add_edge("a", END)
    wf.add_fan_in_edge(["a"], "ghost_merge")  # 合流先が未登録。
    with pytest.raises(ValueError, match="fan-in の合流先 node 'ghost_merge' が未登録"):
        wf.validate(reg)


def test_validate_detects_missing_fan_in_source() -> None:
    """fan-in が参照する未登録ソースノードを検出する。"""
    reg = make_registry()
    wf = WorkflowGraph(name="bad_fan_src")
    wf.add_function_node("a", fn=lambda msg, ctx: msg)
    wf.add_function_node("merge", fn=lambda msg, ctx: msg)
    wf.add_edge(START, "a")
    wf.add_fan_in_edge(["a", "ghost_src"], "merge")
    wf.add_edge("merge", END)
    with pytest.raises(ValueError, match="fan-in のソース node 'ghost_src'"):
        wf.validate(reg)


def test_validate_detects_fan_in_dst_is_agent() -> None:
    """fan-in の合流先が AGENT ノードのとき検出する（合流先は FUNCTION 必須・C-4）。"""
    reg = make_registry("l", "r", "bad")
    wf = WorkflowGraph(name="fan_agent_dst")
    wf.add_agent_node("left", agent="l")
    wf.add_agent_node("right", agent="r")
    wf.add_agent_node("merge", agent="bad")  # 合流先が AGENT（不正）。
    wf.add_edge(START, "left")
    wf.add_edge("left", "right")
    wf.add_fan_in_edge(["left", "right"], "merge")
    wf.add_edge("merge", END)
    with pytest.raises(ValueError, match="fan-in の合流先 node 'merge' は FUNCTION ノードが必須"):
        wf.validate(reg)


def test_validate_detects_missing_recursion_limit() -> None:
    """recursion_limit が 1 未満のとき検出する（FR-6 (e)）。"""
    reg = make_registry()
    wf = WorkflowGraph(name="bad_limit", recursion_limit=0)
    wf.add_function_node("a", fn=lambda msg, ctx: msg)
    wf.add_edge(START, "a")
    wf.add_edge("a", END)
    with pytest.raises(ValueError, match="recursion_limit は 1 以上が必須"):
        wf.validate(reg)


def test_validate_reports_unset_entry() -> None:
    """START からのエントリ未設定を検出する。"""
    reg = make_registry()
    wf = WorkflowGraph(name="no_entry")
    with pytest.raises(ValueError, match="エントリ.*が未設定"):
        wf.validate(reg)


def test_validate_reports_entry_not_registered() -> None:
    """entry が未登録ノード名のとき検出する。"""
    reg = make_registry()
    wf = WorkflowGraph(name="ghost_entry")
    wf.add_edge(START, "ghost")
    with pytest.raises(ValueError, match="エントリノード 'ghost' が未登録"):
        wf.validate(reg)


def test_validate_detects_fan_in_dst_also_normal_target() -> None:
    """fan-in 合流先が通常エッジの終点も兼ねる場合を検出する（dict/単一出力の曖昧化）。"""
    reg = make_registry()
    wf = WorkflowGraph(name="ambiguous_dst")
    wf.add_function_node("a", fn=lambda msg, ctx: msg)
    wf.add_function_node("b", fn=lambda msg, ctx: msg)
    wf.add_function_node("merge", fn=lambda msg, ctx: msg)
    wf.add_edge(START, "a")
    wf.add_edge("a", "b")
    wf.add_edge("a", "merge")  # merge は通常エッジの終点でもある（曖昧）。
    wf.add_fan_in_edge(["b", "a"], "merge")
    wf.add_edge("merge", END)
    with pytest.raises(ValueError, match="通常エッジの終点も兼ねています"):
        wf.validate(reg)


def test_add_conditional_edges_rejects_duplicate_src() -> None:
    """同一 src への条件エッジ 2 度宣言は fail-fast（黙示上書きしない）。"""
    wf = WorkflowGraph(name="dup_cond")
    wf.add_function_node("route", fn=lambda msg, ctx: msg)
    wf.add_function_node("a", fn=lambda msg, ctx: msg)
    wf.add_conditional_edges("route", lambda msg, ctx: "a", {"a": "a"})
    with pytest.raises(ValueError, match="条件エッジが重複"):
        wf.add_conditional_edges("route", lambda msg, ctx: "a", {"a": "a"})


def test_add_conditional_edges_rejects_default_with_mapping_none() -> None:
    """mapping=None で default 指定は fail-fast（mapping=None では default が黙殺されるため）。"""
    wf = WorkflowGraph("cond_default")
    wf.add_function_node("route", fn=lambda msg, ctx: msg)
    wf.add_function_node("a", fn=lambda msg, ctx: msg)
    with pytest.raises(ValueError, match="mapping=None では default"):
        wf.add_conditional_edges("route", lambda msg, ctx: "a", default="a")


def test_add_fan_in_edge_rejects_duplicate_dst() -> None:
    """同一 dst への fan-in 2 度宣言は fail-fast（黙示上書きしない）。"""
    wf = WorkflowGraph(name="dup_fan")
    wf.add_function_node("x", fn=lambda msg, ctx: msg)
    wf.add_function_node("y", fn=lambda msg, ctx: msg)
    wf.add_function_node("merge", fn=lambda msg, ctx: msg)
    wf.add_fan_in_edge(["x", "y"], "merge")
    with pytest.raises(ValueError, match="fan-in が重複"):
        wf.add_fan_in_edge(["x", "y"], "merge")


def test_validate_detects_diamond_double_execution() -> None:
    """fan-out の複数枝が通常エッジで同一ノードへ合流し fan-in 未宣言なら検出する（二重実行）。"""
    reg = make_registry()
    wf = WorkflowGraph("diamond")
    wf.add_function_node("a", fn=lambda msg, ctx: msg)
    wf.add_function_node("b", fn=lambda msg, ctx: msg)
    wf.add_function_node("c", fn=lambda msg, ctx: msg)
    wf.add_function_node("d", fn=lambda msg, ctx: msg)
    wf.add_edge(START, "a")
    wf.add_edge("a", "b")
    wf.add_edge("a", "c")  # a が fan-out
    wf.add_edge("b", "d")
    wf.add_edge("c", "d")  # b/c が通常エッジで d へ合流（fan-in 未宣言）
    wf.add_edge("d", END)
    with pytest.raises(ValueError, match="複数枝から通常エッジで合流"):
        wf.validate(reg)


def test_validate_detects_multiple_independent_end() -> None:
    """fan-out の複数枝が fan-in を介さず独立に END へ到達すると検出する（出力破棄）。"""
    reg = make_registry()
    wf = WorkflowGraph("multi_end")
    wf.add_function_node("a", fn=lambda msg, ctx: msg)
    wf.add_function_node("b", fn=lambda msg, ctx: msg)
    wf.add_function_node("c", fn=lambda msg, ctx: msg)
    wf.add_edge(START, "a")
    wf.add_edge("a", "b")
    wf.add_edge("a", "c")  # a が fan-out
    wf.add_edge("b", END)
    wf.add_edge("c", END)  # 複数枝が独立に END へ
    with pytest.raises(ValueError, match="複数枝が fan-in を介さず END"):
        wf.validate(reg)


def test_validate_passes_for_fan_out_joined_by_fan_in() -> None:
    """fan-out を fan-in で正しく合流するグラフは収束検査でも通る（誤検出しない）。"""
    reg = make_registry()
    wf = WorkflowGraph("ok_diamond")
    wf.add_function_node("a", fn=lambda msg, ctx: msg)
    wf.add_function_node("b", fn=lambda msg, ctx: msg)
    wf.add_function_node("c", fn=lambda msg, ctx: msg)
    wf.add_function_node("merge", fn=lambda msg, ctx: msg)
    wf.add_edge(START, "a")
    wf.add_edge("a", "b")
    wf.add_edge("a", "c")
    wf.add_fan_in_edge(["b", "c"], "merge")  # fan-in で合流（正しい）
    wf.add_edge("merge", END)
    wf.validate(reg)  # 例外が出なければ成功


def test_validate_passes_for_conditional_merge_point() -> None:
    """条件エッジ（排他分岐）が同一ノードへ合流するのは二重実行でないため検出しない。"""
    reg = make_registry()
    wf = WorkflowGraph("cond_merge")
    wf.add_function_node("route", fn=lambda msg, ctx: msg)
    wf.add_function_node("x", fn=lambda msg, ctx: msg)
    wf.add_function_node("y", fn=lambda msg, ctx: msg)
    wf.add_function_node("after", fn=lambda msg, ctx: msg)
    wf.add_edge(START, "route")
    wf.add_conditional_edges("route", lambda msg, ctx: "x", {"x": "x", "y": "y"})
    wf.add_edge("x", "after")
    wf.add_edge("y", "after")  # 排他分岐の合流（並行ではない）
    wf.add_edge("after", END)
    wf.validate(reg)  # 例外が出なければ成功


def test_validate_aggregates_multiple_problems() -> None:
    """複数の検証エラーを 1 つの例外メッセージへ集約する（; 区切り）。"""
    reg = make_registry()
    wf = WorkflowGraph(name="multi")
    wf.add_agent_node("a", agent="ghost1")  # registry 未登録。
    wf.add_function_node("b", fn=lambda msg, ctx: msg)  # 到達不能。
    wf.add_edge(START, "a")
    wf.add_edge("a", END)
    with pytest.raises(ValueError) as exc:
        wf.validate(reg)
    msg = str(exc.value)
    assert "ghost1" in msg
    assert "'b' が START から到達不能" in msg
    assert ";" in msg


# ----------------------------------------------------------------------
# mermaid
# ----------------------------------------------------------------------
def test_mermaid_includes_all_edge_kinds() -> None:
    """mermaid に START/END / 通常エッジ / conditional / fan-in が反映される。"""
    wf = WorkflowGraph(name="viz")
    wf.add_agent_node("src", agent="s")
    wf.add_agent_node("left", agent="l")
    wf.add_agent_node("right", agent="r")
    wf.add_function_node("merge", fn=lambda msg, ctx: msg)
    wf.add_function_node("to_a", fn=lambda msg, ctx: msg)
    wf.add_edge(START, "src")
    wf.add_edge("src", "left")
    wf.add_edge("src", "right")
    wf.add_fan_in_edge(["left", "right"], "merge")
    wf.add_conditional_edges("merge", lambda msg, ctx: "a", {"a": "to_a", "stop": END})
    wf.add_edge("to_a", END)

    out = wf.mermaid()
    assert "flowchart TD" in out
    assert "START([START])" in out
    assert "END([END])" in out
    assert "START --> src" in out
    assert "src --> left" in out
    assert "src --> right" in out
    # conditional は判定キーをラベルに。
    assert "merge -->|a| to_a" in out
    # conditional の END 分岐は END ラベルへ。
    assert "merge -->|stop| END" in out
    # fan-in は破線。
    assert "left -.-> merge" in out
    assert "right -.-> merge" in out
    # 通常エッジの END。
    assert "to_a --> END" in out
    # FUNCTION ノードは丸括弧形状。
    assert "merge(merge)" in out
    # AGENT ノードは角括弧形状。
    assert "src[src]" in out
