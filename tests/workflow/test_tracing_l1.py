"""L1: ワークフロー tracing の span 発行プロトコル / 命名規約 / no-op 経路を検証する。

検証対象（agents 非依存・RecordingTracer / 内部 _NULL_TRACER で完結）:

- `_workflow_span_name` / `_node_span_name` / `_condition_span_name` /
  `_fan_out_span_name` / `_fan_in_span_name` の直接 assert（命名規約・anonymous フォールバック）。
- 単純な 2 ノード graph で `workflow_span` -> `node_span(kind)` の順で開閉。
- 条件分岐 graph で `condition_span` が発火する。
- fan-out graph で `fan_out_span` が発火、子の `node_span` が並列に内包される。
- fan-in graph で `fan_in_span` が発火する。
- `tracer=None` 経路（既定）でも既存挙動は変わらない（後方互換）。
"""

from __future__ import annotations

import pytest

from oai_agentspec._adapters.tracing import (
    _condition_span_name,
    _fan_in_span_name,
    _fan_out_span_name,
    _node_span_name,
    _workflow_span_name,
)
from oai_agentspec.workflow import END, START, WorkflowGraph
from oai_agentspec.workflow._interpreter import _NULL_TRACER, _NullWorkflowTracer, interpret

from .conftest import RecordingTracer
from _helpers.fake_runner import FakeRunnerAdapter

pytestmark = pytest.mark.unit


# ----------------------------------------------------------------------
# 命名関数の直接 assert（規約変更を surgical に保つ）
# ----------------------------------------------------------------------
def test_workflow_span_name_uses_graph_name() -> None:
    """workflow span name は `workflow.run.<graph_name>` 形式である。"""
    assert _workflow_span_name("foo") == "workflow.run.foo"


def test_workflow_span_name_anonymous_for_empty() -> None:
    """graph_name が空文字なら `workflow.run.anonymous` にフォールバックする。"""
    assert _workflow_span_name("") == "workflow.run.anonymous"


def test_workflow_span_name_anonymous_for_none() -> None:
    """graph_name が None なら `workflow.run.anonymous` にフォールバックする。"""
    assert _workflow_span_name(None) == "workflow.run.anonymous"


def test_node_span_name_uses_node_name() -> None:
    """node span name は `workflow.node.<node_name>` 形式である（種別は名前に乗せない）。"""
    assert _node_span_name("step1") == "workflow.node.step1"


def test_condition_span_name_uses_src() -> None:
    """condition span name は `workflow.condition.<src>` 形式である。"""
    assert _condition_span_name("router") == "workflow.condition.router"


def test_fan_out_span_name_uses_src() -> None:
    """fan-out span name は `workflow.fan_out.<src>` 形式である。"""
    assert _fan_out_span_name("split") == "workflow.fan_out.split"


def test_fan_in_span_name_uses_dst() -> None:
    """fan-in span name は `workflow.fan_in.<dst>` 形式である。"""
    assert _fan_in_span_name("merge") == "workflow.fan_in.merge"


# ----------------------------------------------------------------------
# span 発行プロトコル（RecordingTracer 注入）
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_sequential_workflow_emits_workflow_and_node_spans_in_order() -> None:
    """2 ノード（AGENT -> FUNCTION）順次実行で workflow_span -> node_span の順序で開閉する。"""
    wf = WorkflowGraph(name="seq_trace")
    wf.add_agent_node("classify", agent="classifier")
    wf.add_function_node("format", fn=lambda msg, ctx: f"<{msg}>")
    wf.add_edge(START, "classify")
    wf.add_edge("classify", "format")
    wf.add_edge("format", END)

    runner = FakeRunnerAdapter({"classifier": "billing"})
    tracer = RecordingTracer()
    await interpret(wf, runner, "q", tracer=tracer)

    # workflow span が最外で、その内側に各 node span（AGENT -> FUNCTION）が順に開閉する。
    assert tracer.events[0] == ("enter", "workflow", "seq_trace", {})
    assert tracer.events[-1] == ("exit", "workflow", "seq_trace")

    # AGENT ノード（kind="agent"）が先に開閉、FUNCTION ノード（kind="function"）が後に開閉。
    inner = tracer.events[1:-1]
    assert inner[0] == ("enter", "node", "classify", {"kind": "agent"})
    assert inner[1] == ("exit", "node", "classify")
    assert inner[2] == ("enter", "node", "format", {"kind": "function"})
    assert inner[3] == ("exit", "node", "format")


@pytest.mark.asyncio
async def test_conditional_workflow_emits_condition_span() -> None:
    """条件分岐 graph で condition_span が発火し、分岐評価のみを span 化する。"""
    wf = WorkflowGraph(name="cond_trace")
    wf.add_function_node("route", fn=lambda msg, ctx: msg)
    wf.add_function_node("to_a", fn=lambda msg, ctx: f"A:{msg}")
    wf.add_edge(START, "route")
    wf.add_conditional_edges("route", lambda msg, ctx: "x", {"x": "to_a"})
    wf.add_edge("to_a", END)

    runner = FakeRunnerAdapter()
    tracer = RecordingTracer()
    await interpret(wf, runner, "q", tracer=tracer)

    condition_events = [e for e in tracer.events if len(e) >= 2 and e[1] == "condition"]
    # condition_span が enter / exit ペアで発火する（分岐評価ぶんのみ）。
    assert condition_events == [
        ("enter", "condition", "route", {}),
        ("exit", "condition", "route"),
    ]
    # condition span は対応する node の node_span 外側で発火する
    # （exec_node 終了後・next_nodes 内）。
    enter_node_route = tracer.events.index(("enter", "node", "route", {"kind": "function"}))
    exit_node_route = tracer.events.index(("exit", "node", "route"))
    enter_condition = tracer.events.index(("enter", "condition", "route", {}))
    assert enter_node_route < exit_node_route < enter_condition


@pytest.mark.asyncio
async def test_fan_out_workflow_emits_fan_out_span_with_parallel_node_spans() -> None:
    """fan-out graph で fan_out_span が発火、その内側に各枝の node_span が並列に内包される。"""
    wf = WorkflowGraph(name="fan_trace")
    wf.add_function_node("src", fn=lambda msg, ctx: msg)
    wf.add_function_node("left", fn=lambda msg, ctx: "L")
    wf.add_function_node("right", fn=lambda msg, ctx: "R")
    wf.add_function_node("merge", fn=lambda inputs, ctx: ",".join(sorted(inputs)))
    wf.add_edge(START, "src")
    wf.add_edge("src", "left")
    wf.add_edge("src", "right")
    wf.add_fan_in_edge(["left", "right"], "merge")
    wf.add_edge("merge", END)

    runner = FakeRunnerAdapter()
    tracer = RecordingTracer()
    await interpret(wf, runner, "in", tracer=tracer)

    # fan_out_span が enter / exit ペアで発火する。
    fan_out_events = [e for e in tracer.events if len(e) >= 2 and e[1] == "fan_out"]
    assert fan_out_events == [
        ("enter", "fan_out", "src", {}),
        ("exit", "fan_out", "src"),
    ]
    # fan_out_span 内側で left / right の node_span が両方発行されている。
    enter_fo = tracer.events.index(("enter", "fan_out", "src", {}))
    exit_fo = tracer.events.index(("exit", "fan_out", "src"))
    inside = tracer.events[enter_fo + 1 : exit_fo]
    enter_kinds = {(kind, name) for tag, kind, name, *_ in inside if tag == "enter"}
    assert ("node", "left") in enter_kinds
    assert ("node", "right") in enter_kinds


@pytest.mark.asyncio
async def test_fan_in_workflow_emits_fan_in_span() -> None:
    """fan-in 合流で fan_in_span が発火する（合流先 FUNCTION の前段で開く）。"""
    wf = WorkflowGraph(name="fanin_trace")
    wf.add_function_node("src", fn=lambda msg, ctx: msg)
    wf.add_function_node("a", fn=lambda msg, ctx: "A")
    wf.add_function_node("b", fn=lambda msg, ctx: "B")
    wf.add_function_node("merge", fn=lambda inputs, ctx: ",".join(sorted(inputs)))
    wf.add_edge(START, "src")
    wf.add_edge("src", "a")
    wf.add_edge("src", "b")
    wf.add_fan_in_edge(["a", "b"], "merge")
    wf.add_edge("merge", END)

    runner = FakeRunnerAdapter()
    tracer = RecordingTracer()
    await interpret(wf, runner, "in", tracer=tracer)

    fan_in_events = [e for e in tracer.events if len(e) >= 2 and e[1] == "fan_in"]
    # fan_in_span は合流が成立した「最後の到達」でのみ発火する（早期離脱では発火しない）。
    assert fan_in_events == [
        ("enter", "fan_in", "merge", {}),
        ("exit", "fan_in", "merge"),
    ]


# ----------------------------------------------------------------------
# no-op 経路（tracer=None・_NULL_TRACER）
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_interpret_without_tracer_does_not_raise() -> None:
    """tracer=None で `interpret` を呼んでも既存挙動は変わらず正常に完走する（後方互換）。"""
    wf = WorkflowGraph(name="no_tracer")
    wf.add_function_node("a", fn=lambda msg, ctx: f"A:{msg}")
    wf.add_edge(START, "a")
    wf.add_edge("a", END)

    runner = FakeRunnerAdapter()
    result = await interpret(wf, runner, "x")  # tracer 引数なし
    assert result.final_output == "A:x"


@pytest.mark.asyncio
async def test_null_tracer_emits_zero_spans_and_runs_through() -> None:
    """`_NULL_TRACER` は全 span 種別で yield のみを返し、span オブジェクトを生成しない。

    span 発行 0 件は構造的に保証される（`_NullWorkflowTracer` の各メソッドは yield のみ）。
    本テストは「`_NULL_TRACER` を陽に渡しても fan-out / fan-in / condition / node の全パスで
    例外なく走り抜けること」を検証する（no-op 経路の整合性確認）。
    """
    wf = WorkflowGraph(name="null_trace")
    wf.add_function_node("src", fn=lambda msg, ctx: msg)
    wf.add_function_node("a", fn=lambda msg, ctx: "A")
    wf.add_function_node("b", fn=lambda msg, ctx: "B")
    wf.add_function_node("merge", fn=lambda inputs, ctx: ",".join(sorted(inputs)))
    wf.add_edge(START, "src")
    wf.add_edge("src", "a")
    wf.add_edge("src", "b")
    wf.add_fan_in_edge(["a", "b"], "merge")
    wf.add_edge("merge", END)

    runner = FakeRunnerAdapter()
    result = await interpret(wf, runner, "in", tracer=_NULL_TRACER)
    # merge は fan-in 合流先で `{source名: 出力}` の dict を受け取り、キー（source 名）を結合する。
    assert result.final_output == "a,b"


def test_null_tracer_and_noop_tracer_behave_equivalently() -> None:
    """`_NullWorkflowTracer`（workflow 層内蔵）と `_NoopWorkflowTracer`（_adapters 側）の
    5 span メソッドが同一引数で context manager として等価に振る舞うことを検証する。

    - enter で yield・exit で何も起きない・例外も発生しない
    - 両クラス間で yield 値の型不一致が無い（共に None を yield）

    両者は同じ no-op 契約を別レイヤーで提供するため、片方が振る舞いドリフトすると tracer
    切り替え時に挙動差が生じる。本テストはそのドリフトを検出する。
    """
    from oai_agentspec._adapters.tracing import _NoopWorkflowTracer

    null_tracer = _NullWorkflowTracer()
    noop_tracer = _NoopWorkflowTracer()

    # 各 span メソッドに同じ引数を渡し、両 tracer が context manager として等価に動くかを確認。
    span_calls = [
        ("workflow_span", ("graph_name",)),
        ("node_span", ("node_name", "agent")),
        ("condition_span", ("src",)),
        ("fan_out_span", ("src",)),
        ("fan_in_span", ("dst",)),
    ]
    for method_name, args in span_calls:
        null_cm = getattr(null_tracer, method_name)(*args)
        noop_cm = getattr(noop_tracer, method_name)(*args)
        with null_cm as null_value, noop_cm as noop_value:
            # 両者とも yield 値は None で一致する（span オブジェクトを生成しない）。
            assert null_value is None
            assert noop_value is None
            assert type(null_value) is type(noop_value)


@pytest.mark.asyncio
async def test_make_workflow_tracer_returns_noop_when_no_trace_active() -> None:
    """SDK tracing 無効時（親 trace 不在）は `make_workflow_tracer` が no-op tracer を返す。

    `_no_external_calls` autouse が `set_tracing_disabled(True)` を立てているため、
    `get_current_trace()` は None となり no-op 経路が選ばれる（オーバーヘッド 0 を担保）。
    """
    from oai_agentspec._adapters import make_workflow_tracer
    from oai_agentspec._adapters.tracing import _NoopWorkflowTracer

    tracer = make_workflow_tracer("any")
    assert isinstance(tracer, _NoopWorkflowTracer)

    # no-op tracer の全 span メソッドが yield のみで動作することを担保（span 生成なし・例外なし）。
    with tracer.workflow_span("any"):
        with tracer.node_span("n", "agent"):
            pass
    with tracer.condition_span("c"):
        pass
    with tracer.fan_out_span("o"):
        pass
    with tracer.fan_in_span("i"):
        pass


def test_make_workflow_tracer_returns_noop_when_tracing_disabled() -> None:
    """`set_tracing_disabled(True)` 配下で `NoOpTrace` が current にセットされた状態でも
    `make_workflow_tracer` は `_NoopWorkflowTracer` を返す（オーバーヘッド 0 の担保）。

    SDK は `set_tracing_disabled(True)` 配下で `agents.trace(...)` が呼ばれると、`NoOpTrace`
    インスタンスを current trace としてセットする。このとき `get_current_trace() is None` は
    False になるため、`is None` のみの判定では `_SdkWorkflowTracer` が返ってしまい、各 span
    で `custom_span()` が呼ばれて SDK 側で `NoOpSpan` を生成するコストが発生する。

    本テストは `_no_external_calls` autouse（`set_tracing_disabled(True)`）配下で `NoOpTrace`
    を current に立てた状態で `make_workflow_tracer` が確実に no-op 経路を選ぶことを検証する。
    """
    from agents import trace as agents_trace

    from oai_agentspec._adapters import make_workflow_tracer
    from oai_agentspec._adapters.tracing import _NoopWorkflowTracer, get_current_trace

    # set_tracing_disabled(True) 配下で trace() を立てると NoOpTrace が current にセットされる。
    with agents_trace("dummy"):
        current = get_current_trace()
        # 前提条件: current は None ではないが、型名は NoOpTrace である。
        assert current is not None
        assert type(current).__name__ == "NoOpTrace"

        tracer = make_workflow_tracer("g")
        assert isinstance(tracer, _NoopWorkflowTracer)
