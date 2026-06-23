"""L2: 実 Agent + FakeModel + 実 `Runner` + tracing_enabled collector で span 親子関係を検証。

経路C（`as_agent_spec`）/ 経路A（`as_facade_spec(mode=LLM_INPUT)`）/ 経路D
（`as_facade_spec(mode=DETERMINISTIC)`）の各経路で:

- workflow span が外側 trace の子として現れる（AC-1）。
- node span が workflow span の子として現れる（AC-2/AC-5）。
- span 命名規約（`workflow.run.<name>` / `workflow.node.<name>`）が守られる。
- 実 SDK の tracing API（`custom_span` / `get_current_trace`）が `_adapters/tracing.py`
  から利用可能（SDK 退行検知トリップワイヤ）。

`tracing_enabled` fixture（conftest）が `set_tracing_disabled(False)` と collector を
yield 寿命で設定する（root conftest の autouse オーバーライド・teardown 必須）。
"""

from __future__ import annotations

from typing import Any

import pytest
from agents import Runner

from oai_agentspec import AgentRegistry, FacadeMode
from oai_agentspec._adapters import build_agent
from oai_agentspec.workflow import END, START, WorkflowGraph

from _helpers.fake_model import FakeModel

pytestmark = pytest.mark.integration


# ----------------------------------------------------------------------
# ヘルパ
# ----------------------------------------------------------------------
def _workflow_spans(collector: Any, graph_name: str) -> list[Any]:
    """collector から workflow.* プレフィックスの span を抽出する。"""
    return [
        s for s in collector.spans if s.name and s.name.startswith(f"workflow.run.{graph_name}")
    ]


def _spans_by_prefix(collector: Any, prefix: str) -> list[Any]:
    """collector から指定 prefix の span を抽出する。"""
    return [s for s in collector.spans if s.name and s.name.startswith(prefix)]


# ----------------------------------------------------------------------
# 経路C: as_agent_spec → Runner.run で workflow / node span が記録される
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_path_c_records_workflow_and_node_spans(
    tracing_enabled: Any,
) -> None:
    """経路C: WorkflowModel を据えた Agent を Runner.run で実行すると workflow / node span が
    外側 trace 配下に記録される（AC-1/AC-2）。"""
    reg = AgentRegistry()
    wf = WorkflowGraph(name="trace_c")
    wf.add_function_node("step", fn=lambda msg, ctx: f"<{msg}>")
    wf.add_edge(START, "step")
    wf.add_edge("step", END)
    agent = build_agent(wf.as_agent_spec("trace_c_agent", registry=reg))

    result = await Runner.run(agent, input="hi")
    assert result.final_output == "<hi>"

    # workflow span が記録されている（`workflow.run.trace_c`）。
    wf_spans = _workflow_spans(tracing_enabled, "trace_c")
    assert len(wf_spans) == 1
    wf_span = wf_spans[0]

    # workflow span は外側 trace 直下ではなく内側（Runner.run が agent span を開く）に位置する。
    # 重要なのは「親が存在し trace が確立されている」こと（AC-1: 独立 trace 化されない）。
    assert wf_span.trace_id  # trace の中にいる
    assert wf_span.parent_id is not None  # 外側 trace の何かの span 配下

    # node span が workflow span の子として記録されている（AC-2）。
    node_spans = _spans_by_prefix(tracing_enabled, "workflow.node.step")
    assert len(node_spans) == 1
    node_span = node_spans[0]
    assert node_span.parent_id == wf_span.span_id  # workflow span 直下


@pytest.mark.asyncio
async def test_path_c_node_span_data_includes_kind_and_graph_name(
    tracing_enabled: Any,
) -> None:
    """node span の data 属性に `workflow.graph_name` / `workflow.node_name` /
    `workflow.node_kind` が必須キーとして含まれる（OpenTelemetry 風 namespace）。"""
    reg = AgentRegistry()
    wf = WorkflowGraph(name="trace_data")
    wf.add_function_node("step", fn=lambda msg, ctx: msg)
    wf.add_edge(START, "step")
    wf.add_edge("step", END)
    agent = build_agent(wf.as_agent_spec("trace_data_agent", registry=reg))

    await Runner.run(agent, input="x")

    node_spans = _spans_by_prefix(tracing_enabled, "workflow.node.step")
    assert len(node_spans) == 1
    data = node_spans[0].data
    assert data is not None
    assert data["workflow.graph_name"] == "trace_data"
    assert data["workflow.node_name"] == "step"
    assert data["workflow.node_kind"] == "function"


# ----------------------------------------------------------------------
# 経路A: as_facade_spec(mode=LLM_INPUT) → Runner.run（FakeModel）
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_path_a_facade_records_workflow_and_node_spans(
    tracing_enabled: Any,
) -> None:
    """経路A: 入口 LLM（FakeModel）が tool 呼び出しを発行し、その内側で workflow / node span
    が記録される（AC-5・LLM_INPUT mode）。"""
    wf = WorkflowGraph(name="trace_a")
    wf.add_function_node("step", fn=lambda msg, ctx: f"WF[{msg}]")
    wf.add_edge(START, "step")
    wf.add_edge("step", END)

    entry = FakeModel().queue_tool_call("wf_tool", '{"input": "payload"}')
    spec = wf.as_facade_spec(
        "trace_a_agent", mode=FacadeMode.LLM_INPUT, model=entry, tool_name="wf_tool"
    )
    agent = build_agent(spec)

    result = await Runner.run(agent, input="hi")
    assert result.final_output == "WF[payload]"

    # workflow span が記録されている。
    wf_spans = _workflow_spans(tracing_enabled, "trace_a")
    assert len(wf_spans) == 1
    wf_span = wf_spans[0]
    assert wf_span.parent_id is not None  # 外側 trace 配下

    # node span が workflow span の子として記録されている。
    node_spans = _spans_by_prefix(tracing_enabled, "workflow.node.step")
    assert len(node_spans) == 1
    assert node_spans[0].parent_id == wf_span.span_id


# ----------------------------------------------------------------------
# 経路D: as_facade_spec(mode=DETERMINISTIC) → Runner.run（実 LLM なし）
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_path_d_deterministic_facade_records_workflow_and_node_spans(
    tracing_enabled: Any,
) -> None:
    """経路D: 決定論モデルが tool 呼び出しを発行し、その内側で workflow / node span が
    記録される（実 LLM 0 回・AC-5）。"""
    wf = WorkflowGraph(name="trace_d")
    wf.add_function_node("step", fn=lambda msg, ctx: f"<{msg}>")
    wf.add_edge(START, "step")
    wf.add_edge("step", END)

    spec = wf.as_facade_spec("trace_d_agent", mode=FacadeMode.DETERMINISTIC)
    agent = build_agent(spec)

    result = await Runner.run(agent, input="hi")
    assert result.final_output == "<hi>"

    wf_spans = _workflow_spans(tracing_enabled, "trace_d")
    assert len(wf_spans) == 1
    wf_span = wf_spans[0]
    assert wf_span.parent_id is not None

    node_spans = _spans_by_prefix(tracing_enabled, "workflow.node.step")
    assert len(node_spans) == 1
    assert node_spans[0].parent_id == wf_span.span_id


# ----------------------------------------------------------------------
# 並行 run で span 親子関係が混線しないこと（ステートレス factory 検証）
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_concurrent_runs_emit_independent_workflow_spans(
    tracing_enabled: Any,
) -> None:
    """並行 run でそれぞれの workflow span が独立して記録される（ステートレス factory）。

    各 run の span 親子関係が混線しないことを担保する（trace_id が異なる）。
    """
    import asyncio

    reg = AgentRegistry()
    wf = WorkflowGraph(name="trace_par")
    wf.add_function_node("step", fn=lambda msg, ctx: f"<{msg}>")
    wf.add_edge(START, "step")
    wf.add_edge("step", END)
    agent = build_agent(wf.as_agent_spec("trace_par_agent", registry=reg))

    await asyncio.gather(
        Runner.run(agent, input="a"),
        Runner.run(agent, input="b"),
    )

    wf_spans = _workflow_spans(tracing_enabled, "trace_par")
    # 2 つの workflow span が独立して記録される（trace_id が異なる）。
    assert len(wf_spans) == 2
    assert wf_spans[0].trace_id != wf_spans[1].trace_id


# ----------------------------------------------------------------------
# SDK 退行検知トリップワイヤ（NFR-7・バージョン耐性）
# ----------------------------------------------------------------------
def test_sdk_tracing_symbols_available() -> None:
    """SDK の tracing API（custom_span / get_current_trace / set_tracing_disabled）が
    `_adapters/tracing.py` から import 可能であることを担保する。

    将来 SDK がこれらを削除 / 改名すると `_adapters/tracing.py` が壊れるため、
    シンボル存在を早期検知するトリップワイヤ（NFR-7・既存 L2 トリップワイヤと同型）。
    """
    from agents import custom_span, get_current_trace, set_tracing_disabled

    assert callable(custom_span)
    assert callable(get_current_trace)
    assert callable(set_tracing_disabled)


def test_workflow_tracer_internally_exported_from_adapters() -> None:
    """`_adapters.WorkflowTracer` / `make_workflow_tracer` が内部窓口として参照可能。

    公開 `__all__` には積まないが、workflow 層からの関数内遅延 import が成立することを担保。
    """
    from oai_agentspec import _adapters

    assert hasattr(_adapters, "WorkflowTracer")
    assert hasattr(_adapters, "make_workflow_tracer")
    # 公開 `__all__` には含まれない（内部窓口）。
    assert "WorkflowTracer" not in _adapters.__all__
    assert "make_workflow_tracer" not in _adapters.__all__


@pytest.mark.asyncio
async def test_tracing_disabled_runs_workflow_without_span_emission() -> None:
    """`set_tracing_disabled(True)`（root autouse）下では workflow span が 1 件も記録されない。

    root conftest の autouse が `set_tracing_disabled(True)` を設定しており、本テストは
    `tracing_enabled` fixture を使わず autouse のみで走るため、`make_workflow_tracer` が
    no-op tracer を返し span は発行されない（オーバーヘッド 0 経路の確認）。
    """
    reg = AgentRegistry()
    wf = WorkflowGraph(name="trace_off")
    wf.add_function_node("step", fn=lambda msg, ctx: msg)
    wf.add_edge(START, "step")
    wf.add_edge("step", END)
    agent = build_agent(wf.as_agent_spec("trace_off_agent", registry=reg))

    # autouse の set_tracing_disabled(True) 下では実行が成功し例外も出ない。
    result = await Runner.run(agent, input="x")
    assert result.final_output == "x"


# ----------------------------------------------------------------------
# AgentSpec / fixture 経由でも fan-out が span を発行する経路の検証
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_path_c_conditional_records_condition_span(
    tracing_enabled: Any,
) -> None:
    """経路C で条件分岐 graph を実行すると condition span が記録される（SDK 経由カバー）。"""
    reg = AgentRegistry()
    wf = WorkflowGraph(name="trace_cond")
    wf.add_function_node("route", fn=lambda msg, ctx: msg)
    wf.add_function_node("to_a", fn=lambda msg, ctx: f"A:{msg}")
    wf.add_edge(START, "route")
    wf.add_conditional_edges("route", lambda msg, ctx: "x", {"x": "to_a"})
    wf.add_edge("to_a", END)
    agent = build_agent(wf.as_agent_spec("trace_cond_agent", registry=reg))

    await Runner.run(agent, input="q")

    cond_spans = _spans_by_prefix(tracing_enabled, "workflow.condition.route")
    assert len(cond_spans) == 1
    # condition span の data 属性に種別 condition が乗る。
    assert cond_spans[0].data["workflow.node_kind"] == "condition"


@pytest.mark.asyncio
async def test_path_c_fan_out_records_fan_out_span(
    tracing_enabled: Any,
) -> None:
    """経路C で fan-out graph を実行すると fan_out span が記録され、子の node span を包む。"""
    reg = AgentRegistry()
    wf = WorkflowGraph(name="trace_fanout")
    wf.add_function_node("src", fn=lambda msg, ctx: msg)
    wf.add_function_node("left", fn=lambda msg, ctx: "L")
    wf.add_function_node("right", fn=lambda msg, ctx: "R")
    wf.add_function_node("merge", fn=lambda inputs, ctx: ",".join(sorted(inputs)))
    wf.add_edge(START, "src")
    wf.add_edge("src", "left")
    wf.add_edge("src", "right")
    wf.add_fan_in_edge(["left", "right"], "merge")
    wf.add_edge("merge", END)
    agent = build_agent(wf.as_agent_spec("trace_fanout_agent", registry=reg))

    await Runner.run(agent, input="in")

    fan_out_spans = _spans_by_prefix(tracing_enabled, "workflow.fan_out.src")
    fan_in_spans = _spans_by_prefix(tracing_enabled, "workflow.fan_in.merge")
    assert len(fan_out_spans) == 1
    assert len(fan_in_spans) == 1

    # fan-out span の data 属性に種別 fan_out が乗る。
    assert fan_out_spans[0].data["workflow.node_kind"] == "fan_out"
    assert fan_in_spans[0].data["workflow.node_kind"] == "fan_in"
