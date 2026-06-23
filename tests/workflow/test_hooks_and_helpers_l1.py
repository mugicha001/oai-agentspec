"""L1（agents 非依存）: ノードフック・default_input_filter・周辺分岐の補完テスト。

on_node_start / on_node_end の発火（sync / async）、default_input_filter の有界化、
_exec_node の未登録ノードガードを node/edge 方式で検証する。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from oai_agentspec import default_input_filter
from oai_agentspec.workflow import END, START, NodeResults, WorkflowGraph

from _helpers.fake_runner import FakeRunnerAdapter

pytestmark = pytest.mark.unit


# ----------------------------------------------------------------------
# ノードフック
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_node_hooks_fire_around_each_node() -> None:
    """on_node_start / on_node_end が各ノードの前後で発火する（sync フック）。"""
    events: list[str] = []

    def on_start(name: str, results: NodeResults, ctx: object) -> None:
        events.append(f"start:{name}")

    def on_end(name: str, results: NodeResults, ctx: object) -> None:
        events.append(f"end:{name}={results.get(name)}")

    wf = WorkflowGraph(name="hooks")
    wf.add_function_node("a", fn=lambda msg, ctx: "A")
    wf.add_function_node("b", fn=lambda msg, ctx: "B")
    wf.add_edge(START, "a")
    wf.add_edge("a", "b")
    wf.add_edge("b", END)

    runner = FakeRunnerAdapter()
    await wf._interpret(runner, "in", on_node_start=on_start, on_node_end=on_end)

    assert events == ["start:a", "end:a=A", "start:b", "end:b=B"]


@pytest.mark.asyncio
async def test_async_node_hooks_awaited() -> None:
    """async なノードフックは await される。"""
    events: list[str] = []

    async def on_start(name: str, results: NodeResults, ctx: object) -> None:
        events.append(f"start:{name}")

    wf = WorkflowGraph(name="async_hooks")
    wf.add_function_node("a", fn=lambda msg, ctx: msg)
    wf.add_edge(START, "a")
    wf.add_edge("a", END)

    runner = FakeRunnerAdapter()
    await wf._interpret(runner, "x", on_node_start=on_start)
    assert events == ["start:a"]


@pytest.mark.asyncio
async def test_hooks_receive_shared_context() -> None:
    """ノードフックは共有 context を第 3 引数で受け取る。"""
    seen: list[object] = []

    def on_end(name: str, results: NodeResults, ctx: object) -> None:
        seen.append(ctx)

    ctx_obj = {"k": "v"}
    wf = WorkflowGraph(name="hook_ctx")
    wf.add_function_node("a", fn=lambda msg, ctx: msg)
    wf.add_edge(START, "a")
    wf.add_edge("a", END)

    runner = FakeRunnerAdapter()
    await wf._interpret(runner, "x", context=ctx_obj, on_node_end=on_end)
    assert seen == [ctx_obj]


# ----------------------------------------------------------------------
# default_input_filter（C-10）
# ----------------------------------------------------------------------
@dataclass
class _FakeHandoffData:
    """HandoffInputData 相当の最小スタブ（input_history + clone）。"""

    input_history: Any

    def clone(self, *, input_history: Any) -> _FakeHandoffData:
        return _FakeHandoffData(input_history=input_history)


def test_default_input_filter_trims_to_limit() -> None:
    """既定 input_filter は input_history を直近 N 件へ切り詰める。"""
    filt = default_input_filter(limit=2)
    data = _FakeHandoffData(input_history=[1, 2, 3, 4, 5])
    trimmed = filt(data)
    assert trimmed.input_history == (4, 5)


def test_default_input_filter_default_limit_is_one() -> None:
    """既定 limit は 1（直近 1 件）。"""
    filt = default_input_filter()
    data = _FakeHandoffData(input_history=("a", "b", "c"))
    assert filt(data).input_history == ("c",)


def test_default_input_filter_passes_short_history() -> None:
    """履歴が limit 以下ならそのまま透過する（clone しない）。"""
    filt = default_input_filter(limit=3)
    data = _FakeHandoffData(input_history=[1, 2])
    assert filt(data) is data


def test_default_input_filter_passes_non_sequence_history() -> None:
    """input_history が list/tuple でない（文字列等）場合はそのまま透過する。"""
    filt = default_input_filter(limit=1)
    data = _FakeHandoffData(input_history="not-a-sequence")
    assert filt(data) is data


# ----------------------------------------------------------------------
# _exec_node の未登録ノードガード
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_unregistered_node_in_edge_raises() -> None:
    """通常エッジが未登録ノードを指す場合、実行時に ValueError。"""
    wf = WorkflowGraph(name="dangling")
    wf.add_function_node("a", fn=lambda msg, ctx: msg)
    wf.add_edge(START, "a")
    wf.edges["a"] = ["ghost"]  # 登録されていないノードへのエッジ。

    runner = FakeRunnerAdapter()
    with pytest.raises(ValueError, match="未登録の node を実行"):
        await wf._interpret(runner, "x")
