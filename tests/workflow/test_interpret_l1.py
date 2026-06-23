"""L1（agents 非依存）: 内部インタプリタの制御フローを FakeRunnerAdapter で検証する。

node/edge 方式を網羅する: add_edge 順次 / fan-out 並行 / fan-in（dict 合流・合流先
FUNCTION）/ conditional（router -> mapping）/ ループ（recursion_limit 超過で実行時エラー）/
START 単一エントリ / 最終出力 = END へ到達したノードの出力 / FUNCTION sync・async /
前段出力 + 共有 context 透過 / fan-out + session の fail-fast。SDK（`agents`）には依存
しない（runner シームへ fake を注入し内部インタプリタを直接検証する）。
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from oai_agentspec.workflow import END, START, NodeResults, WorkflowGraph

from _helpers.fake_runner import FakeRunnerAdapter, FakeSession

pytestmark = pytest.mark.unit


# ----------------------------------------------------------------------
# add_edge: 順次
def _moderation_wf() -> WorkflowGraph:
    """classify が条件 fan-out し、merge が動的 fan-in する検証用ワークフロー。"""

    def route(msg: str, ctx: object) -> list:
        targets = []
        if "img" in msg:
            targets.append("image_check")
        if "url" in msg:
            targets.append("link_check")
        return targets or [END]

    wf = WorkflowGraph(name="moderation")
    wf.add_function_node("classify", fn=lambda msg, ctx: msg)
    wf.add_function_node("image_check", fn=lambda msg, ctx: "image-ok")
    wf.add_function_node("link_check", fn=lambda msg, ctx: "link-ok")
    wf.add_function_node("merge", fn=lambda inputs, ctx: "OK:" + ",".join(sorted(inputs)))
    wf.add_edge(START, "classify")
    wf.add_conditional_edges("classify", route)
    wf.add_fan_in_edge(["image_check", "link_check"], "merge")
    wf.add_edge("merge", END)
    return wf


@pytest.mark.asyncio
async def test_conditional_fan_out_full_set_runs_both_in_parallel() -> None:
    """条件 fan-out で 2 枝とも起動すると fan-in は両方を待って dict 合流する。"""
    wf = _moderation_wf()
    result = await wf._interpret(FakeRunnerAdapter({}), "img and url")
    assert result.final_output == "OK:image_check,link_check"


@pytest.mark.asyncio
async def test_conditional_fan_out_subset_does_not_deadlock() -> None:
    """部分集合（1 枝のみ）でも fan-in は動的に 1 件だけ待ち、走った枝のみ dict に入る。"""
    wf = _moderation_wf()
    one = await wf._interpret(FakeRunnerAdapter({}), "img only")
    assert one.final_output == "OK:image_check"  # link_check はキーごと omit
    none = await wf._interpret(FakeRunnerAdapter({}), "no signals")
    assert none.final_output == "no signals"  # [END] で終端


# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_conditional_default_routes_unmatched_key() -> None:
    """mapping に無いキーは default の行き先へ分岐する（MS Agent Framework の Default 相当）。"""
    wf = WorkflowGraph(name="cond_default")
    wf.add_function_node("c", fn=lambda msg, ctx: msg)
    wf.add_function_node("fallback", fn=lambda msg, ctx: f"default:{msg}")
    wf.add_edge(START, "c")
    wf.add_conditional_edges("c", lambda msg, ctx: "unknown", {"x": "fallback"}, default="fallback")
    wf.add_edge("fallback", END)
    result = await wf._interpret(FakeRunnerAdapter({}), "q")
    assert result.final_output == "default:q"


@pytest.mark.asyncio
async def test_conditional_without_mapping_returns_node_name_directly() -> None:
    """mapping=None なら router の戻り値を次ノード名 | END として直接使う（LangGraph 相当）。"""
    wf = WorkflowGraph(name="cond_direct")
    wf.add_function_node("c", fn=lambda msg, ctx: msg)
    wf.add_function_node("big", fn=lambda msg, ctx: f"BIG:{msg}")
    wf.add_edge(START, "c")
    wf.add_conditional_edges("c", lambda msg, ctx: "big" if len(str(msg)) >= 3 else END)
    wf.add_edge("big", END)
    assert (await wf._interpret(FakeRunnerAdapter({}), "abcd")).final_output == "BIG:abcd"
    assert (await wf._interpret(FakeRunnerAdapter({}), "ab")).final_output == "ab"


@pytest.mark.asyncio
async def test_add_edge_runs_nodes_in_order() -> None:
    """add_edge で AGENT -> FUNCTION -> END を順に実行し最終出力を返す。"""
    wf = WorkflowGraph(name="seq")
    wf.add_agent_node("classify", agent="classifier")
    wf.add_function_node("format", fn=lambda msg, ctx: f"[{msg}]")
    wf.add_edge(START, "classify")
    wf.add_edge("classify", "format")
    wf.add_edge("format", END)

    runner = FakeRunnerAdapter({"classifier": "billing"})
    result = await wf._interpret(runner, "question")

    assert result.final_output == "[billing]"
    assert result.results.outputs == {"classify": "billing", "format": "[billing]"}
    assert wf.entry == "classify"


@pytest.mark.asyncio
async def test_start_input_flows_to_first_node() -> None:
    """START 直後のノードは Runner.run 入力（初期 input）を msg として受ける。"""
    wf = WorkflowGraph(name="entry_input")
    wf.add_agent_node("a", agent="agent_a")
    wf.add_edge(START, "a")
    wf.add_edge("a", END)

    runner = FakeRunnerAdapter({"agent_a": "out"})
    await wf._interpret(runner, "seed-input")

    assert runner.calls[0].input == "seed-input"


@pytest.mark.asyncio
async def test_prev_output_and_shared_context_flow_to_nodes() -> None:
    """前段出力が次ノードの msg に、共有 context が各ノードへ素通しされる。"""
    seen: list[tuple[str, object]] = []

    def capture(msg: object, ctx: object) -> str:
        seen.append((str(msg), ctx))
        return f"{msg}->fn"

    wf = WorkflowGraph(name="ctx")
    wf.add_agent_node("a", agent="agent_a")
    wf.add_function_node("b", fn=capture)
    wf.add_edge(START, "a")
    wf.add_edge("a", "b")
    wf.add_edge("b", END)

    ctx_obj = {"tenant": "acme"}
    runner = FakeRunnerAdapter({"agent_a": lambda inp, ctx: f"{inp}|{ctx['tenant']}"})
    result = await wf._interpret(runner, "start", context=ctx_obj)

    # AGENT は input=初期入力・context=共有 context を受け取る。
    assert runner.calls[0].input == "start"
    assert runner.calls[0].context is ctx_obj
    # FUNCTION は前段(AGENT)出力を msg に、共有 context を受け取る。
    assert seen == [("start|acme", ctx_obj)]
    assert result.final_output == "start|acme->fn"


# ----------------------------------------------------------------------
# FUNCTION ノード（sync / async）
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_function_node_sync_and_async() -> None:
    """FUNCTION ノードは sync / async どちらの callable も await して扱う。"""

    async def async_double(msg: object, ctx: object) -> int:
        return int(msg) * 2

    wf = WorkflowGraph(name="fn")
    wf.add_function_node("inc", fn=lambda msg, ctx: int(msg) + 1)
    wf.add_function_node("double", fn=async_double)
    wf.add_edge(START, "inc")
    wf.add_edge("inc", "double")
    wf.add_edge("double", END)

    runner = FakeRunnerAdapter()
    result = await wf._interpret(runner, 10)

    assert result.results.outputs["inc"] == 11
    assert result.final_output == 22


# ----------------------------------------------------------------------
# fan-out（並行）+ fan-in（dict 合流・合流先 FUNCTION）
# ----------------------------------------------------------------------
def _make_fan_wf() -> WorkflowGraph:
    """src -> (left, right) を fan-out し merge(FUNCTION) で fan-in 合流するグラフ。"""
    wf = WorkflowGraph(name="fan")
    wf.add_agent_node("src", agent="src_agent")
    wf.add_agent_node("left", agent="left_agent")
    wf.add_agent_node("right", agent="right_agent")
    wf.add_function_node("merge", fn=lambda msg, ctx: f"{msg['left']}+{msg['right']}")
    wf.add_edge(START, "src")
    wf.add_edge("src", "left")
    wf.add_edge("src", "right")  # 同一 src 複数で fan-out。
    wf.add_fan_in_edge(["left", "right"], "merge")
    wf.add_edge("merge", END)
    return wf


@pytest.mark.asyncio
async def test_fan_out_and_fan_in_dict_merge() -> None:
    """fan-out が各枝を並行実行し fan-in 合流先 FUNCTION が dict で受け取る。"""
    wf = _make_fan_wf()
    runner = FakeRunnerAdapter(
        {
            "src_agent": "seed",
            "left_agent": lambda inp, ctx: f"L:{inp}",
            "right_agent": lambda inp, ctx: f"R:{inp}",
        }
    )
    result = await wf._interpret(runner, "go")

    # merge は {source名: 出力} の dict を受け取り合流する（C-4）。
    assert result.final_output == "L:seed+R:seed"
    assert result.results.outputs["left"] == "L:seed"
    assert result.results.outputs["right"] == "R:seed"


@pytest.mark.asyncio
async def test_fan_in_receives_source_named_dict() -> None:
    """fan-in 合流先の msg は {source名: 出力} の dict そのものである。"""
    captured: dict[str, object] = {}

    def merge(msg: dict[str, object], ctx: object) -> str:
        captured.update(msg)
        return "merged"

    wf = WorkflowGraph(name="fan_dict")
    wf.add_agent_node("src", agent="s")
    wf.add_agent_node("x", agent="ax")
    wf.add_agent_node("y", agent="ay")
    wf.add_function_node("merge", fn=merge)
    wf.add_edge(START, "src")
    wf.add_edge("src", "x")
    wf.add_edge("src", "y")
    wf.add_fan_in_edge(["x", "y"], "merge")
    wf.add_edge("merge", END)

    runner = FakeRunnerAdapter({"s": "s0", "ax": "vx", "ay": "vy"})
    await wf._interpret(runner, "in")

    assert captured == {"x": "vx", "y": "vy"}


@pytest.mark.asyncio
async def test_nested_fan_in_does_not_drop_inner_merge() -> None:
    """多段 fan-in: 浅い枝(b)が深い fan-in(m2)に先着しても内側合流(m1)出力を取りこぼさない。

    topology: split fan-out -> {a, b}; a fan-out -> {c, d}; fan_in([c, d]) -> m1;
    fan_in([m1, b]) -> m2。b は m2 に浅く先着するため、m1 の合流完了前に m2 が早期発火
    すると m1 出力が dict から欠落する（activated 登録レース・回帰防止）。
    """
    captured: dict[str, object] = {}

    def m2_merge(inputs: dict[str, object], ctx: object) -> str:
        captured.update(inputs)
        return ",".join(sorted(inputs))

    wf = WorkflowGraph(name="nested_fanin")
    wf.add_function_node("split", fn=lambda msg, ctx: msg)
    wf.add_function_node("a", fn=lambda msg, ctx: "A")
    wf.add_function_node("b", fn=lambda msg, ctx: "B")
    wf.add_function_node("c", fn=lambda msg, ctx: "C")
    wf.add_function_node("d", fn=lambda msg, ctx: "D")
    wf.add_function_node("m1", fn=lambda inp, ctx: "M1")
    wf.add_function_node("m2", fn=m2_merge)
    wf.add_edge(START, "split")
    wf.add_edge("split", "a")
    wf.add_edge("split", "b")
    wf.add_edge("a", "c")
    wf.add_edge("a", "d")
    wf.add_fan_in_edge(["c", "d"], "m1")
    wf.add_fan_in_edge(["m1", "b"], "m2")
    wf.add_edge("m2", END)

    result = await wf._interpret(FakeRunnerAdapter({}), "go")

    assert set(captured) == {"m1", "b"}
    assert result.final_output == "b,m1"


@pytest.mark.asyncio
async def test_fan_in_waits_for_conditional_source_branch() -> None:
    """条件エッジ経由でしか到達しない fan-in ソースを、浅い枝が先着しても取りこぼさない。

    split fan-out -> {a(async sleep で遅延), b}; a -conditional-> x; fan_in([x, b]) -> m。
    b が m へ先着しても、まだ条件分岐が未解決の a（frontier）から x が到達可能なため m は
    早期発火せず、x の合流を待つ（WF-CONC-01 回帰防止）。
    """
    captured: dict[str, Any] = {}

    async def slow_a(msg: str, ctx: Any) -> str:
        await asyncio.sleep(0.02)  # b を先に m へ到達させる
        return "A"

    def m_merge(inputs: dict[str, Any], ctx: Any) -> str:
        captured.update(inputs)
        return ",".join(sorted(inputs))

    wf = WorkflowGraph("cond_fanin")
    wf.add_function_node("split", fn=lambda msg, ctx: msg)
    wf.add_function_node("a", fn=slow_a)
    wf.add_function_node("b", fn=lambda msg, ctx: "B")
    wf.add_function_node("x", fn=lambda msg, ctx: "X")
    wf.add_function_node("m", fn=m_merge)
    wf.add_edge(START, "split")
    wf.add_edge("split", "a")
    wf.add_edge("split", "b")
    wf.add_conditional_edges("a", lambda msg, ctx: "x", {"x": "x"})
    wf.add_fan_in_edge(["x", "b"], "m")
    wf.add_edge("m", END)

    result = await wf._interpret(FakeRunnerAdapter({}), "go")

    assert set(captured) == {"x", "b"}
    assert result.final_output == "b,x"


@pytest.mark.asyncio
async def test_loop_with_conditional_fan_in_does_not_leak_pending() -> None:
    """ループ内の条件 fan-out + fan-in で、反復ごとに正しく発火し _PENDING を漏らさない。

    classify -cond(1回目['a'] / 2回目['b'] / 以降[END])->; fan_in([a, b]) -> m -loop-> classify。
    activated がループ反復をまたいで累積すると required を過大評価し m がデッドロック → _PENDING
    が最終出力へ漏出する（WF-CONC-02 回帰防止）。
    """
    fired: list[str] = []
    calls = {"n": 0}

    def route(msg: str, ctx: Any) -> list[Any]:
        calls["n"] += 1
        if calls["n"] == 1:
            return ["a"]
        if calls["n"] == 2:
            return ["b"]
        return [END]

    def m_merge(inputs: dict[str, Any], ctx: Any) -> str:
        fired.append(",".join(sorted(inputs)))
        return "merged"

    wf = WorkflowGraph("loop_fanin")
    wf.add_function_node("classify", fn=lambda msg, ctx: msg)
    wf.add_function_node("a", fn=lambda msg, ctx: "A")
    wf.add_function_node("b", fn=lambda msg, ctx: "B")
    wf.add_function_node("m", fn=m_merge)
    wf.add_edge(START, "classify")
    wf.add_conditional_edges("classify", route, candidates=["a", "b", END])
    wf.add_fan_in_edge(["a", "b"], "m")
    wf.add_edge("m", "classify")

    result = await wf._interpret(FakeRunnerAdapter({}), "go")

    # 反復1で {a}、反復2で {b} がそれぞれ発火する（累積による取りこぼしなし）。
    assert fired == ["a", "b"]
    assert "PENDING" not in str(result.final_output)


@pytest.mark.asyncio
async def test_fan_out_with_session_fails_fast() -> None:
    """fan-out（並列）と session を併用すると実行時 ValueError（FR-15・並行安全非保証）。"""
    wf = WorkflowGraph(name="fan_sess", run_defaults={"session": FakeSession()})
    wf.add_agent_node("src", agent="src_agent")
    wf.add_agent_node("left", agent="left_agent")
    wf.add_agent_node("right", agent="right_agent")
    wf.add_function_node("merge", fn=lambda msg, ctx: msg)
    wf.add_edge(START, "src")
    wf.add_edge("src", "left")
    wf.add_edge("src", "right")
    wf.add_fan_in_edge(["left", "right"], "merge")
    wf.add_edge("merge", END)

    runner = FakeRunnerAdapter({"src_agent": "x", "left_agent": "y", "right_agent": "z"})
    with pytest.raises(ValueError, match="fan-out（並列）と session"):
        await wf._interpret(runner, "in")


@pytest.mark.asyncio
async def test_conditional_fan_out_with_session_fails_fast() -> None:
    """条件 fan-out（router がリスト返し）+ session も実行時に fail-fast する（FR-15）。

    通常エッジ由来の fan-out が無いため run-entry の静的ガードはすり抜けるが、実行時に
    複数枝を並行起動する時点で拒否される（Codex 指摘の取りこぼし対策）。
    """
    wf = WorkflowGraph(name="cond_fan_sess", run_defaults={"session": FakeSession()})
    wf.add_function_node("classify", fn=lambda msg, ctx: msg)
    wf.add_function_node("a", fn=lambda msg, ctx: "a")
    wf.add_function_node("b", fn=lambda msg, ctx: "b")
    wf.add_function_node("merge", fn=lambda inputs, ctx: ",".join(sorted(inputs)))
    wf.add_edge(START, "classify")
    wf.add_conditional_edges("classify", lambda msg, ctx: ["a", "b"], candidates=["a", "b"])
    wf.add_fan_in_edge(["a", "b"], "merge")
    wf.add_edge("merge", END)

    with pytest.raises(ValueError, match="fan-out（並列）と session"):
        await wf._interpret(FakeRunnerAdapter({}), "in")


@pytest.mark.asyncio
async def test_agent_node_receives_session() -> None:
    """session 指定時（fan-out 無し）、AGENT ノードへ session が素通しされる。"""
    session = FakeSession()
    wf = WorkflowGraph(name="sess", run_defaults={"session": session})
    wf.add_agent_node("a", agent="agent_a")
    wf.add_edge(START, "a")
    wf.add_edge("a", END)

    runner = FakeRunnerAdapter({"agent_a": "out"})
    await wf._interpret(runner, "in")

    assert runner.calls[0].session is session


# ----------------------------------------------------------------------
# Runner パラメータ passthrough（run_defaults / run_options・FR-15）
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_run_defaults_passed_to_all_agent_nodes() -> None:
    """グラフ run_defaults は全 AGENT ノードの Runner.run へ素通しされる。"""
    wf = WorkflowGraph(name="defaults", run_defaults={"max_turns": 7})
    wf.add_agent_node("a", agent="agent_a")
    wf.add_agent_node("b", agent="agent_b")
    wf.add_edge(START, "a")
    wf.add_edge("a", "b")
    wf.add_edge("b", END)

    runner = FakeRunnerAdapter({"agent_a": "x", "agent_b": "y"})
    await wf._interpret(runner, "in")

    assert runner.calls[0].max_turns == 7
    assert runner.calls[1].max_turns == 7


@pytest.mark.asyncio
async def test_node_run_options_override_run_defaults() -> None:
    """ノード run_options はグラフ run_defaults を dict マージで上書きする。"""
    wf = WorkflowGraph(name="override", run_defaults={"max_turns": 5})
    wf.add_agent_node("a", agent="agent_a", run_options={"max_turns": 99})
    wf.add_agent_node("b", agent="agent_b")
    wf.add_edge(START, "a")
    wf.add_edge("a", "b")
    wf.add_edge("b", END)

    runner = FakeRunnerAdapter({"agent_a": "x", "agent_b": "y"})
    await wf._interpret(runner, "in")

    assert runner.calls[0].max_turns == 99  # run_options が上書き
    assert runner.calls[1].max_turns == 5  # run_defaults を継承


def test_run_defaults_rejects_reserved_keys() -> None:
    """run_defaults に予約キー（input / context）を入れると構築時に ValueError。"""
    with pytest.raises(ValueError, match="run_defaults に予約キー"):
        WorkflowGraph(name="bad", run_defaults={"context": object()})


def test_node_run_options_reject_session() -> None:
    """ノード run_options に session を入れると ValueError（session はグラフ既定のみ）。"""
    wf = WorkflowGraph(name="bad_node")
    with pytest.raises(ValueError, match="run_options に予約キー"):
        wf.add_agent_node("a", agent="agent_a", run_options={"session": object()})


def test_node_run_options_reject_input_context() -> None:
    """ノード run_options に input / context を入れると ValueError（lib 管理）。"""
    wf = WorkflowGraph(name="bad_node2")
    with pytest.raises(ValueError, match="run_options に予約キー"):
        wf.add_agent_node("a", agent="agent_a", run_options={"input": "x"})


# ----------------------------------------------------------------------
# conditional（router -> mapping）
# ----------------------------------------------------------------------
def _make_conditional_wf() -> WorkflowGraph:
    wf = WorkflowGraph(name="cond")
    wf.add_agent_node("route", agent="router")
    wf.add_function_node("to_billing", fn=lambda msg, ctx: f"billing:{msg}")
    wf.add_function_node("to_support", fn=lambda msg, ctx: f"support:{msg}")
    wf.add_edge(START, "route")
    wf.add_conditional_edges(
        "route",
        lambda msg, ctx: msg,
        {"billing": "to_billing", "support": "to_support"},
    )
    wf.add_edge("to_billing", END)
    wf.add_edge("to_support", END)
    return wf


@pytest.mark.asyncio
async def test_conditional_selects_mapped_node() -> None:
    """conditional の router 判定キーが mapping のキーを選び次ノードへ進む。"""
    wf = _make_conditional_wf()
    runner = FakeRunnerAdapter({"router": "support"})
    result = await wf._interpret(runner, "q")
    assert result.final_output == "support:support"


@pytest.mark.asyncio
async def test_conditional_routes_to_end() -> None:
    """conditional の mapping 値が END のとき、その経路は分岐元出力を最終出力にする。"""
    wf = WorkflowGraph(name="cond_end")
    wf.add_agent_node("route", agent="router")
    wf.add_function_node("more", fn=lambda msg, ctx: f"more:{msg}")
    wf.add_edge(START, "route")
    wf.add_conditional_edges(
        "route",
        lambda msg, ctx: msg,
        {"stop": END, "go": "more"},
    )
    wf.add_edge("more", END)

    runner = FakeRunnerAdapter({"router": "stop"})
    result = await wf._interpret(runner, "q")
    assert result.final_output == "stop"


@pytest.mark.asyncio
async def test_conditional_unmatched_key_raises() -> None:
    """router が mapping に無いキーを返すと実行時 ValueError。"""
    wf = _make_conditional_wf()
    runner = FakeRunnerAdapter({"router": "unknown"})
    with pytest.raises(ValueError, match="mapping に解決せず default もありません"):
        await wf._interpret(runner, "q")


@pytest.mark.asyncio
async def test_conditional_router_receives_output_and_context() -> None:
    """conditional の router は分岐元の出力と共有 context を受け取る。"""
    captured: dict[str, object] = {}

    def router(msg: object, ctx: object) -> str:
        captured["msg"] = msg
        captured["ctx"] = ctx
        return "a"

    wf = WorkflowGraph(name="cond_args")
    wf.add_agent_node("route", agent="router")
    wf.add_function_node("to_a", fn=lambda msg, ctx: msg)
    wf.add_edge(START, "route")
    wf.add_conditional_edges("route", router, {"a": "to_a"})
    wf.add_edge("to_a", END)

    ctx_obj = object()
    runner = FakeRunnerAdapter({"router": "out"})
    await wf._interpret(runner, "q", context=ctx_obj)
    assert captured == {"msg": "out", "ctx": ctx_obj}


# ----------------------------------------------------------------------
# ループ（戻りエッジ + conditional で END へ抜ける・recursion_limit 超過で例外）
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_loop_runs_until_conditional_exits() -> None:
    """戻りエッジ + conditional で END へ抜けるループが正しく終了し最終出力を返す。"""
    wf = WorkflowGraph(name="lp", recursion_limit=25)
    wf.add_function_node("tick", fn=lambda msg, ctx: msg + 1)
    wf.add_edge(START, "tick")
    # tick の出力が 3 未満なら tick へ戻り、3 以上で END へ抜ける。
    wf.add_conditional_edges(
        "tick",
        lambda msg, ctx: "loop" if msg < 3 else "exit",
        {"loop": "tick", "exit": END},
    )

    runner = FakeRunnerAdapter()
    result = await wf._interpret(runner, 0)

    # 0 -> 1 -> 2 -> 3 で exit。
    assert result.results.outputs["tick"] == 3
    assert result.final_output == 3


@pytest.mark.asyncio
async def test_loop_exceeding_recursion_limit_raises() -> None:
    """常にループへ戻る場合 recursion_limit 超過で実行時 ValueError（C-5）。"""
    wf = WorkflowGraph(name="lp_over", recursion_limit=3)
    wf.add_function_node("tick", fn=lambda msg, ctx: msg)
    wf.add_edge(START, "tick")
    wf.add_conditional_edges("tick", lambda msg, ctx: "loop", {"loop": "tick", "exit": END})

    runner = FakeRunnerAdapter()
    with pytest.raises(ValueError, match="recursion_limit=3 を超過"):
        await wf._interpret(runner, "x")


# ----------------------------------------------------------------------
# START 単一エントリ
# ----------------------------------------------------------------------
def test_start_single_entry_enforced() -> None:
    """START からのエッジを 2 本張ると ValueError（単一エントリ・FR-2）。"""
    wf = WorkflowGraph(name="dual_start")
    wf.add_function_node("a", fn=lambda msg, ctx: msg)
    wf.add_function_node("b", fn=lambda msg, ctx: msg)
    wf.add_edge(START, "a")
    with pytest.raises(ValueError, match="START からのエッジは 1 本のみ"):
        wf.add_edge(START, "b")


def test_start_same_entry_twice_is_idempotent() -> None:
    """同じ entry への START エッジ再宣言は冪等（競合扱いしない）。"""
    wf = WorkflowGraph(name="same_start")
    wf.add_function_node("a", fn=lambda msg, ctx: msg)
    wf.add_edge(START, "a")
    wf.add_edge(START, "a")  # 例外にならない。
    assert wf.entry == "a"


@pytest.mark.asyncio
async def test_interpret_without_entry_raises() -> None:
    """entry 未設定のワークフローは実行時 ValueError。"""
    wf = WorkflowGraph(name="empty")
    runner = FakeRunnerAdapter()
    with pytest.raises(ValueError, match="START からのエントリが未設定"):
        await wf._interpret(runner, "x")


# ----------------------------------------------------------------------
# 最終出力 = END へ到達したノードの出力
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_final_output_is_end_reaching_node() -> None:
    """最終出力は END へ到達したノードの出力である（途中ノード出力ではない）。"""
    wf = WorkflowGraph(name="final")
    wf.add_function_node("first", fn=lambda msg, ctx: "FIRST")
    wf.add_function_node("last", fn=lambda msg, ctx: f"LAST<{msg}>")
    wf.add_edge(START, "first")
    wf.add_edge("first", "last")
    wf.add_edge("last", END)

    runner = FakeRunnerAdapter()
    result = await wf._interpret(runner, "in")
    assert result.final_output == "LAST<FIRST>"


@pytest.mark.asyncio
async def test_node_without_out_edge_returns_its_output() -> None:
    """出辺の無いノード（END へ繋がない）はその出力をそのまま返す。"""
    wf = WorkflowGraph(name="dangling_out")
    wf.add_function_node("a", fn=lambda msg, ctx: f"A:{msg}")
    wf.add_edge(START, "a")  # a から先のエッジは張らない。

    runner = FakeRunnerAdapter()
    result = await wf._interpret(runner, "in")
    assert result.final_output == "A:in"


# ----------------------------------------------------------------------
# 重複ノード名
# ----------------------------------------------------------------------
def test_duplicate_node_name_raises() -> None:
    """同名ノードの二重登録は ValueError。"""
    wf = WorkflowGraph(name="dup")
    wf.add_agent_node("a", agent="x")
    with pytest.raises(ValueError, match="node が重複"):
        wf.add_agent_node("a", agent="y")


def test_duplicate_function_node_name_raises() -> None:
    """FUNCTION ノードでも同名の二重登録は ValueError。"""
    wf = WorkflowGraph(name="dup_fn")
    wf.add_function_node("a", fn=lambda msg, ctx: msg)
    with pytest.raises(ValueError, match="node が重複"):
        wf.add_function_node("a", fn=lambda msg, ctx: msg)


# ----------------------------------------------------------------------
# NodeResults
# ----------------------------------------------------------------------
def test_node_results_record_and_get() -> None:
    """NodeResults.record は outputs を追記し get は記録値（または default）を返す。"""
    results = NodeResults()
    results.record("a", 1)
    results.record("b", 2)
    assert results.outputs == {"a": 1, "b": 2}
    assert results.get("a") == 1
    assert results.get("missing") is None
    assert results.get("missing", "fallback") == "fallback"


# ----------------------------------------------------------------------
# メソッドチェーン
# ----------------------------------------------------------------------
def test_declaration_methods_return_self_for_chaining() -> None:
    """宣言メソッドは self を返しチェーンできる（FR-2）。"""
    wf = WorkflowGraph(name="chain")
    returned = (
        wf.add_function_node("a", fn=lambda msg, ctx: msg).add_edge(START, "a").add_edge("a", END)
    )
    assert returned is wf
