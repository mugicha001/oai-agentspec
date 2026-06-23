"""L2: 実 Agent + FakeModel + `from agents import Runner` + Runner.run で経路 A/C を検証。

経路C（as_agent_spec）: WorkflowModel を据えた Agent を Runner.run で決定論起動し、内部
AGENT ノード（FakeModel ベース）を回す。START 入力が素テキスト化される回帰（FR-10・
[{CONTENT:..}] 問題の解消）を含む。経路A（as_facade_spec）: tool_choice を model_settings
に積むこと（extra ではない・FR-9）と ToolContext.context 透過の回帰。
回帰: tool_choice を extra に入れると build_agent が ValueError（FR-9 ブロッカーの恒久ガード）。
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, replace
from typing import Any

import pytest
from agents import MaxTurnsExceeded, Runner

from oai_agentspec import AgentRegistry, AgentSpec
from oai_agentspec._adapters import build_agent
from oai_agentspec.workflow import END, START, WorkflowGraph

from _helpers.fake_model import ChoiceAwareModel, FakeModel

pytestmark = pytest.mark.integration


def _reg_with_models(**name_to_model: FakeModel) -> AgentRegistry:
    """FakeModel を据えた AGENT を登録した registry を返す。"""
    reg = AgentRegistry()
    for name, model in name_to_model.items():
        reg.register(AgentSpec(name=name, instructions=name, model=model))
    return reg


# ----------------------------------------------------------------------
# 経路C: as_agent_spec → Runner.run（決定論起動・AGENT/FUNCTION 混在）
# ----------------------------------------------------------------------
def _text_delta_count(events: list[Any]) -> tuple[int, str]:
    """stream_events から text-delta 件数と連結テキストを取り出す（テスト補助）。"""
    from openai.types.responses import ResponseTextDeltaEvent

    deltas = [
        e.data.delta
        for e in events
        if e.type == "raw_response_event" and isinstance(e.data, ResponseTextDeltaEvent)
    ]
    return len(deltas), "".join(deltas)


@pytest.mark.asyncio
async def test_path_c_run_streamed_emits_text_deltas() -> None:
    """経路C は Runner.run_streamed で最終出力を text-delta + completed として流す（offline）。"""
    wf = WorkflowGraph("stream_c")
    wf.add_function_node("a", fn=lambda msg, ctx: msg.upper())
    wf.add_function_node("b", fn=lambda msg, ctx: f"[{msg}]")
    wf.add_edge(START, "a")
    wf.add_edge("a", "b")
    wf.add_edge("b", END)
    agent = build_agent(wf.as_agent_spec("stream_c_agent"))

    streamed = Runner.run_streamed(agent, input="hi")
    events = [e async for e in streamed.stream_events()]
    count, text = _text_delta_count(events)

    assert streamed.final_output == "[HI]"
    assert count > 0  # トークン（擬似）が流れた
    assert text == "[HI]"  # 流れたテキストが最終出力と一致


@pytest.mark.asyncio
async def test_path_d_run_streamed_does_not_crash() -> None:
    """経路D（DETERMINISTIC）は Runner.run_streamed でクラッシュせず最終出力を返す（offline）。"""
    wf = WorkflowGraph("stream_d")
    wf.add_function_node("a", fn=lambda msg, ctx: f"<{msg}>")
    wf.add_edge(START, "a")
    wf.add_edge("a", END)
    from oai_agentspec import FacadeMode

    agent = build_agent(wf.as_facade_spec("stream_d_agent", mode=FacadeMode.DETERMINISTIC))

    streamed = Runner.run_streamed(agent, input="hi")
    async for _ in streamed.stream_events():
        pass

    assert streamed.final_output == "<hi>"


@pytest.mark.asyncio
async def test_path_c_agent_spec_runs_deterministically() -> None:
    """as_agent_spec の WorkflowModel が LLM を呼ばず内部 AGENT ノードを回す。"""
    inner_model = FakeModel().queue_text("classified")
    reg = _reg_with_models(classifier=inner_model)

    wf = WorkflowGraph(name="pipeline")
    wf.add_agent_node("classify", agent="classifier")
    wf.add_function_node("format", fn=lambda msg, ctx: f"<{msg}>")
    wf.add_edge(START, "classify")
    wf.add_edge("classify", "format")
    wf.add_edge("format", END)

    spec = wf.as_agent_spec("pipeline_agent", registry=reg)
    agent = build_agent(spec)

    result = await Runner.run(agent, input="question")

    # 内部 AGENT は FakeModel で 1 回呼ばれ、FUNCTION で整形された最終出力が返る。
    assert result.final_output == "<classified>"
    assert len(inner_model.calls) == 1


@pytest.mark.asyncio
async def test_path_c_function_only_workflow_runs_offline() -> None:
    """FUNCTION のみのワークフローは LLM を一切呼ばず offline 実行される（経路C）。"""
    reg = AgentRegistry()
    wf = WorkflowGraph(name="fn_only")
    wf.add_function_node("upper", fn=lambda msg, ctx: str(msg).upper())
    wf.add_edge(START, "upper")
    wf.add_edge("upper", END)

    spec = wf.as_agent_spec("fn_only_agent", registry=reg)
    agent = build_agent(spec)
    result = await Runner.run(agent, input="hello")
    assert result.final_output == "HELLO"


@pytest.mark.asyncio
async def test_path_c_start_input_is_plain_text() -> None:
    """START 入力が素テキスト化され、先頭ノードが生メッセージ列でなく素のテキストを受ける。

    SDK の Runner.run は input をメッセージ列へ正規化するが、WorkflowModel は最新の
    user テキストを取り出して START 入力にする（FR-10・[{CONTENT:..}] 問題の回帰）。
    """
    seen: dict[str, Any] = {}

    def capture(msg: object, ctx: object) -> str:
        seen["msg"] = msg
        return f"echo:{msg}"

    reg = AgentRegistry()
    wf = WorkflowGraph(name="plain")
    wf.add_function_node("head", fn=capture)
    wf.add_edge(START, "head")
    wf.add_edge("head", END)

    spec = wf.as_agent_spec("plain_agent", registry=reg)
    agent = build_agent(spec)
    result = await Runner.run(agent, input="raw question")

    # 先頭ノードは素のテキストを受け取る（dict/メッセージ列ではない）。
    assert seen["msg"] == "raw question"
    assert result.final_output == "echo:raw question"


@pytest.mark.asyncio
async def test_path_c_output_extractor_applied() -> None:
    """as_agent_spec の output_extractor が最終出力の文字列化に使われる。"""
    inner_model = FakeModel().queue_text("raw")
    reg = _reg_with_models(a=inner_model)

    wf = WorkflowGraph(name="wf_extract")
    wf.add_agent_node("a", agent="a")
    wf.add_edge(START, "a")
    wf.add_edge("a", END)

    spec = wf.as_agent_spec(
        "extract_agent",
        registry=reg,
        output_extractor=lambda out: f"EXTRACTED:{out}",
    )
    agent = build_agent(spec)
    result = await Runner.run(agent, input="in")
    assert result.final_output == "EXTRACTED:raw"


@pytest.mark.asyncio
async def test_path_c_registered_workflow_is_handoff_target() -> None:
    """as_agent_spec を registry.register すると本物の Agent として get でき run できる。"""
    inner_model = FakeModel().queue_text("done")
    reg = _reg_with_models(worker=inner_model)

    wf = WorkflowGraph(name="wf")
    wf.add_agent_node("worker", agent="worker")
    wf.add_edge(START, "worker")
    wf.add_edge("worker", END)
    reg.register(wf.as_agent_spec("wf_agent", registry=reg))

    agent = reg.get("wf_agent")
    result = await Runner.run(agent, input="go")
    assert result.final_output == "done"


@pytest.mark.asyncio
async def test_path_c_node_hooks_fire_under_runner() -> None:
    """as_agent_spec に渡した on_node_start/end が Runner.run 経由でも発火する（FR-13）。"""
    events: list[str] = []
    reg = AgentRegistry()

    wf = WorkflowGraph(name="hook_c")
    wf.add_function_node("a", fn=lambda msg, ctx: msg)
    wf.add_edge(START, "a")
    wf.add_edge("a", END)

    spec = wf.as_agent_spec(
        "hook_c_agent",
        registry=reg,
        on_node_start=lambda name, results, ctx: events.append(f"start:{name}"),
        on_node_end=lambda name, results, ctx: events.append(f"end:{name}"),
    )
    agent = build_agent(spec)
    await Runner.run(agent, input="x")
    assert events == ["start:a", "end:a"]


# ----------------------------------------------------------------------
# 経路A: as_facade_spec（tool_choice / context 透過 / 回帰ガード）
# ----------------------------------------------------------------------
def test_path_a_facade_sets_tool_choice_on_model_settings() -> None:
    """as_facade_spec は tool_choice='required' を model_settings に積む（extra ではない）。"""
    wf = WorkflowGraph(name="facade")
    wf.add_function_node("a", fn=lambda msg, ctx: msg)
    wf.add_edge(START, "a")
    wf.add_edge("a", END)
    spec = wf.as_facade_spec("facade_agent")

    assert spec.model_settings is not None
    assert spec.model_settings.tool_choice == "required"
    # tool_choice は extra に入れない（FR-9）。
    assert "tool_choice" not in spec.extra
    # tool_use_behavior は Agent フィールドなので extra に積む。
    assert spec.extra.get("tool_use_behavior") == "stop_on_first_tool"
    assert len(spec.tools) == 1


def test_path_a_facade_builds_into_real_agent() -> None:
    """as_facade_spec が build_agent を通り tool_choice/tool_use_behavior を持つ Agent になる。

    model_settings 経由なら build_agent の未知キーガードに弾かれず構築できる
    （FR-9 ブロッカーの恒久ガード）。
    """
    wf = WorkflowGraph(name="facade2")
    wf.add_function_node("a", fn=lambda msg, ctx: msg)
    wf.add_edge(START, "a")
    wf.add_edge("a", END)
    spec = wf.as_facade_spec("facade_agent2")

    agent = build_agent(spec)
    assert agent.model_settings.tool_choice == "required"
    assert agent.tool_use_behavior == "stop_on_first_tool"
    assert len(agent.tools) == 1


def test_regression_tool_choice_in_extra_raises() -> None:
    """tool_choice を extra に入れると build_agent が ValueError（FR-9 恒久ガード）。

    ModelSettings 経由でなく Agent extra へ tool_choice を積もうとすると、
    build_agent の未知キーガードが弾く（経路A が model_settings を選んだ理由の回帰）。
    """
    spec = AgentSpec(name="bad", instructions="x", extra={"tool_choice": "required"})
    with pytest.raises(ValueError, match="受け付けないキー"):
        build_agent(spec)


@pytest.mark.asyncio
async def test_path_a_tool_context_passes_through_to_nodes() -> None:
    """経路A の workflow tool が ToolContext.context を内部ノードへ透過する（回帰）。

    SDK の on_invoke_tool を直接呼び、tool_context.context が FUNCTION ノードの
    ctx 引数へ届くことを確認する（as_tool と異なり自動透過しない不変条件・FR-10）。
    """

    @dataclass
    class Ctx:
        tenant: str

    seen: dict[str, Any] = {}

    def capture(msg: object, ctx: Any) -> str:
        # ctx は RunContextWrapper（ToolContext）。利用者の object は ctx.context。
        seen["tenant"] = ctx.context.tenant
        return f"{msg}@{ctx.context.tenant}"

    wf = WorkflowGraph(name="ctx_facade")
    wf.add_function_node("node", fn=capture)
    wf.add_edge(START, "node")
    wf.add_edge("node", END)
    spec = wf.as_facade_spec("ctx_agent")
    tool = spec.tools[0]

    # ToolContext を最小構築して on_invoke_tool を直接呼ぶ。
    from agents import RunContextWrapper
    from agents.tool_context import ToolContext

    ctx_obj = Ctx(tenant="acme")
    wrapper: RunContextWrapper[Ctx] = RunContextWrapper(context=ctx_obj)
    tool_context = ToolContext(
        context=wrapper.context,
        usage=wrapper.usage,
        tool_name=tool.name,
        tool_call_id="call_test",
        tool_arguments='{"input": "hello"}',
    )

    output = await tool.on_invoke_tool(tool_context, '{"input": "hello"}')

    assert seen["tenant"] == "acme"
    assert output == "hello@acme"


def _facade_input_filter(registry: AgentRegistry, src: str, dst: str) -> Any:
    """src -> dst の handoff に実際に適用された input_filter を返す。

    設定付きエッジは SDK Handoff(input_filter) として、設定なしエッジは生 Agent として
    結線されるため、後者は input_filter なし（None）とみなす。
    """
    from agents import Handoff

    for h in registry.get(src).handoffs:
        if isinstance(h, Handoff) and h.agent_name == dst:
            return h.input_filter
        if getattr(h, "name", None) == dst:  # 設定なし -> 生 Agent（filter なし）
            return None
    raise AssertionError(f"{src} -> {dst} の handoff が見つからない")


def _facade_workflow() -> WorkflowGraph:
    wf = WorkflowGraph(name="hf")
    wf.add_function_node("a", fn=lambda msg, ctx: msg)
    wf.add_edge(START, "a")
    wf.add_edge("a", END)
    return wf


def test_path_a_connect_as_facade_applies_default_input_filter() -> None:
    """connect_as_facade は handoff エッジ（registry が読む場所）へ既定 input_filter を載せる。"""
    from oai_agentspec import HandoffGraph

    registry = AgentRegistry()
    registry.register(AgentSpec(name="triage", instructions="t", model=FakeModel()))
    graph = HandoffGraph(entry="triage")
    _facade_workflow().connect_as_facade(registry, graph, "hf_agent", "triage")
    graph.apply(registry)

    # 既定 input_filter が実際に handoff へ適用されている（死蔵でない）。
    assert _facade_input_filter(registry, "triage", "hf_agent") is not None


def test_path_a_connect_as_facade_explicit_none_opt_out() -> None:
    """input_filter=None を明示すると handoff に filter が載らない（全履歴流入・opt-in）。"""
    from oai_agentspec import HandoffGraph

    registry = AgentRegistry()
    registry.register(AgentSpec(name="triage", instructions="t", model=FakeModel()))
    graph = HandoffGraph(entry="triage")
    _facade_workflow().connect_as_facade(
        registry, graph, "hf_none_agent", "triage", input_filter=None
    )
    graph.apply(registry)
    assert _facade_input_filter(registry, "triage", "hf_none_agent") is None


def test_path_a_facade_custom_tool_name() -> None:
    """tool_name 明示時はその名前が workflow tool に使われる。"""
    wf = WorkflowGraph(name="named")
    wf.add_function_node("a", fn=lambda msg, ctx: msg)
    wf.add_edge(START, "a")
    wf.add_edge("a", END)
    spec = wf.as_facade_spec("named_agent", tool_name="run_pipeline")
    assert spec.tools[0].name == "run_pipeline"


def test_path_a_facade_default_tool_name_derived_from_name() -> None:
    """tool_name 未指定時は name 由来の既定名（<name>_workflow）になる。"""
    wf = WorkflowGraph(name="derive")
    wf.add_function_node("a", fn=lambda msg, ctx: msg)
    wf.add_edge(START, "a")
    wf.add_edge("a", END)
    spec = wf.as_facade_spec("derive_agent")
    assert spec.tools[0].name == "derive_agent_workflow"


# ----------------------------------------------------------------------
# WorkflowModel / adapter helpers の直接検証
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_workflow_model_get_response_keyword_input() -> None:
    """WorkflowModel.get_response は input をキーワードでも受け取れる。"""
    from oai_agentspec._adapters import WorkflowModel
    from oai_agentspec.workflow import NodeResults, WorkflowResult

    async def interpret(input: Any, *, context: Any = None) -> WorkflowResult:
        return WorkflowResult(final_output=f"got:{input}", results=NodeResults())

    model = WorkflowModel(interpret)
    resp = await model.get_response(input="hello")
    assert resp.output[0].content[0].text == "got:hello"


@pytest.mark.asyncio
async def test_workflow_model_get_response_positional_input() -> None:
    """get_response(system_instructions, input, ...) の第 2 位置引数も input として扱う。"""
    from oai_agentspec._adapters import WorkflowModel
    from oai_agentspec.workflow import NodeResults, WorkflowResult

    async def interpret(input: Any, *, context: Any = None) -> WorkflowResult:
        return WorkflowResult(final_output=f"pos:{input}", results=NodeResults())

    model = WorkflowModel(interpret)
    # 第 1 位置引数は system_instructions、第 2 位置引数が input。
    resp = await model.get_response("sys", "world")
    assert resp.output[0].content[0].text == "pos:world"


@pytest.mark.asyncio
async def test_workflow_model_stream_response_yields_text_and_completed() -> None:
    """WorkflowModel.stream_response は text-delta + completed を流す（run_streamed 対応）。"""
    from openai.types.responses import ResponseCompletedEvent, ResponseTextDeltaEvent

    from oai_agentspec._adapters import WorkflowModel
    from oai_agentspec.workflow import NodeResults, WorkflowResult

    async def interpret(input: Any, *, context: Any = None) -> WorkflowResult:
        return WorkflowResult(final_output="hello world", results=NodeResults())

    model = WorkflowModel(interpret)
    events = [e async for e in model.stream_response("sys", "q", object(), [], None)]

    deltas = [e for e in events if isinstance(e, ResponseTextDeltaEvent)]
    completed = [e for e in events if isinstance(e, ResponseCompletedEvent)]
    assert "".join(e.delta for e in deltas) == "hello world"  # 全 delta 連結で本文
    assert len(completed) == 1  # 終端は ResponseCompletedEvent 1 件
    assert completed[0].response.output  # 最終出力を載せる


@pytest.mark.asyncio
async def test_deterministic_model_stream_response_yields_tool_call_completed() -> None:
    """DeterministicToolCallModel.stream_response は ToolCall を載せた completed を 1 件流す。"""
    from openai.types.responses import (
        ResponseCompletedEvent,
        ResponseFunctionToolCall,
        ResponseTextDeltaEvent,
    )

    from oai_agentspec._adapters import DeterministicToolCallModel

    model = DeterministicToolCallModel("wf_tool")
    events = [e async for e in model.stream_response("sys", "q", object(), [], None)]

    completed = [e for e in events if isinstance(e, ResponseCompletedEvent)]
    assert not [e for e in events if isinstance(e, ResponseTextDeltaEvent)]  # text delta なし
    assert len(completed) == 1
    out = completed[0].response.output
    assert len(out) == 1 and isinstance(out[0], ResponseFunctionToolCall)
    assert out[0].name == "wf_tool"


def test_adapter_text_response_and_settings_helpers() -> None:
    """text_response / make_required_tool_choice_settings / make_facade_extra の形を確認。"""
    from oai_agentspec._adapters import (
        make_facade_extra,
        make_required_tool_choice_settings,
        text_response,
    )

    resp = text_response("hi")
    assert resp.output[0].content[0].text == "hi"
    assert make_required_tool_choice_settings().tool_choice == "required"
    assert make_facade_extra() == {"tool_use_behavior": "stop_on_first_tool"}


@pytest.mark.asyncio
async def test_workflow_as_tool_handles_plain_string_input() -> None:
    """workflow_as_tool は JSON でない素の文字列入力も input として扱う。"""
    from oai_agentspec._adapters import workflow_as_tool
    from oai_agentspec.workflow import NodeResults, WorkflowResult

    captured: dict[str, Any] = {}

    async def interpret(input: Any, *, context: Any = None) -> WorkflowResult:
        captured["input"] = input
        return WorkflowResult(final_output=input, results=NodeResults())

    tool = workflow_as_tool(interpret, tool_name="t")

    from agents import RunContextWrapper
    from agents.tool_context import ToolContext

    wrapper: RunContextWrapper[None] = RunContextWrapper(context=None)
    tc = ToolContext(
        context=wrapper.context,
        usage=wrapper.usage,
        tool_name="t",
        tool_call_id="c1",
        tool_arguments="not-json",
    )
    out = await tool.on_invoke_tool(tc, "not-json")
    assert captured["input"] == "not-json"
    assert out == "not-json"


@pytest.mark.asyncio
async def test_workflow_model_get_response_binds_input_positionally() -> None:
    """WorkflowModel.get_response は SDK と同じ位置引数（第2=input）で input を束縛する。

    SDK は get_response(system_instructions, input, model_settings, ...) と位置渡しするため、
    第1引数を system_instructions、第2引数を input として束縛できることを確認する
    （index ハックではなく実シグネチャ追従・回帰防止）。
    """
    from oai_agentspec._adapters import WorkflowModel
    from oai_agentspec.workflow import NodeResults, WorkflowResult

    seen: dict[str, Any] = {}

    async def interpret(input: Any, *, context: Any = None) -> WorkflowResult:
        seen["input"] = input
        return WorkflowResult(final_output=input, results=NodeResults())

    model = WorkflowModel(interpret)
    # SDK 流の位置呼び出し（system_instructions, input, 以降は余剰引数）。
    resp = await model.get_response("you are helpful", "user question", object(), [], None)
    assert seen["input"] == "user question"
    assert resp.output  # 単一メッセージ ModelResponse


def test_workflow_model_covers_sdk_abstract_methods() -> None:
    """SDK Model の抽象メソッドが get_response / stream_response だけであることを担保する。

    将来 SDK が抽象メソッドを追加すると WorkflowModel がインスタンス化不能になり全起動が
    壊れるため、増分を早期検知するトリップワイヤ（NFR-7・バージョン耐性）。
    """
    from agents import Model

    from oai_agentspec._adapters import WorkflowModel

    assert Model.__abstractmethods__ <= {"get_response", "stream_response"}
    # 実際にインスタンス化できる（抽象メソッド未実装で TypeError にならない）。
    assert isinstance(WorkflowModel(lambda *a, **k: None), Model)


def test_handcrafted_response_types_cover_required_sdk_fields() -> None:
    """手組みレスポンス型の必須フィールドを _adapters が全て埋めていることを担保する。

    将来 SDK が Response / ResponseFunctionToolCall に必須フィールドを増やすと、手組みが
    NULL のまま静かに壊れるため、増分を早期検知するトリップワイヤ（NFR-7・バージョン耐性）。
    """
    from openai.types.responses import Response, ResponseFunctionToolCall

    # これらは _adapters の _completed_event が手で埋めているフィールド。
    # SDK が必須を増やしたら fail。
    handled_response_fields = {
        "id",
        "created_at",
        "model",
        "object",
        "output",
        "parallel_tool_calls",
        "tool_choice",
        "tools",
    }
    # これらは _adapters の tool_call_response が手で埋めているフィールド。
    # SDK が必須を増やしたら fail。
    handled_tool_call_fields = {"type", "call_id", "name", "arguments"}

    response_required = {n for n, f in Response.model_fields.items() if f.is_required()}
    tool_call_required = {
        n for n, f in ResponseFunctionToolCall.model_fields.items() if f.is_required()
    }

    assert response_required <= handled_response_fields
    assert tool_call_required <= handled_tool_call_fields


def test_latest_user_text_depends_on_itemhelpers_normalization() -> None:
    """latest_user_text が SDK の入力正規化シンボルに依存することを担保する。

    将来 SDK が ItemHelpers.input_to_new_input_list を消す/part 種別や正規化構造を変えると
    素テキスト抽出が空に退行するため、依存崩れを早期検知するトリップワイヤ
    （NFR-7・バージョン耐性）。
    """
    from agents import ItemHelpers

    from oai_agentspec._adapters import latest_user_text

    # SDK 正規化シンボルの存在（メソッド消失も検知）。
    assert hasattr(ItemHelpers, "input_to_new_input_list")

    # str 入力は素テキストがそのまま返る。
    assert latest_user_text("hello") == "hello"

    # input_text part を持つ message 入力から非空の素テキストが抽出できる。
    extracted_input_text = latest_user_text(
        [{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}]
    )
    assert isinstance(extracted_input_text, str)
    assert "hi" in extracted_input_text

    # text part を持つ message 入力からも非空の素テキストが抽出できる。
    # （input_text / text 双方の part 種別への依存を両方突く。）
    extracted_text = latest_user_text(
        [{"role": "user", "content": [{"type": "text", "text": "yo"}]}]
    )
    assert isinstance(extracted_text, str)
    assert "yo" in extracted_text


def test_context_propagation_relies_on_runcontextwrapper_context() -> None:
    """経路A/D の context 透過が RunContextWrapper.context 公開に依存することを担保する。

    将来 SDK が ToolContext と RunContextWrapper の継承を切る/context フィールド公開を変えると
    内部ノードへの context 透過が壊れるため、公開構造の変化を早期検知するトリップワイヤ
    （NFR-7・バージョン耐性）。
    """
    import dataclasses

    from agents import RunContextWrapper
    from agents.tool_context import ToolContext

    assert issubclass(ToolContext, RunContextWrapper)
    assert "context" in {f.name for f in dataclasses.fields(RunContextWrapper)}


def test_required_tool_choice_is_known_toolchoice_literal() -> None:
    """tool_choice='required' が ToolChoice の Literal 明示列挙に含まれることを担保する。

    検知範囲の限界: ToolChoice は `| str` 分岐を持ち ModelSettings は無検証 dataclass のため
    『"required" が SDK 有効値か』は突けない。本テストが検知できるのは SDK が ToolChoice の
    Literal 明示列挙から "required" を外したことに限定される（AC2/AC3/AC5 より検知力は弱い）。
    将来 SDK が当該 Literal から外すと tool_choice='required' 経路の前提が崩れるため、その脱落を
    早期検知するトリップワイヤ（NFR-7・バージョン耐性）。
    """
    from typing import Literal, get_args, get_origin

    from agents.model_settings import ModelSettings, ToolChoice

    from oai_agentspec.constants import WORKFLOW_TOOL_CHOICE_REQUIRED

    # ToolChoice Union から Literal 部分を取り出し再展開して "required" を確認する。
    literal_members: set[str] | None = None
    for arg in get_args(ToolChoice):
        if get_origin(arg) is Literal:
            literal_members = set(get_args(arg))
            break
    assert literal_members is not None, "ToolChoice の Literal 明示列挙が見つからない"
    assert "required" in literal_members

    # sanity 確認: ModelSettings は無検証 dataclass のため常に True で、単独では
    # トリップワイヤにならない（構築可能性の確認に留める）。
    assert (
        ModelSettings(tool_choice=WORKFLOW_TOOL_CHOICE_REQUIRED).tool_choice
        == WORKFLOW_TOOL_CHOICE_REQUIRED
    )


# ----------------------------------------------------------------------
# 経路D / FacadeMode: 入口モデル可変（deterministic / llm_input / llm_input_output）
# ----------------------------------------------------------------------
@dataclass
class _AppCtx:
    """経路D の context 透過検証用のアプリ context。"""

    user_id: str


def _ctx_echo_workflow() -> WorkflowGraph:
    """先頭 FUNCTION ノードで ctx.context.user_id を読み込むワークフロー。"""
    wf = WorkflowGraph("ctx_flow")
    wf.add_function_node(
        "step",
        fn=lambda msg, ctx: f"user={getattr(getattr(ctx, 'context', None), 'user_id', None)}|{msg}",
    )
    wf.add_edge(START, "step")
    wf.add_edge("step", END)
    return wf


@pytest.mark.asyncio
async def test_facade_deterministic_propagates_context_without_llm() -> None:
    """mode=DETERMINISTIC: 実 LLM 0 回で context が内部ノードへ透過し決定論で一致する。"""
    from oai_agentspec import FacadeMode
    from oai_agentspec._adapters import DeterministicToolCallModel

    wf = _ctx_echo_workflow()
    spec = wf.as_facade_spec("facade", mode=FacadeMode.DETERMINISTIC)
    # 入口は決定論モデル（実 LLM ではない）。
    assert isinstance(spec.model, DeterministicToolCallModel)

    agent = build_agent(spec)
    r1 = await Runner.run(agent, input="hello", context=_AppCtx(user_id="vip_123"))
    r2 = await Runner.run(agent, input="hello", context=_AppCtx(user_id="vip_123"))

    # context が内部 FUNCTION ノードへ透過している。
    assert "user=vip_123" in str(r1.final_output)
    # 実 LLM を介さないため 2 回実行が完全一致（決定論）。
    assert r1.final_output == r2.final_output


@pytest.mark.asyncio
async def test_facade_llm_input_calls_model_once_and_passes_through() -> None:
    """mode=LLM_INPUT: 入口 LLM 1 回・出口要約なし（tool 結果を素通し）。"""
    from oai_agentspec import FacadeMode

    wf = WorkflowGraph("passthrough")
    wf.add_function_node("step", fn=lambda msg, ctx: f"WF[{msg}]")
    wf.add_edge(START, "step")
    wf.add_edge("step", END)

    entry = FakeModel().queue_tool_call("wf_tool", '{"input": "payload"}')
    spec = wf.as_facade_spec("facade", mode=FacadeMode.LLM_INPUT, model=entry, tool_name="wf_tool")
    agent = build_agent(spec)

    result = await Runner.run(agent, input="hi")

    # 入口 LLM は 1 回だけ（stop_on_first_tool で出口の要約呼び出しが起きない）。
    assert len(entry.calls) == 1
    # tool 結果がそのまま最終出力（LLM が要約し直さない）。
    assert result.final_output == "WF[payload]"


@pytest.mark.asyncio
async def test_facade_llm_input_output_calls_model_twice_and_summarizes() -> None:
    """mode=LLM_INPUT_OUTPUT: 入口 + 出口で LLM 2 回（tool 結果を LLM が要約）。"""
    from oai_agentspec import FacadeMode

    wf = WorkflowGraph("summarize")
    wf.add_function_node("step", fn=lambda msg, ctx: f"WF[{msg}]")
    wf.add_edge(START, "step")
    wf.add_edge("step", END)

    # 1 ターン目: tool 呼び出し / 2 ターン目: 要約テキスト。
    entry = FakeModel().queue_tool_call("wf_tool", '{"input": "payload"}').queue_text("要約済み")
    spec = wf.as_facade_spec(
        "facade", mode=FacadeMode.LLM_INPUT_OUTPUT, model=entry, tool_name="wf_tool"
    )
    agent = build_agent(spec)

    result = await Runner.run(agent, input="hi")

    # 入口 + 出口要約で LLM 2 回。
    assert len(entry.calls) == 2
    # 最終出力は 2 ターン目（LLM の要約テキスト）。
    assert result.final_output == "要約済み"


def test_facade_mode_controls_stop_on_first_tool() -> None:
    """mode により stop_on_first_tool の有無が決まる（出口の LLM 要約可否を直接担保）。

    DETERMINISTIC / LLM_INPUT は extra に stop_on_first_tool を積み（出口要約なし）、
    LLM_INPUT_OUTPUT は extra を空にする（2 ターン目で LLM が tool 結果を要約する）。
    """
    from oai_agentspec import FacadeMode

    wf = WorkflowGraph("modes")
    wf.add_function_node("step", fn=lambda msg, ctx: msg)
    wf.add_edge(START, "step")
    wf.add_edge("step", END)

    stop = wf.as_facade_spec("f1", mode=FacadeMode.DETERMINISTIC).extra
    one = wf.as_facade_spec("f2", mode=FacadeMode.LLM_INPUT).extra
    two = wf.as_facade_spec("f3", mode=FacadeMode.LLM_INPUT_OUTPUT).extra

    assert stop.get("tool_use_behavior") == "stop_on_first_tool"
    assert one.get("tool_use_behavior") == "stop_on_first_tool"
    # LLM_INPUT_OUTPUT は stop を積まない（出口 LLM 要約を許可）。
    assert "tool_use_behavior" not in two


def test_facade_default_mode_is_llm_input_backward_compatible() -> None:
    """mode 既定（LLM_INPUT）は従来の経路A と同一構成（model なし・stop_on_first_tool）。"""
    wf = WorkflowGraph("compat")
    wf.add_function_node("step", fn=lambda msg, ctx: msg)
    wf.add_edge(START, "step")
    wf.add_edge("step", END)

    spec = wf.as_facade_spec("facade")
    # 既定では入口モデルを注入しない（SDK 既定モデル任せ）。
    assert spec.model is None
    # stop_on_first_tool が積まれている（出口の LLM 要約なし）。
    assert spec.extra.get("tool_use_behavior") == "stop_on_first_tool"


def test_deterministic_model_is_stateless() -> None:
    """DeterministicToolCallModel は実行状態を持たず、複数呼び出しで同型 ToolCall を返す。"""
    from oai_agentspec._adapters import DeterministicToolCallModel

    model = DeterministicToolCallModel("wf_tool")
    before = dict(model.__dict__)

    async def _twice() -> tuple[Any, Any]:
        r1 = await model.get_response("sys", "q1", object(), [], None)
        r2 = await model.get_response("sys", "q2", object(), [], None)
        return r1, r2

    r1, r2 = asyncio.run(_twice())

    # get_response はインスタンス状態を変化させない（ステートレス）。
    assert model.__dict__ == before
    # 毎回 1 つの ToolCall を返し、tool 名は固定・引数は {"input": 入力} に正しく載る。
    assert r1.output[0].name == "wf_tool"
    assert r2.output[0].name == "wf_tool"
    assert json.loads(r1.output[0].arguments) == {"input": "q1"}
    assert json.loads(r2.output[0].arguments) == {"input": "q2"}


def test_sdk_reset_tool_choice_default_is_true() -> None:
    """LLM_INPUT_OUTPUT の無限ループ回避は SDK の reset_tool_choice 既定 True に依存する。

    将来 SDK が既定を変えると LLM_INPUT_OUTPUT が静かに max_turns ループへ退行するため、
    既定値の変化を早期検知するトリップワイヤ（NFR-7・バージョン耐性）。
    """
    from agents import Agent

    assert Agent(name="probe").reset_tool_choice is True


def _wf_wrap() -> WorkflowGraph:
    """tool 入力を WF[...] で包む最小ワークフロー（LLM_INPUT_OUTPUT 検証用）。"""
    wf = WorkflowGraph("wrap")
    wf.add_function_node("step", fn=lambda msg, ctx: f"WF[{msg}]")
    wf.add_edge(START, "step")
    wf.add_edge("step", END)
    return wf


@pytest.mark.asyncio
async def test_llm_input_output_terminates_via_reset_tool_choice() -> None:
    """mode=LLM_INPUT_OUTPUT: reset_tool_choice 既定 True で 2 ターン目に解除→終了する。

    tool_choice を尊重する ChoiceAwareModel を使い、turn1 で required→ToolCall、turn2 で
    reset により tool_choice=None→text となって 2 回で停止することを直接検証する。
    """
    from oai_agentspec import FacadeMode

    model = ChoiceAwareModel(tool_name="wf_tool", text="done")
    spec = _wf_wrap().as_facade_spec(
        "f", mode=FacadeMode.LLM_INPUT_OUTPUT, model=model, tool_name="wf_tool"
    )
    agent = build_agent(spec)

    result = await Runner.run(agent, input="hi")

    assert model.calls == 2  # turn1=ToolCall, turn2=text（reset で required 解除）
    assert result.final_output == "done"


@pytest.mark.asyncio
async def test_llm_input_output_loops_without_reset_tool_choice() -> None:
    """reset_tool_choice=False だと LLM_INPUT_OUTPUT は tool を呼び続け max_turns 例外になる。

    ループ回避が reset_tool_choice に依存することの裏返し検証（reset を切ると守りが外れる）。
    """
    from oai_agentspec import FacadeMode

    model = ChoiceAwareModel(tool_name="wf_tool", text="done")
    spec = _wf_wrap().as_facade_spec(
        "f", mode=FacadeMode.LLM_INPUT_OUTPUT, model=model, tool_name="wf_tool"
    )
    # reset_tool_choice=False を Agent へ注入（required が毎ターン解除されない）。
    spec_no_reset = replace(spec, extra={**spec.extra, "reset_tool_choice": False})
    agent = build_agent(spec_no_reset)

    with pytest.raises(MaxTurnsExceeded):
        await Runner.run(agent, input="hi", max_turns=4)


def test_facade_deterministic_rejects_explicit_model() -> None:
    """mode=DETERMINISTIC で model を指定すると fail-fast（無音で破棄しない）。"""
    from oai_agentspec import FacadeMode

    wf = _wf_wrap()
    with pytest.raises(ValueError, match="mode=DETERMINISTIC では model を指定できません"):
        wf.as_facade_spec("f", mode=FacadeMode.DETERMINISTIC, model=FakeModel())


def test_facade_mode_accepts_raw_string_value() -> None:
    """mode に生文字列 "deterministic" を渡しても enum と同一に扱う（str enum の is 対策）。"""
    from oai_agentspec._adapters import DeterministicToolCallModel

    wf = _wf_wrap()
    spec = wf.as_facade_spec("f", mode="deterministic")
    # 文字列指定でも決定論モデルが注入され、stop_on_first_tool が付く（経路D として成立）。
    assert isinstance(spec.model, DeterministicToolCallModel)
    assert spec.extra.get("tool_use_behavior") == "stop_on_first_tool"


def test_facade_mode_rejects_unknown_string() -> None:
    """未知の mode 文字列は ValueError で fail-fast（黙って LLM 既定に落ちない）。"""
    wf = _wf_wrap()
    with pytest.raises(ValueError):
        wf.as_facade_spec("f", mode="bogus")


def test_facade_deterministic_model_tool_name_matches_tool() -> None:
    """DETERMINISTIC: 決定論モデルが呼ぶ tool 名がファサード tool 名と一致する（省略/明示）。"""
    from oai_agentspec import FacadeMode

    wf = _wf_wrap()
    derived = wf.as_facade_spec("flow", mode=FacadeMode.DETERMINISTIC)
    assert derived.model._tool_name == derived.tools[0].name == "flow_workflow"

    explicit = wf.as_facade_spec("flow2", mode=FacadeMode.DETERMINISTIC, tool_name="run_it")
    assert explicit.model._tool_name == explicit.tools[0].name == "run_it"


def test_deterministic_model_coerces_non_string_input() -> None:
    """DETERMINISTIC 入口: latest_user_text が非 str を返しても tool 引数 input は str になる。"""
    from oai_agentspec._adapters import DeterministicToolCallModel

    model = DeterministicToolCallModel("wf_tool")
    # user メッセージの無い input-list は latest_user_text が抽出に失敗し非 str を返すため、
    # tool スキーマ（input:string）整合のため str 化される。
    no_user = [{"role": "assistant", "content": "prior reply"}]
    resp = asyncio.run(model.get_response("sys", no_user, object(), [], None))
    parsed = json.loads(resp.output[0].arguments)
    assert isinstance(parsed["input"], str)
