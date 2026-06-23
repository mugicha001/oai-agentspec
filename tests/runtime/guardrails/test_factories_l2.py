"""L2: helper ファクトリ（`factories`）の SDK 結合検証（実 Agent + FakeModel + Runner）。

agent 境界 factory が SDK 互換 `InputGuardrail` / `OutputGuardrail` を返し、`AgentSpec` の専用
フィールド `input_guardrails` / `output_guardrails` へ載せて実 Agent を Runner で走らせると trip
時に `InputGuardrailTripwireTriggered` / `OutputGuardrailTripwireTriggered` が上がり、非 trip 時は
通常完了することを検証する。`prompt_llm_guardrail` は判定 model を FakeModel でモックし trip /
pass 両経路と `verdict` DI 差し替えを、`guard_tool` は tool guardrail の装着・メタ維持・`on_trip`
分岐を検証する。FakeModel で出力を制御し実 LLM を呼ばない（決定的）。
"""

from __future__ import annotations

import pytest
from agents import (
    Agent,
    FunctionTool,
    InputGuardrail,
    InputGuardrailTripwireTriggered,
    OutputGuardrail,
    OutputGuardrailTripwireTriggered,
    Runner,
    ToolGuardrailFunctionOutput,
    ToolInputGuardrail,
    ToolInputGuardrailData,
    ToolOutputGuardrail,
    ToolOutputGuardrailData,
)
from agents.tool_context import ToolContext

from oai_agentspec import AgentSpec, function_tool
from oai_agentspec._adapters import build_agent
from oai_agentspec.runtime.guardrails._detectors import Detection
from oai_agentspec.runtime.guardrails.factories import (
    allow_deny_guardrail,
    canary_guardrail,
    external_detector_guardrail,
    guard_tool,
    injection_baseline_guardrail,
    length_guardrail,
    predicate_guardrail,
    prompt_llm_guardrail,
    regex_guardrail,
    tool_guardrail,
)

from _helpers.fake_model import FakeModel

pytestmark = pytest.mark.integration


# ----------------------------------------------------------------------
# フィールド転送回帰: AgentSpec 専用フィールドが build_agent された Agent に渡る
# ----------------------------------------------------------------------


def test_guardrails_field_forwarded_to_built_agent() -> None:
    """helper の戻り値を AgentSpec の input/output_guardrails 専用フィールドへ渡して build_agent
    すると、生成 Agent の input/output_guardrails に転送される（agents.Agent と同型の宣言面）。"""
    g_in = regex_guardrail(r"\d", on="input")
    g_out = canary_guardrail("LEAK")
    spec = AgentSpec(
        name="bot",
        instructions="i",
        model=FakeModel(),
        input_guardrails=[g_in],
        output_guardrails=[g_out],
    )
    agent = build_agent(spec)
    assert agent.input_guardrails == [g_in]
    assert agent.output_guardrails == [g_out]


def test_guardrail_fields_default_empty_and_independent() -> None:
    """input/output_guardrails は既定で空リストかつインスタンス間で独立（default_factory）。"""
    s1 = AgentSpec(name="a", instructions="i")
    s2 = AgentSpec(name="b", instructions="i")
    assert s1.input_guardrails == [] and s1.output_guardrails == []
    s1.input_guardrails.append(regex_guardrail("x", on="input"))
    assert s2.input_guardrails == []  # 共有されない


def test_guardrail_in_extra_collides_with_dedicated_field() -> None:
    """extra に input/output_guardrails を入れると専用フィールドと衝突して ValueError。"""
    g = regex_guardrail("x", on="input")
    with pytest.raises(ValueError, match="input_guardrails"):
        build_agent(AgentSpec(name="bot", instructions="i", extra={"input_guardrails": [g]}))


# ----------------------------------------------------------------------
# helper: 専用フィールドで Agent を build し Runner で走らせる
# ----------------------------------------------------------------------


def _input_agent(guardrail: object, *, output_text: str = "ok") -> Agent:
    """input_guardrails 専用フィールドへ載せた実 Agent を build する。"""
    spec = AgentSpec(
        name="bot",
        instructions="i",
        model=FakeModel().queue_text(output_text),
        input_guardrails=[guardrail],
    )
    return build_agent(spec)


def _output_agent(guardrail: object, *, output_text: str) -> Agent:
    """output_guardrails 専用フィールドへ載せた実 Agent を build する。"""
    spec = AgentSpec(
        name="bot",
        instructions="i",
        model=FakeModel().queue_text(output_text),
        output_guardrails=[guardrail],
    )
    return build_agent(spec)


# ----------------------------------------------------------------------
# canary_guardrail（OutputGuardrail）
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_canary_guardrail_trips_on_leak() -> None:
    """出力に canary が含まれると OutputGuardrailTripwireTriggered。"""
    agent = _output_agent(canary_guardrail("LEAK-TOKEN"), output_text="oops LEAK-TOKEN here")
    with pytest.raises(OutputGuardrailTripwireTriggered):
        await Runner.run(agent, "hi")


@pytest.mark.asyncio
async def test_canary_guardrail_passes_when_clean() -> None:
    """canary を含まない出力は通常完了する。"""
    agent = _output_agent(canary_guardrail("LEAK-TOKEN"), output_text="clean output")
    result = await Runner.run(agent, "hi")
    assert result.final_output == "clean output"


# ----------------------------------------------------------------------
# predicate_guardrail（input / output）
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_predicate_guardrail_input_trips() -> None:
    """input predicate が True で InputGuardrailTripwireTriggered。"""
    agent = _input_agent(predicate_guardrail(lambda t: "block" in t, on="input"))
    with pytest.raises(InputGuardrailTripwireTriggered):
        await Runner.run(agent, "please block this")


@pytest.mark.asyncio
async def test_predicate_guardrail_input_passes() -> None:
    """input predicate が False なら通常完了。"""
    agent = _input_agent(predicate_guardrail(lambda t: "block" in t, on="input"))
    result = await Runner.run(agent, "all good")
    assert result.final_output == "ok"


@pytest.mark.asyncio
async def test_predicate_guardrail_async_predicate_trips() -> None:
    """async 述語（async def）も await されて trip する（同期/非同期両対応の回帰）。"""

    async def predicate(text: str) -> bool:
        return "block" in text

    agent = _input_agent(predicate_guardrail(predicate, on="input"))
    with pytest.raises(InputGuardrailTripwireTriggered):
        await Runner.run(agent, "please block this")


@pytest.mark.asyncio
async def test_predicate_guardrail_async_predicate_passes() -> None:
    """async 述語が False を返すと通常完了する（await 経路の pass 側）。"""

    async def predicate(text: str) -> bool:
        return "block" in text

    agent = _input_agent(predicate_guardrail(predicate, on="input"))
    result = await Runner.run(agent, "all good")
    assert result.final_output == "ok"


@pytest.mark.asyncio
async def test_predicate_guardrail_async_callable_object_trips() -> None:
    """`async __call__` を持つ述語オブジェクト（DI でよくある形）でも await されて trip する。

    `inspect.iscoroutinefunction` は async __call__ オブジェクトに False を返すため型判定では
    取りこぼすが、戻り値 awaitable 判定なら await される回帰。
    """

    class _AsyncPredicate:
        async def __call__(self, text: str) -> bool:
            return "block" in text

    agent = _input_agent(predicate_guardrail(_AsyncPredicate(), on="input"))
    with pytest.raises(InputGuardrailTripwireTriggered):
        await Runner.run(agent, "please block this")


# ----------------------------------------------------------------------
# regex_guardrail
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_regex_guardrail_input_trips() -> None:
    """input でパターン一致すると trip。"""
    agent = _input_agent(regex_guardrail(r"\d{3}-\d{4}", on="input"))
    with pytest.raises(InputGuardrailTripwireTriggered):
        await Runner.run(agent, "call 123-4567")


@pytest.mark.asyncio
async def test_regex_guardrail_input_passes() -> None:
    """input でパターン不一致なら通常完了。"""
    agent = _input_agent(regex_guardrail(r"\d{3}-\d{4}", on="input"))
    result = await Runner.run(agent, "no numbers")
    assert result.final_output == "ok"


# ----------------------------------------------------------------------
# length_guardrail（on 必須）
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_length_guardrail_output_trips_on_overflow() -> None:
    """出力が max_length 超過で trip（on='output'）。"""
    agent = _output_agent(length_guardrail(max_length=3, on="output"), output_text="toolong")
    with pytest.raises(OutputGuardrailTripwireTriggered):
        await Runner.run(agent, "hi")


@pytest.mark.asyncio
async def test_length_guardrail_output_passes_within_bound() -> None:
    """出力が範囲内なら通常完了。"""
    agent = _output_agent(length_guardrail(max_length=10, on="output"), output_text="short")
    result = await Runner.run(agent, "hi")
    assert result.final_output == "short"


def test_length_guardrail_requires_a_threshold() -> None:
    """max_length / min_length 両方 None なら ValueError（無言の no-op を排す）。"""
    with pytest.raises(ValueError, match="max_length / min_length"):
        length_guardrail(on="output")


# ----------------------------------------------------------------------
# allow_deny_guardrail（on 必須）
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_allow_deny_guardrail_output_deny_trips() -> None:
    """出力に deny 語が含まれると trip。"""
    agent = _output_agent(
        allow_deny_guardrail(deny=["badword"], on="output"), output_text="has badword in it"
    )
    with pytest.raises(OutputGuardrailTripwireTriggered):
        await Runner.run(agent, "hi")


@pytest.mark.asyncio
async def test_allow_deny_guardrail_output_passes() -> None:
    """deny 語を含まない出力は通常完了。"""
    agent = _output_agent(allow_deny_guardrail(deny=["badword"], on="output"), output_text="clean")
    result = await Runner.run(agent, "hi")
    assert result.final_output == "clean"


# ----------------------------------------------------------------------
# _agent_guardrail: 不正な on 値の検証
# ----------------------------------------------------------------------


def test_agent_guardrail_invalid_on_raises_value_error() -> None:
    """on が 'input' / 'output' 以外なら ValueError（_agent_guardrail のガード）。"""
    with pytest.raises(ValueError, match="on must be 'input' or 'output'"):
        regex_guardrail("x", on="sideways")


# ----------------------------------------------------------------------
# injection_baseline_guardrail（InputGuardrail）
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_injection_baseline_guardrail_input_trips() -> None:
    """注入ベースライン入力で trip（InputGuardrail）。"""
    agent = _input_agent(injection_baseline_guardrail())
    with pytest.raises(InputGuardrailTripwireTriggered):
        await Runner.run(agent, "admin' OR 1=1 --")


@pytest.mark.asyncio
async def test_injection_baseline_guardrail_input_passes() -> None:
    """良性入力は通常完了。"""
    agent = _input_agent(injection_baseline_guardrail())
    result = await Runner.run(agent, "what is the weather")
    assert result.final_output == "ok"


@pytest.mark.asyncio
async def test_injection_baseline_guardrail_extra_patterns() -> None:
    """extra_patterns DI で追加検知が効く。"""
    agent = _input_agent(injection_baseline_guardrail(extra_patterns=[r"eval\("]))
    with pytest.raises(InputGuardrailTripwireTriggered):
        await Runner.run(agent, "run eval( something )")


# ----------------------------------------------------------------------
# external_detector_guardrail（A 家族・利用者検知 DI）
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_external_detector_guardrail_input_trips() -> None:
    """利用者検知 callable が triggered を返すと trip（on='input'）。"""

    def detect(text: str) -> Detection:
        return Detection(triggered="pii" in text, reason="external pii")

    agent = _input_agent(external_detector_guardrail(detect, on="input"))
    with pytest.raises(InputGuardrailTripwireTriggered):
        await Runner.run(agent, "contains pii data")


@pytest.mark.asyncio
async def test_external_detector_guardrail_output_passes() -> None:
    """利用者検知が not triggered なら通常完了（on='output'）。"""

    def detect(text: str) -> Detection:
        return Detection(triggered=False)

    agent = _output_agent(external_detector_guardrail(detect, on="output"), output_text="fine")
    result = await Runner.run(agent, "hi")
    assert result.final_output == "fine"


@pytest.mark.asyncio
async def test_external_detector_guardrail_async_detector_input_trips() -> None:
    """async 検知器（async def）を渡しても coroutine を await して trip する（回帰: 未 await で

    `AttributeError: 'coroutine' object has no attribute 'reason'` にならないこと）。
    """

    async def detect(text: str) -> Detection:
        return Detection(triggered="pii" in text, reason="async pii")

    agent = _input_agent(external_detector_guardrail(detect, on="input"))
    with pytest.raises(InputGuardrailTripwireTriggered):
        await Runner.run(agent, "contains pii data")


@pytest.mark.asyncio
async def test_external_detector_guardrail_async_detector_output_passes() -> None:
    """async output 検知器が not triggered なら通常完了する（await 経路の pass 側）。"""

    async def detect(text: str) -> Detection:
        return Detection(triggered=False)

    agent = _output_agent(external_detector_guardrail(detect, on="output"), output_text="fine")
    result = await Runner.run(agent, "hi")
    assert result.final_output == "fine"


@pytest.mark.asyncio
async def test_external_detector_guardrail_async_callable_object_trips() -> None:
    """`async __call__` を持つ callable オブジェクト（DI でよくある形）でも await されて trip する。

    `inspect.iscoroutinefunction` は async __call__ オブジェクトに False を返すため型判定では
    取りこぼすが、戻り値 awaitable 判定なら await され `AttributeError` を起こさない回帰。
    """

    class _AsyncDetector:
        async def __call__(self, text: str) -> Detection:
            return Detection(triggered="pii" in text, reason="async obj pii")

    agent = _input_agent(external_detector_guardrail(_AsyncDetector(), on="input"))
    with pytest.raises(InputGuardrailTripwireTriggered):
        await Runner.run(agent, "contains pii data")


@pytest.mark.asyncio
async def test_external_detector_guardrail_async_callable_object_passes() -> None:
    """`async __call__` オブジェクトが not triggered なら通常完了する（await 経路の pass 側）。"""

    class _AsyncDetector:
        async def __call__(self, text: str) -> Detection:
            return Detection(triggered=False)

    agent = _output_agent(
        external_detector_guardrail(_AsyncDetector(), on="output"), output_text="fine"
    )
    result = await Runner.run(agent, "hi")
    assert result.final_output == "fine"


# ----------------------------------------------------------------------
# prompt_llm_guardrail（LLM-as-judge・判定 model モック）
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prompt_llm_guardrail_input_trips_on_unsafe_verdict() -> None:
    """judge model が UNSAFE を返すと既定 verdict で trip（on='input'）。"""
    judge_model = FakeModel().queue_text("UNSAFE: policy violation")
    guardrail = prompt_llm_guardrail(judge_model, "judge prompt", on="input")
    agent = _input_agent(guardrail)
    with pytest.raises(InputGuardrailTripwireTriggered):
        await Runner.run(agent, "questionable input")


@pytest.mark.asyncio
async def test_prompt_llm_guardrail_input_passes_on_safe_verdict() -> None:
    """judge model が SAFE を返すと既定 verdict では trip せず通常完了。"""
    judge_model = FakeModel().queue_text("SAFE")
    guardrail = prompt_llm_guardrail(judge_model, "judge prompt", on="input")
    agent = _input_agent(guardrail)
    result = await Runner.run(agent, "benign input")
    assert result.final_output == "ok"


@pytest.mark.asyncio
async def test_prompt_llm_guardrail_output_boundary() -> None:
    """on='output' で OutputGuardrail として trip 経路が機能する。"""
    judge_model = FakeModel().queue_text("verdict: UNSAFE")
    guardrail = prompt_llm_guardrail(judge_model, "judge prompt", on="output")
    agent = _output_agent(guardrail, output_text="model answer")
    with pytest.raises(OutputGuardrailTripwireTriggered):
        await Runner.run(agent, "hi")


@pytest.mark.asyncio
async def test_prompt_llm_guardrail_custom_verdict_changes_behavior() -> None:
    """verdict DI 差し替えで挙動が変わる（既定では pass する出力を trip させる）。"""
    judge_model = FakeModel().queue_text("SAFE")

    def strict_verdict(text: str) -> Detection:
        # 既定と逆に「SAFE が含まれていなければ pass、含まれれば trip」させる反転パーサ。
        return Detection(triggered="SAFE" in text, reason="custom")

    guardrail = prompt_llm_guardrail(
        judge_model, "judge prompt", on="input", verdict=strict_verdict
    )
    agent = _input_agent(guardrail)
    # 既定 verdict なら SAFE で pass するが、custom verdict は SAFE を trip 扱いにする。
    with pytest.raises(InputGuardrailTripwireTriggered):
        await Runner.run(agent, "anything")


def test_prompt_llm_guardrail_invalid_on_raises_value_error() -> None:
    """on が 'input' / 'output' 以外なら ValueError（黙った input フォールスルーを排す）。"""
    with pytest.raises(ValueError, match="on must be 'input' or 'output'"):
        prompt_llm_guardrail(FakeModel(), "judge prompt", on="bogus")


# ----------------------------------------------------------------------
# guard_tool（ツール境界・装着 / メタ維持 / on_trip 分岐）
# ----------------------------------------------------------------------


def _make_tool() -> object:
    """検証用の FunctionTool を作る（name / description / 引数スキーマを持つ）。"""

    @function_tool(name_override="search", needs_approval=True)
    def _tool(query: str) -> str:
        """検索ツール。"""
        return f"result:{query}"

    return _tool


def _trip_detector() -> object:
    """常に triggered を返す検知器。"""

    def detect(text: str) -> Detection:
        return Detection(triggered=True, reason="tool trip")

    return detect


def test_guard_tool_attaches_input_and_output_guardrails() -> None:
    """input / output 検知器を与えると tool_input/output_guardrails が装着される。"""
    tool = _make_tool()
    guarded = guard_tool(tool, input_detector=_trip_detector(), output_detector=_trip_detector())
    assert len(guarded.tool_input_guardrails) == 1
    assert len(guarded.tool_output_guardrails) == 1


def test_guard_tool_preserves_tool_metadata() -> None:
    """name / description / params_json_schema / needs_approval を維持する。"""
    tool = _make_tool()
    guarded = guard_tool(tool, input_detector=_trip_detector())
    assert guarded.name == tool.name
    assert guarded.description == tool.description
    assert guarded.params_json_schema == tool.params_json_schema
    assert guarded.needs_approval == tool.needs_approval


def test_guard_tool_guardrail_names_are_dedicated() -> None:
    """生成 guardrail の get_name() は専用名（tool_input_guardrail / tool_output_guardrail）。"""
    tool = _make_tool()
    guarded = guard_tool(tool, input_detector=_trip_detector(), output_detector=_trip_detector())
    assert guarded.tool_input_guardrails[0].get_name() == "tool_input_guardrail"
    assert guarded.tool_output_guardrails[0].get_name() == "tool_output_guardrail"


def test_guard_tool_none_detectors_returns_original_tool() -> None:
    """input / output 双方 None なら元 tool をそのまま返す。"""
    tool = _make_tool()
    assert guard_tool(tool) is tool


def _tool_input_data(arguments: str = '{"query": "x"}') -> ToolInputGuardrailData:
    """ツール入力 guardrail データを構築する。"""
    ctx = ToolContext(context=None, tool_name="search", tool_call_id="c1", tool_arguments=arguments)
    return ToolInputGuardrailData(context=ctx, agent=None)


def _tool_output_data(output: object) -> ToolOutputGuardrailData:
    """ツール出力 guardrail データを構築する。"""
    ctx = ToolContext(context=None, tool_name="search", tool_call_id="c1", tool_arguments="{}")
    return ToolOutputGuardrailData(context=ctx, agent=None, output=output)


@pytest.mark.asyncio
async def test_guard_tool_on_trip_reject_returns_reject_content() -> None:
    """on_trip='reject'（既定）: trip 時に reject_content になる（guardrail 関数は async）。"""
    tool = _make_tool()
    guarded = guard_tool(tool, input_detector=_trip_detector())
    gr = guarded.tool_input_guardrails[0]
    out: ToolGuardrailFunctionOutput = await gr.guardrail_function(_tool_input_data())
    assert out.behavior["type"] == "reject_content"
    assert out.behavior["message"] == "tool trip"


@pytest.mark.asyncio
async def test_guard_tool_on_trip_raise_returns_raise_exception() -> None:
    """on_trip='raise': trip 時に raise_exception の出力になる。"""
    tool = _make_tool()
    guarded = guard_tool(tool, input_detector=_trip_detector(), on_trip="raise")
    gr = guarded.tool_input_guardrails[0]
    out = await gr.guardrail_function(_tool_input_data())
    assert out.behavior["type"] == "raise_exception"


@pytest.mark.asyncio
async def test_guard_tool_on_trip_allow_returns_allow() -> None:
    """on_trip='allow': trip しても allow の出力になる（通過）。"""
    tool = _make_tool()
    guarded = guard_tool(tool, input_detector=_trip_detector(), on_trip="allow")
    gr = guarded.tool_input_guardrails[0]
    out = await gr.guardrail_function(_tool_input_data())
    assert out.behavior["type"] == "allow"


@pytest.mark.asyncio
async def test_guard_tool_on_trip_callable_is_invoked() -> None:
    """on_trip に callable を渡すと Detection を受けて出力を組める（DI 分岐）。"""
    tool = _make_tool()
    sentinel = ToolGuardrailFunctionOutput.allow(output_info={"via": "callable"})

    def on_trip(detection: Detection) -> ToolGuardrailFunctionOutput:
        assert detection.triggered is True
        return sentinel

    guarded = guard_tool(tool, output_detector=_trip_detector(), on_trip=on_trip)
    gr = guarded.tool_output_guardrails[0]
    out = await gr.guardrail_function(_tool_output_data("anything"))
    assert out is sentinel


@pytest.mark.asyncio
async def test_guard_tool_output_no_trip_allows() -> None:
    """検知器が triggered=False を返すと allow（通過）になる。"""

    def detect(text: str) -> Detection:
        return Detection(triggered=False)

    tool = _make_tool()
    guarded = guard_tool(tool, output_detector=detect)
    gr = guarded.tool_output_guardrails[0]
    out = await gr.guardrail_function(_tool_output_data("safe output"))
    assert out.behavior["type"] == "allow"


@pytest.mark.asyncio
async def test_guard_tool_output_detector_inspects_output_text() -> None:
    """output_detector は data.output のテキストを検査する（trip 条件が出力依存）。"""

    def detect(text: str) -> Detection:
        return Detection(triggered="secret" in text, reason="leak")

    tool = _make_tool()
    guarded = guard_tool(tool, output_detector=detect)
    gr = guarded.tool_output_guardrails[0]
    tripped = await gr.guardrail_function(_tool_output_data("contains secret"))
    assert tripped.behavior["type"] == "reject_content"
    passed = await gr.guardrail_function(_tool_output_data("clean"))
    assert passed.behavior["type"] == "allow"


@pytest.mark.asyncio
async def test_guard_tool_async_output_detector_awaited() -> None:
    """async output_detector を渡すと tool guardrail が coroutine を await して trip/pass する。"""

    async def detect(text: str) -> Detection:
        return Detection(triggered="secret" in text, reason="async leak")

    tool = _make_tool()
    guarded = guard_tool(tool, output_detector=detect)
    gr = guarded.tool_output_guardrails[0]
    tripped = await gr.guardrail_function(_tool_output_data("contains secret"))
    assert tripped.behavior["type"] == "reject_content"
    assert tripped.behavior["message"] == "async leak"
    passed = await gr.guardrail_function(_tool_output_data("clean"))
    assert passed.behavior["type"] == "allow"


@pytest.mark.asyncio
async def test_guard_tool_async_input_detector_awaited() -> None:
    """async input_detector を渡すと tool guardrail が coroutine を await して引数を検査する。"""

    async def detect(text: str) -> Detection:
        return Detection(triggered="danger" in text, reason="async arg")

    tool = _make_tool()
    guarded = guard_tool(tool, input_detector=detect)
    gr = guarded.tool_input_guardrails[0]
    tripped = await gr.guardrail_function(_tool_input_data('{"query": "danger"}'))
    assert tripped.behavior["type"] == "reject_content"
    passed = await gr.guardrail_function(_tool_input_data('{"query": "ok"}'))
    assert passed.behavior["type"] == "allow"


@pytest.mark.asyncio
async def test_guard_tool_invalid_on_trip_string_raises_value_error() -> None:
    """on_trip に不正な文字列を渡すと trip 解決時に ValueError（契約統一・callable は対象外）。"""
    tool = _make_tool()
    guarded = guard_tool(tool, input_detector=_trip_detector(), on_trip="bogus")  # type: ignore[arg-type]
    gr = guarded.tool_input_guardrails[0]
    with pytest.raises(ValueError, match="on_trip must be"):
        await gr.guardrail_function(_tool_input_data())


@pytest.mark.asyncio
async def test_guard_tool_async_callable_object_output_detector() -> None:
    """`async __call__` を持つ callable オブジェクトの output_detector でも await されて機能する。

    `inspect.iscoroutinefunction` は async __call__ オブジェクトに False を返すため型判定では
    取りこぼすが、戻り値 awaitable 判定なら正しく await される回帰（DI でよくある形）。
    """

    class _AsyncDetector:
        async def __call__(self, text: str) -> Detection:
            return Detection(triggered="secret" in text, reason="async obj")

    tool = _make_tool()
    guarded = guard_tool(tool, output_detector=_AsyncDetector())
    gr = guarded.tool_output_guardrails[0]
    tripped = await gr.guardrail_function(_tool_output_data("contains secret"))
    assert tripped.behavior["type"] == "reject_content"
    assert tripped.behavior["message"] == "async obj"
    passed = await gr.guardrail_function(_tool_output_data("clean"))
    assert passed.behavior["type"] == "allow"


@pytest.mark.asyncio
async def test_guard_tool_async_callable_object_input_detector() -> None:
    """`async __call__` を持つ callable オブジェクトの input_detector でも await されて機能する。"""

    class _AsyncDetector:
        async def __call__(self, text: str) -> Detection:
            return Detection(triggered="danger" in text, reason="async obj in")

    tool = _make_tool()
    guarded = guard_tool(tool, input_detector=_AsyncDetector())
    gr = guarded.tool_input_guardrails[0]
    tripped = await gr.guardrail_function(_tool_input_data('{"query": "danger"}'))
    assert tripped.behavior["type"] == "reject_content"
    passed = await gr.guardrail_function(_tool_input_data('{"query": "ok"}'))
    assert passed.behavior["type"] == "allow"


# ----------------------------------------------------------------------
# run_in_parallel の露出（input 系 factory・既定 True / False 伝播）
# ----------------------------------------------------------------------


def test_injection_baseline_guardrail_run_in_parallel_default_true() -> None:
    """既定では生成 InputGuardrail.run_in_parallel が True。"""
    g = injection_baseline_guardrail()
    assert g.run_in_parallel is True


def test_injection_baseline_guardrail_run_in_parallel_false() -> None:
    """run_in_parallel=False を渡すと InputGuardrail.run_in_parallel が False になる。"""
    g = injection_baseline_guardrail(run_in_parallel=False)
    assert g.run_in_parallel is False


def test_external_detector_guardrail_run_in_parallel_default_true() -> None:
    """external（async builder 経路）の input でも既定 run_in_parallel=True。"""
    g = external_detector_guardrail(lambda t: Detection(triggered=False), on="input")
    assert g.run_in_parallel is True


def test_external_detector_guardrail_run_in_parallel_false() -> None:
    """external の input で run_in_parallel=False が async builder 経由で伝播する。"""
    g = external_detector_guardrail(
        lambda t: Detection(triggered=False), on="input", run_in_parallel=False
    )
    assert g.run_in_parallel is False


def test_prompt_llm_guardrail_run_in_parallel_default_true() -> None:
    """prompt_llm_guardrail（on="input"・async builder 経路）も既定 run_in_parallel=True。"""
    g = prompt_llm_guardrail(FakeModel(), "judge prompt", on="input")
    assert g.run_in_parallel is True


def test_prompt_llm_guardrail_run_in_parallel_false() -> None:
    """prompt_llm_guardrail の input で run_in_parallel=False が伝播する。"""
    g = prompt_llm_guardrail(FakeModel(), "judge prompt", on="input", run_in_parallel=False)
    assert g.run_in_parallel is False


def test_predicate_guardrail_run_in_parallel_false() -> None:
    """predicate_guardrail の input で run_in_parallel=False が伝播する（既定は True）。"""
    assert predicate_guardrail(lambda t: False, on="input").run_in_parallel is True
    g = predicate_guardrail(lambda t: False, on="input", run_in_parallel=False)
    assert g.run_in_parallel is False


def test_regex_guardrail_run_in_parallel_false() -> None:
    """regex_guardrail（同期 builder 経路）の input で run_in_parallel=False が伝播する。"""
    assert regex_guardrail(r"\d", on="input").run_in_parallel is True
    g = regex_guardrail(r"\d", on="input", run_in_parallel=False)
    assert g.run_in_parallel is False


def test_length_guardrail_input_run_in_parallel_false() -> None:
    """length_guardrail を on="input" にして run_in_parallel=False が伝播する。"""
    assert length_guardrail(max_length=5, on="input").run_in_parallel is True
    g = length_guardrail(max_length=5, on="input", run_in_parallel=False)
    assert g.run_in_parallel is False


def test_allow_deny_guardrail_input_run_in_parallel_false() -> None:
    """allow_deny_guardrail を on="input" にして run_in_parallel=False が伝播する。"""
    assert allow_deny_guardrail(deny=["x"], on="input").run_in_parallel is True
    g = allow_deny_guardrail(deny=["x"], on="input", run_in_parallel=False)
    assert g.run_in_parallel is False


def test_run_in_parallel_ignored_on_output_boundary() -> None:
    """on="output" では run_in_parallel は無視される（OutputGuardrail に該当フィールドなし）。

    output 境界へ run_in_parallel=False を渡しても OutputGuardrail が返り、当該フィールドを
    持たないこと（silently 無視・例外も属性追加もしない）を確認する。
    """
    g = regex_guardrail(r"\d", on="output", run_in_parallel=False)
    assert not hasattr(g, "run_in_parallel")
    g2 = external_detector_guardrail(
        lambda t: Detection(triggered=False), on="output", run_in_parallel=False
    )
    assert not hasattr(g2, "run_in_parallel")


# ----------------------------------------------------------------------
# on キーワード必須化（既定撤廃）・戻り型が SDK 互換型
# ----------------------------------------------------------------------


def test_two_boundary_factories_require_on_keyword() -> None:
    """二境界 factory は on をキーワード必須にした（位置引数 / 省略は TypeError）。"""
    with pytest.raises(TypeError):
        regex_guardrail(r"\d")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        predicate_guardrail(lambda t: False)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        length_guardrail(max_length=5)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        allow_deny_guardrail(deny=["x"])  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        external_detector_guardrail(lambda t: Detection(triggered=False))  # type: ignore[call-arg]


def test_factory_return_types_are_sdk_guardrails() -> None:
    """各 factory の戻り値が宣言した SDK 互換型のインスタンスであること（注釈の実値整合）。"""
    assert isinstance(canary_guardrail("X"), OutputGuardrail)
    assert isinstance(injection_baseline_guardrail(), InputGuardrail)
    assert isinstance(regex_guardrail(r"\d", on="input"), InputGuardrail)
    assert isinstance(regex_guardrail(r"\d", on="output"), OutputGuardrail)
    assert isinstance(length_guardrail(max_length=5, on="input"), InputGuardrail)
    assert isinstance(allow_deny_guardrail(deny=["x"], on="output"), OutputGuardrail)
    assert isinstance(
        external_detector_guardrail(lambda t: Detection(triggered=False), on="input"),
        InputGuardrail,
    )
    assert isinstance(predicate_guardrail(lambda t: False, on="output"), OutputGuardrail)
    assert isinstance(prompt_llm_guardrail(FakeModel(), "p", on="input"), InputGuardrail)
    tool = _make_tool()
    assert isinstance(guard_tool(tool, input_detector=_trip_detector()), FunctionTool)


# ----------------------------------------------------------------------
# tool_guardrail（function_tool 流儀・検知器 → ToolGuardrail）
# ----------------------------------------------------------------------


def test_tool_guardrail_input_returns_tool_input_guardrail() -> None:
    """on="input" は ToolInputGuardrail を返す。"""
    g = tool_guardrail(_trip_detector(), on="input")
    assert isinstance(g, ToolInputGuardrail)
    assert g.get_name() == "tool_input_guardrail"


def test_tool_guardrail_output_returns_tool_output_guardrail() -> None:
    """on="output" は ToolOutputGuardrail を返す。"""
    g = tool_guardrail(_trip_detector(), on="output")
    assert isinstance(g, ToolOutputGuardrail)
    assert g.get_name() == "tool_output_guardrail"


def test_tool_guardrail_invalid_on_raises_value_error() -> None:
    """on が 'input' / 'output' 以外なら ValueError。"""
    with pytest.raises(ValueError, match="on must be 'input' or 'output'"):
        tool_guardrail(_trip_detector(), on="sideways")


def test_tool_guardrail_custom_name() -> None:
    """name 指定で guardrail 名を上書きできる。"""
    g = tool_guardrail(_trip_detector(), on="input", name="custom_tool_guard")
    assert g.get_name() == "custom_tool_guard"


@pytest.mark.asyncio
async def test_tool_guardrail_on_trip_reject() -> None:
    """on_trip='reject'（既定）: trip 時に reject_content（メッセージは検知理由）。"""
    g = tool_guardrail(_trip_detector(), on="input")
    out: ToolGuardrailFunctionOutput = await g.guardrail_function(_tool_input_data())
    assert out.behavior["type"] == "reject_content"
    assert out.behavior["message"] == "tool trip"


@pytest.mark.asyncio
async def test_tool_guardrail_on_trip_raise() -> None:
    """on_trip='raise': trip 時に raise_exception。"""
    g = tool_guardrail(_trip_detector(), on="output", on_trip="raise")
    out = await g.guardrail_function(_tool_output_data("x"))
    assert out.behavior["type"] == "raise_exception"


@pytest.mark.asyncio
async def test_tool_guardrail_on_trip_allow() -> None:
    """on_trip='allow': trip しても allow（通過）。"""
    g = tool_guardrail(_trip_detector(), on="input", on_trip="allow")
    out = await g.guardrail_function(_tool_input_data())
    assert out.behavior["type"] == "allow"


@pytest.mark.asyncio
async def test_tool_guardrail_on_trip_callable() -> None:
    """on_trip に callable を渡すと Detection を受けて出力を組める（DI 分岐）。"""
    sentinel = ToolGuardrailFunctionOutput.allow(output_info={"via": "callable"})

    def on_trip(detection: Detection) -> ToolGuardrailFunctionOutput:
        assert detection.triggered is True
        return sentinel

    g = tool_guardrail(_trip_detector(), on="output", on_trip=on_trip)
    out = await g.guardrail_function(_tool_output_data("anything"))
    assert out is sentinel


@pytest.mark.asyncio
async def test_tool_guardrail_async_detector() -> None:
    """async 検知器（async def）でも coroutine を await して trip/pass する。"""

    async def detect(text: str) -> Detection:
        return Detection(triggered="leak" in text, reason="async out")

    g = tool_guardrail(detect, on="output")
    tripped = await g.guardrail_function(_tool_output_data("contains leak"))
    assert tripped.behavior["type"] == "reject_content"
    allowed = await g.guardrail_function(_tool_output_data("clean"))
    assert allowed.behavior["type"] == "allow"


def test_tool_guardrail_declared_via_function_tool() -> None:
    """function_tool(_func, tool_*_guardrails=[tool_guardrail(...)]) で FunctionTool に載る。"""

    def _func(query: str) -> str:
        """ツール。"""
        return query

    tool = function_tool(
        _func,
        tool_input_guardrails=[tool_guardrail(_trip_detector(), on="input")],
        tool_output_guardrails=[tool_guardrail(_trip_detector(), on="output")],
    )
    assert isinstance(tool, FunctionTool)
    assert len(tool.tool_input_guardrails) == 1
    assert len(tool.tool_output_guardrails) == 1
    assert tool.tool_input_guardrails[0].get_name() == "tool_input_guardrail"
    assert tool.tool_output_guardrails[0].get_name() == "tool_output_guardrail"


@pytest.mark.asyncio
async def test_function_tool_declared_output_guardrail_trips_via_runner() -> None:
    """function_tool で宣言したツール出力 guardrail が実 Agent + Runner で trip する。

    ツールが PII を返すと on_trip='raise' で ToolOutputGuardrailTripwireTriggered。
    """
    from agents import ToolOutputGuardrailTripwireTriggered

    def _lookup(customer_id: str) -> str:
        """顧客情報を返す（PII を含む）。"""
        return "contact: taro@example.com"

    def _email_detector(text: str) -> Detection:
        return Detection(triggered="@" in text, reason="email leak")

    tool = function_tool(
        _lookup,
        name_override="lookup",
        tool_output_guardrails=[tool_guardrail(_email_detector, on="output", on_trip="raise")],
    )
    # FakeModel が lookup ツールを呼ぶ -> ツール出力 guardrail が PII を検知して中断。
    model = FakeModel().queue_tool_call("lookup", '{"customer_id": "C-1"}').queue_text("done")
    spec = AgentSpec(name="bot", instructions="lookup ツールを必ず使う", model=model, tools=[tool])
    agent = build_agent(spec)
    with pytest.raises(ToolOutputGuardrailTripwireTriggered):
        await Runner.run(agent, "顧客 C-1 の連絡先は?")
