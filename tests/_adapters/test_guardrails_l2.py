"""L2: 内容ガードレールの SDK 結合窓口（`_adapters.guardrails`）を検証する。

plain な `Detection` → SDK `GuardrailFunctionOutput` / `ToolGuardrailFunctionOutput` への写像、
agent / tool 境界 guardrail の接着、`attach_tool_guardrails` の連結 / 素通し、`_resolve_trip` の
文字列 / callable / 不正値分岐、`run_judge_prompt` の judge 実行（FakeModel モック）を検証する。
SDK 型を直接組み立てて guardrail_function を呼ぶ（Runner を介さない単体写像の検証）。
"""

from __future__ import annotations

import pytest
from agents import (
    FunctionTool,
    ToolGuardrailFunctionOutput,
    ToolInputGuardrailData,
    ToolOutputGuardrailData,
)
from agents.tool_context import ToolContext

from oai_agentspec import function_tool
from oai_agentspec._adapters.guardrails import (
    _resolve_trip,
    attach_tool_guardrails,
    build_async_input_guardrail,
    build_async_output_guardrail,
    build_input_guardrail,
    build_output_guardrail,
    build_tool_input_guardrail,
    build_tool_output_guardrail,
    run_judge_prompt,
)
from oai_agentspec.runtime.guardrails._detectors import Detection

from _helpers.fake_model import FakeModel

pytestmark = pytest.mark.integration


# ----------------------------------------------------------------------
# helper: SDK guardrail データ構築
# ----------------------------------------------------------------------


def _tool_input_data(arguments: str = '{"q": "x"}') -> ToolInputGuardrailData:
    ctx = ToolContext(context=None, tool_name="t", tool_call_id="c1", tool_arguments=arguments)
    return ToolInputGuardrailData(context=ctx, agent=None)


def _tool_output_data(output: object) -> ToolOutputGuardrailData:
    ctx = ToolContext(context=None, tool_name="t", tool_call_id="c1", tool_arguments="{}")
    return ToolOutputGuardrailData(context=ctx, agent=None, output=output)


def _make_tool(name: str = "search") -> FunctionTool:
    @function_tool(name_override=name)
    def _tool(query: str) -> str:
        """ツール。"""
        return query

    return _tool


# ----------------------------------------------------------------------
# build_input_guardrail / build_output_guardrail（同期）: Detection → GuardrailFunctionOutput
# ----------------------------------------------------------------------


def test_build_input_guardrail_maps_triggered_to_tripwire() -> None:
    """triggered=True が tripwire_triggered=True に写り、reason / info が output_info に載る。"""
    g = build_input_guardrail("ig", lambda t: Detection(triggered=True, reason="r", info={"k": 1}))
    # guardrail_function は SDK の (context, agent, input) シグネチャの薄い包み。
    out = g.guardrail_function(None, None, "hello")
    assert out.tripwire_triggered is True
    assert out.output_info == {"reason": "r", "info": {"k": 1}}


def test_build_input_guardrail_not_triggered() -> None:
    """triggered=False は tripwire_triggered=False。"""
    g = build_input_guardrail("ig", lambda t: Detection(triggered=False))
    out = g.guardrail_function(None, None, "hello")
    assert out.tripwire_triggered is False


def test_build_output_guardrail_maps_triggered() -> None:
    """output guardrail も triggered → tripwire_triggered に写る。"""
    g = build_output_guardrail("og", lambda t: Detection(triggered=True, reason="bad"))
    out = g.guardrail_function(None, None, "answer")
    assert out.tripwire_triggered is True
    assert out.output_info["reason"] == "bad"


def test_build_input_guardrail_text_of_non_str() -> None:
    """非 str 入力は str(...) でテキスト化して detect に渡る。"""
    seen: list[str] = []

    def detect(text: str) -> Detection:
        seen.append(text)
        return Detection(triggered=False)

    g = build_input_guardrail("ig", detect)
    g.guardrail_function(None, None, [1, 2, 3])
    assert seen == [str([1, 2, 3])]


# ----------------------------------------------------------------------
# build_async_input/output_guardrail（同期 / 非同期 / async callable オブジェクトを一様に await）
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_build_async_input_guardrail() -> None:
    """非同期検知（await）の結果が tripwire に写る。"""

    async def detect(text: str) -> Detection:
        return Detection(triggered=True, reason="async")

    g = build_async_input_guardrail("aig", detect)
    out = await g.guardrail_function(None, None, "x")
    assert out.tripwire_triggered is True
    assert out.output_info["reason"] == "async"


@pytest.mark.asyncio
async def test_build_async_output_guardrail() -> None:
    """非同期 output 検知も写像が機能する。"""

    async def detect(text: str) -> Detection:
        return Detection(triggered=False)

    g = build_async_output_guardrail("aog", detect)
    out = await g.guardrail_function(None, None, "y")
    assert out.tripwire_triggered is False


@pytest.mark.asyncio
async def test_build_async_input_guardrail_accepts_sync_detector() -> None:
    """同期検知（非 awaitable な戻り値）を async builder に渡しても await せず正規化される。"""

    def detect(text: str) -> Detection:
        return Detection(triggered=True, reason="sync via async builder")

    g = build_async_input_guardrail("aig", detect)
    out = await g.guardrail_function(None, None, "x")
    assert out.tripwire_triggered is True
    assert out.output_info["reason"] == "sync via async builder"


@pytest.mark.asyncio
async def test_build_async_input_guardrail_accepts_async_callable_object() -> None:
    """`async __call__` を持つ callable オブジェクトでも戻り値 awaitable 判定で await される。

    `inspect.iscoroutinefunction` は async __call__ オブジェクトに False を返すため型判定では
    取りこぼすが、`inspect.isawaitable` による戻り値判定なら正しく await される回帰。
    """

    class _AsyncDetector:
        async def __call__(self, text: str) -> Detection:
            return Detection(triggered="bad" in text, reason="async obj")

    g = build_async_input_guardrail("aig", _AsyncDetector())
    tripped = await g.guardrail_function(None, None, "this is bad")
    assert tripped.tripwire_triggered is True
    assert tripped.output_info["reason"] == "async obj"
    passed = await g.guardrail_function(None, None, "all good")
    assert passed.tripwire_triggered is False


# ----------------------------------------------------------------------
# run_in_parallel の伝播（input builder のみ・OutputGuardrail には該当フィールドなし）
# ----------------------------------------------------------------------


def test_build_input_guardrail_run_in_parallel_default_true() -> None:
    """既定では生成 InputGuardrail.run_in_parallel が True（SDK 既定）。"""
    g = build_input_guardrail("ig", lambda t: Detection(triggered=False))
    assert g.run_in_parallel is True


def test_build_input_guardrail_run_in_parallel_false() -> None:
    """run_in_parallel=False を渡すと InputGuardrail.run_in_parallel が False になる。"""
    g = build_input_guardrail("ig", lambda t: Detection(triggered=False), run_in_parallel=False)
    assert g.run_in_parallel is False


def test_build_async_input_guardrail_run_in_parallel_default_true() -> None:
    """async builder も既定で run_in_parallel=True。"""

    async def detect(text: str) -> Detection:
        return Detection(triggered=False)

    g = build_async_input_guardrail("aig", detect)
    assert g.run_in_parallel is True


def test_build_async_input_guardrail_run_in_parallel_false() -> None:
    """async builder でも run_in_parallel=False が伝播する。"""

    async def detect(text: str) -> Detection:
        return Detection(triggered=False)

    g = build_async_input_guardrail("aig", detect, run_in_parallel=False)
    assert g.run_in_parallel is False


# ----------------------------------------------------------------------
# build_tool_input_guardrail / build_tool_output_guardrail
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_build_tool_input_guardrail_inspects_arguments_and_allows_when_clean() -> None:
    """triggered=False のときは allow を返す（tool guardrail 関数は常に async）。"""
    g = build_tool_input_guardrail("tig", lambda t: Detection(triggered=False))
    out = await g.guardrail_function(_tool_input_data())
    assert out.behavior["type"] == "allow"


@pytest.mark.asyncio
async def test_build_tool_input_guardrail_resolves_trip_reject() -> None:
    """triggered のとき on_trip 既定（reject）で reject_content を返し、引数テキストを検査する。"""
    seen: list[str] = []

    def detect(text: str) -> Detection:
        seen.append(text)
        return Detection(triggered=True, reason="arg trip")

    g = build_tool_input_guardrail("tig", detect)
    out = await g.guardrail_function(_tool_input_data('{"q": "danger"}'))
    assert out.behavior["type"] == "reject_content"
    assert out.behavior["message"] == "arg trip"
    assert seen == ['{"q": "danger"}']  # data.context.tool_arguments を検査


@pytest.mark.asyncio
async def test_build_tool_output_guardrail_inspects_output() -> None:
    """data.output をテキスト化して検査し、trip 時 reject_content を返す。"""

    def detect(text: str) -> Detection:
        return Detection(triggered="leak" in text, reason="out trip")

    g = build_tool_output_guardrail("tog", detect)
    tripped = await g.guardrail_function(_tool_output_data("contains leak"))
    assert tripped.behavior["type"] == "reject_content"
    allowed = await g.guardrail_function(_tool_output_data("clean"))
    assert allowed.behavior["type"] == "allow"


@pytest.mark.asyncio
async def test_build_tool_input_guardrail_missing_arguments_falls_back_to_empty() -> None:
    """tool_arguments を持たない context でも空文字フォールバックで検知器が呼ばれる。"""
    seen: list[str] = []

    def detect(text: str) -> Detection:
        seen.append(text)
        return Detection(triggered=False)

    class _Ctx:
        """tool_arguments 属性を持たない最小 context スタブ。"""

    class _Data:
        context = _Ctx()

    g = build_tool_input_guardrail("tig", detect)
    out = await g.guardrail_function(_Data())  # type: ignore[arg-type]
    assert out.behavior["type"] == "allow"
    assert seen == [""]  # getattr フォールバックで空文字


# ----------------------------------------------------------------------
# build_tool_*_guardrail（非同期検知器・async callable オブジェクト）: 戻り値が awaitable なら await
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_build_tool_input_guardrail_async_detector() -> None:
    """async 検知器（async def）でも guardrail 関数が coroutine を await して写像する。"""

    async def detect(text: str) -> Detection:
        return Detection(triggered="danger" in text, reason="async arg")

    g = build_tool_input_guardrail("tig", detect)
    tripped = await g.guardrail_function(_tool_input_data('{"q": "danger"}'))
    assert tripped.behavior["type"] == "reject_content"
    assert tripped.behavior["message"] == "async arg"
    allowed = await g.guardrail_function(_tool_input_data('{"q": "ok"}'))
    assert allowed.behavior["type"] == "allow"


@pytest.mark.asyncio
async def test_build_tool_output_guardrail_async_detector() -> None:
    """async output 検知器でも await されて trip/pass が機能する（AttributeError を起こさない）。"""

    async def detect(text: str) -> Detection:
        return Detection(triggered="leak" in text, reason="async out")

    g = build_tool_output_guardrail("tog", detect)
    tripped = await g.guardrail_function(_tool_output_data("contains leak"))
    assert tripped.behavior["type"] == "reject_content"
    allowed = await g.guardrail_function(_tool_output_data("clean"))
    assert allowed.behavior["type"] == "allow"


@pytest.mark.asyncio
async def test_build_tool_output_guardrail_async_callable_object_detector() -> None:
    """`async __call__` を持つ callable オブジェクト（型は coroutine 関数でない）でも await される。

    `inspect.iscoroutinefunction` は async __call__ オブジェクトに False を返すため、型判定では
    取りこぼす。戻り値 awaitable 判定（`inspect.isawaitable`）なら正しく await される回帰。
    """

    class _AsyncDetector:
        async def __call__(self, text: str) -> Detection:
            return Detection(triggered="leak" in text, reason="async obj")

    g = build_tool_output_guardrail("tog", _AsyncDetector())
    tripped = await g.guardrail_function(_tool_output_data("contains leak"))
    assert tripped.behavior["type"] == "reject_content"
    assert tripped.behavior["message"] == "async obj"
    allowed = await g.guardrail_function(_tool_output_data("clean"))
    assert allowed.behavior["type"] == "allow"


# ----------------------------------------------------------------------
# _resolve_trip: 文字列 / callable / 不正値
# ----------------------------------------------------------------------


def test_resolve_trip_string_reject() -> None:
    """'reject' は reject_content（メッセージは検知理由）。"""
    out = _resolve_trip("reject", Detection(triggered=True, reason="why"))
    assert out.behavior["type"] == "reject_content"
    assert out.behavior["message"] == "why"


def test_resolve_trip_string_reject_default_message() -> None:
    """reason が None のとき reject の message は既定文言。"""
    out = _resolve_trip("reject", Detection(triggered=True))
    assert out.behavior["type"] == "reject_content"
    assert out.behavior["message"] == "guardrail tripped"


def test_resolve_trip_string_raise() -> None:
    """'raise' は raise_exception。"""
    out = _resolve_trip("raise", Detection(triggered=True))
    assert out.behavior["type"] == "raise_exception"


def test_resolve_trip_string_allow() -> None:
    """'allow' は allow。"""
    out = _resolve_trip("allow", Detection(triggered=True))
    assert out.behavior["type"] == "allow"


def test_resolve_trip_callable_is_invoked() -> None:
    """callable はそのまま呼ばれ Detection を受けて出力を委ねる。"""
    sentinel = ToolGuardrailFunctionOutput.allow(output_info={"x": 1})
    out = _resolve_trip(lambda d: sentinel, Detection(triggered=True))
    assert out is sentinel


def test_resolve_trip_unknown_string_raises_value_error() -> None:
    """未知の文字列定数（OnTrip Literal 外）は ValueError を上げる（契約統一・typo 早期検出）。

    agent 境界の `on` 引数が不正値で ValueError を上げるのと対称に、`on_trip` の不正文字列も
    黙って reject へフォールバックせず ValueError を上げる。callable はこの検証の対象外。
    """
    with pytest.raises(ValueError, match="on_trip must be"):
        _resolve_trip("bogus", Detection(triggered=True, reason="x"))  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# attach_tool_guardrails: 連結 / 素通し
# ----------------------------------------------------------------------


def test_attach_tool_guardrails_none_returns_original() -> None:
    """input / output 双方 None なら元 tool をそのまま返す（素通し）。"""
    tool = _make_tool()
    assert attach_tool_guardrails(tool) is tool


def test_attach_tool_guardrails_attaches_input_and_output() -> None:
    """input / output guardrail を装着した新 tool を返す（元 tool は不変）。"""
    tool = _make_tool()
    ig = build_tool_input_guardrail("tig", lambda t: Detection(triggered=False))
    og = build_tool_output_guardrail("tog", lambda t: Detection(triggered=False))
    guarded = attach_tool_guardrails(tool, input=ig, output=og)
    assert guarded is not tool
    assert guarded.tool_input_guardrails == [ig]
    assert guarded.tool_output_guardrails == [og]
    # 元 tool は不変。
    assert tool.tool_input_guardrails is None


def test_attach_tool_guardrails_concatenates_existing() -> None:
    """既存 guardrails がある tool には連結する（置換しない）。"""
    tool = _make_tool()
    first = build_tool_input_guardrail("first", lambda t: Detection(triggered=False))
    once = attach_tool_guardrails(tool, input=first)
    second = build_tool_input_guardrail("second", lambda t: Detection(triggered=False))
    twice = attach_tool_guardrails(once, input=second)
    assert twice.tool_input_guardrails == [first, second]


# ----------------------------------------------------------------------
# run_judge_prompt: judge 実行（FakeModel モック）
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_judge_prompt_returns_final_output_text() -> None:
    """judge model を 1 ターン実行し final_output テキストを返す。"""
    model = FakeModel().queue_text("UNSAFE: reason")
    out = await run_judge_prompt(model, "judge prompt", "content to check")
    assert out == "UNSAFE: reason"


@pytest.mark.asyncio
async def test_run_judge_prompt_empty_output_returns_empty_string() -> None:
    """judge の final_output が空（None 相当）のとき空文字を返す。"""
    # FakeModel は空キューで空テキストを返す（final_output は ""）。
    model = FakeModel()
    out = await run_judge_prompt(model, "judge prompt", "content")
    assert out == ""
