"""L2: 内容ガードレールの SDK 結合窓口（`_adapters.guardrails`）を検証する。

plain な `Detection` → SDK `GuardrailFunctionOutput` / `ToolGuardrailFunctionOutput` への写像、
agent / tool 境界 guardrail の接着、`attach_tool_guardrails` の連結 / 素通し、`_resolve_trip` の
文字列 / callable / 不正値分岐、`run_judge_prompt` の judge 実行（FakeModel モック）を検証する。
SDK 型を直接組み立てて guardrail_function を呼ぶ（Runner を介さない単体写像の検証）。
"""

from __future__ import annotations

import functools
import inspect

import pytest
from agents import (
    FunctionTool,
    InputGuardrail,
    OutputGuardrail,
    RunConfig,
    ToolGuardrailFunctionOutput,
    ToolInputGuardrail,
    ToolInputGuardrailData,
    ToolOutputGuardrail,
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
    guardrail_boundary,
    guardrail_visible_name,
    run_judge_prompt,
)
from oai_agentspec.runtime.guardrails._detectors import Detection
from oai_agentspec.runtime.guardrails.types import Boundary

from _helpers.fake_model import FakeModel

pytestmark = pytest.mark.integration


# ----------------------------------------------------------------------
# helper: SDK guardrail データ構築
# ----------------------------------------------------------------------


def _always_trigger(text: str) -> Detection:
    """常に trip する同期検知（可視名 / 境界判定の pin で検知内容は問わないため）。"""
    return Detection(triggered=True, reason="pinned")


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


# ----------------------------------------------------------------------
# 宣言的登録の SDK 接着（可視名の読み取り / 境界種別の判定）
# ----------------------------------------------------------------------


def test_guardrail_visible_name_は4型すべてでget_nameの戻り値を返す() -> None:
    """上流 4 型の `get_name()` をそのまま返す（登録キーとの照合に使う唯一の窓口）。"""
    detect = _always_trigger
    assert guardrail_visible_name(build_input_guardrail("in-name", detect)) == "in-name"
    assert guardrail_visible_name(build_output_guardrail("out-name", detect)) == "out-name"
    assert guardrail_visible_name(build_tool_input_guardrail("ti-name", detect)) == "ti-name"
    assert guardrail_visible_name(build_tool_output_guardrail("to-name", detect)) == "to-name"


def test_guardrail_boundary_は4型を対応する境界文字列へ判定する() -> None:
    """上流 4 型 → 境界文字列の写像を固定する（型分離に依拠した判定の pin）。"""
    detect = _always_trigger
    assert guardrail_boundary(build_input_guardrail("a", detect)) == "input"
    assert guardrail_boundary(build_output_guardrail("b", detect)) == "output"
    assert guardrail_boundary(build_tool_input_guardrail("c", detect)) == "tool_input"
    assert guardrail_boundary(build_tool_output_guardrail("d", detect)) == "tool_output"


def test_guardrail_boundary_は4型でない実体にNoneを返す() -> None:
    """上流 guardrail 型でない実体は境界を推測せず `None`（呼び出し側が拒否できる形）。"""

    class _NotAGuardrail:
        name = "duck"

        def get_name(self) -> str:
            return "duck"

    assert guardrail_boundary(_NotAGuardrail()) is None
    assert guardrail_boundary(object()) is None
    assert guardrail_boundary(None) is None


def test_guardrail_visible_name_はname未設定かつ__name__なしでAttributeErrorになる() -> None:
    """境界判定を通った実体でも可視名の取得は失敗しうる（docstring の Raises 契約の pin）。

    上流 4 型の `get_name()` は `name or guardrail_function.__name__` 相当のため、`name` を
    渡さず `functools.partial` を guardrail 関数にした実体は、型としては正当（`isinstance` を
    通る）だが可視名を取得できない。「境界判定を通れば可視名取得は安全」と読める実装を
    後段（登録簿）で書くと、集約された `ValueError` ではなく生の `AttributeError` が利用者へ
    漏れるため、非対称性をここで固定する。
    """
    partial_fn = functools.partial(_always_trigger)
    guardrail = InputGuardrail(guardrail_function=partial_fn)

    assert guardrail_boundary(guardrail) == "input"
    with pytest.raises(AttributeError):
        guardrail_visible_name(guardrail)


def test_guardrail_boundary_の戻り値集合は4境界に固定される() -> None:
    """`guardrail_boundary` が返しうる非 None 値は 4 境界のみ。

    集合の `==` で pin する（境界の追加 = 過大と切り詰め = 過小の両方向を同時に検知するため。
    値域は `Boundary`（`runtime.guardrails.types`）と二重に存在するので、乖離をここで固定する）。
    """
    detect = _always_trigger
    produced = {
        guardrail_boundary(build_input_guardrail("a", detect)),
        guardrail_boundary(build_output_guardrail("b", detect)),
        guardrail_boundary(build_tool_input_guardrail("c", detect)),
        guardrail_boundary(build_tool_output_guardrail("d", detect)),
    }
    assert produced == {b.value for b in Boundary}


# ----------------------------------------------------------------------
# SDK バージョン耐性トリップワイヤ（NFR-5 の 4 前提）
# ----------------------------------------------------------------------


def test_sdk_guardrail_api_tripwire_name_injection_and_get_name() -> None:
    """前提 1 / 2: 渡した `name` が `get_name()` の戻り値になり、4 型が `get_name` を持つ。

    この前提が崩れると「登録キー = 上流 SDK 可視名」の構造的一致が成立しなくなるため、
    追従要否の判断ポイントとして fail させる。
    """
    detect = _always_trigger
    for factory, expected in (
        (build_input_guardrail, "pin-in"),
        (build_output_guardrail, "pin-out"),
        (build_tool_input_guardrail, "pin-ti"),
        (build_tool_output_guardrail, "pin-to"),
    ):
        guardrail = factory(expected, detect)
        assert hasattr(guardrail, "get_name"), f"{factory.__name__}: get_name が消えた"
        assert guardrail.get_name() == expected, f"{factory.__name__}: name 注入が効いていない"


def test_sdk_guardrail_api_tripwire_four_types_are_distinct() -> None:
    """前提 3: 上流 guardrail 4 型が相互に継承関係を持たない別型として分離されている。

    型分離が崩れると `isinstance` による境界判定が誤答するため、判断ポイントとして fail させる。
    """
    types = (InputGuardrail, OutputGuardrail, ToolInputGuardrail, ToolOutputGuardrail)
    assert len({t.__name__ for t in types}) == 4
    for outer in types:
        for inner in types:
            if outer is inner:
                continue
            assert not issubclass(outer, inner), f"{outer.__name__} が {inner.__name__} を継承した"


def test_sdk_run_config_api_tripwire_guardrail_parameter_names() -> None:
    """前提 4: `RunConfig` が `input_guardrails` / `output_guardrails` の引数名を持つ。

    run 単位へ渡す境界別マッピングのキーがこの名前に依拠するため、上流の改名・廃止を
    ここで検知する（改名しても壊れるのは利用者コードで、本リポジトリの CI は緑のまま通る）。
    """
    params = inspect.signature(RunConfig).parameters
    assert "input_guardrails" in params
    assert "output_guardrails" in params
    # 空のマッピングを展開しても構築できる（キー名が揃っていることの実行時確認）。
    assert RunConfig(input_guardrails=[], output_guardrails=[]) is not None
