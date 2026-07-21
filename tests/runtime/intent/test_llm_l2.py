"""L2: `runtime.intent._llm.LLMCandidateGenerator` の LLM 呼び出し・parse・policy 適用挙動。

`_adapters.intent.run_intent_prompt` を monkeypatch で差し替えて capture し、
LLM 出力 JSON を IntentPrediction へパースする経路と、policy allowlist /
max_candidates の適用、レベル降順ソート、system prompt 注入切替、
allowlist 外候補の silent 除外 + warning ログを検証する。
実 SDK / 実 LLM は使わず、IntentFakeModel と関数差し替えのみで確認する。
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import pytest

from oai_agentspec.runtime.intent.types import (
    ConfidenceLevel,
    ConsistencyReport,
    IntentCategory,
    IntentContext,
    IntentPolicy,
    IntentPrediction,
)

from _helpers.intent_fakes import IntentFakeModel

pytestmark = pytest.mark.integration


# ---- 共通ヘルパ ----


def _policy(
    *,
    max_candidates: int = 3,
    extra_instructions: str = "",
    names: tuple[str, ...] = ("ask", "chitchat", "task"),
) -> IntentPolicy:
    """テスト用 IntentPolicy を組み立てる。"""
    categories = tuple(IntentCategory(name=n, description=f"desc-{n}") for n in names)
    return IntentPolicy(
        categories=categories,
        max_candidates=max_candidates,
        extra_instructions=extra_instructions,
    )


def _ctx(
    utterance: str = "hi",
    history_items: tuple[Mapping[str, Any], ...] = (),
    run_context: Any = None,
) -> IntentContext[Any]:
    """テスト用 IntentContext を返す。"""
    return IntentContext(
        utterance=utterance,
        history_items=history_items,
        run_context=run_context,
    )


def _make_generator(
    text: str,
    *,
    policy: IntentPolicy | None = None,
    prompt: Any = None,
    include_policy_in_system: bool = True,
) -> Any:
    """LLMCandidateGenerator を IntentFakeModel + 固定 prompt callable で組み立てる。"""
    from oai_agentspec.runtime.intent._llm import LLMCandidateGenerator

    model = IntentFakeModel(text=text)
    p = prompt if prompt is not None else (lambda _ctx: "USER_PROMPT")
    return LLMCandidateGenerator(
        model,
        p,
        policy=policy or _policy(),
        include_policy_in_system=include_policy_in_system,
    )


class _Capturer:
    """run_intent_prompt の呼び出し引数を捕捉する差し替え関数（返却テキスト固定）。"""

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def __call__(
        self,
        model: Any,
        system: str,
        history_items: tuple[Mapping[str, Any], ...],
        user_content: str,
        *,
        context: Any = None,
    ) -> str:
        self.calls.append(
            {
                "model": model,
                "system": system,
                "history_items": history_items,
                "user_content": user_content,
                "context": context,
            }
        )
        return self.response


def _patch_run_intent_prompt(monkeypatch: pytest.MonkeyPatch, capturer: _Capturer) -> None:
    """`_adapters.intent.run_intent_prompt` を差し替える（利用側 import パスも同時差し替え）。

    実装が `from ..._adapters.intent import run_intent_prompt` で name binding する
    場合と `from ..._adapters import intent; intent.run_intent_prompt(...)` の場合の
    両方に耐えるため、原本と利用側の両方を差し替える。
    """
    monkeypatch.setattr("oai_agentspec._adapters.intent.run_intent_prompt", capturer, raising=True)
    import importlib

    try:
        mod = importlib.import_module("oai_agentspec.runtime.intent._llm")
    except Exception:  # pragma: no cover
        return
    if hasattr(mod, "run_intent_prompt"):
        monkeypatch.setattr(mod, "run_intent_prompt", capturer, raising=True)


# ---- 正常系 (parse / sort / report / metadata) ----


async def test_single_candidate_parsed() -> None:
    """1 件の candidate JSON が IntentPrediction に反映される。"""
    text = (
        '{"candidates":[{"text":"ask","level":"high","rationale":"r1"}],'
        '"report":null,"metadata":null}'
    )
    gen = _make_generator(text)
    pred = await gen.generate(_ctx())
    assert isinstance(pred, IntentPrediction)
    assert len(pred.candidates) == 1
    c = pred.candidates[0]
    assert c.text == "ask"
    assert c.level == ConfidenceLevel.HIGH
    assert c.rationale == "r1"


async def test_multiple_candidates_included() -> None:
    """複数 candidate が全て含まれる（allowlist 内）。"""
    text = (
        '{"candidates":['
        '{"text":"ask","level":"certain","rationale":"a"},'
        '{"text":"chitchat","level":"high","rationale":"b"},'
        '{"text":"task","level":"medium","rationale":"c"}'
        '],"report":null,"metadata":null}'
    )
    gen = _make_generator(text)
    pred = await gen.generate(_ctx())
    assert [c.text for c in pred.candidates] == ["ask", "chitchat", "task"]


async def test_candidates_already_sorted_preserved() -> None:
    """レベル降順で来たら順序保存（CERTAIN, HIGH, MEDIUM）。"""
    text = (
        '{"candidates":['
        '{"text":"ask","level":"certain","rationale":"a"},'
        '{"text":"chitchat","level":"high","rationale":"b"},'
        '{"text":"task","level":"medium","rationale":"c"}'
        '],"report":null,"metadata":null}'
    )
    gen = _make_generator(text)
    pred = await gen.generate(_ctx())
    assert [c.level for c in pred.candidates] == [
        ConfidenceLevel.CERTAIN,
        ConfidenceLevel.HIGH,
        ConfidenceLevel.MEDIUM,
    ]


async def test_candidates_unsorted_are_sorted_by_level_desc() -> None:
    """レベル不順 (MEDIUM, CERTAIN, HIGH) → 降順にソート。"""
    text = (
        '{"candidates":['
        '{"text":"ask","level":"medium","rationale":"a"},'
        '{"text":"chitchat","level":"certain","rationale":"b"},'
        '{"text":"task","level":"high","rationale":"c"}'
        '],"report":null,"metadata":null}'
    )
    gen = _make_generator(text)
    pred = await gen.generate(_ctx())
    assert [c.level for c in pred.candidates] == [
        ConfidenceLevel.CERTAIN,
        ConfidenceLevel.HIGH,
        ConfidenceLevel.MEDIUM,
    ]
    assert [c.text for c in pred.candidates] == ["chitchat", "task", "ask"]


async def test_same_level_preserves_llm_output_order() -> None:
    """同レベル内 (HIGH1, HIGH2) は LLM 出力順を保持する（stable sort）。"""
    text = (
        '{"candidates":['
        '{"text":"ask","level":"high","rationale":"first"},'
        '{"text":"chitchat","level":"high","rationale":"second"}'
        '],"report":null,"metadata":null}'
    )
    gen = _make_generator(text)
    pred = await gen.generate(_ctx())
    assert [c.rationale for c in pred.candidates] == ["first", "second"]


async def test_report_populated_when_present() -> None:
    """report フィールドが埋まっていれば ConsistencyReport として反映される。"""
    text = (
        '{"candidates":[{"text":"ask","level":"high","rationale":"r"}],'
        '"report":{"conflicts":["c1"],"stale_context":["s1"],"over_inference":["o1"]},'
        '"metadata":null}'
    )
    gen = _make_generator(text)
    pred = await gen.generate(_ctx())
    assert isinstance(pred.report, ConsistencyReport)
    assert pred.report.conflicts == ("c1",)
    assert pred.report.stale_context == ("s1",)
    assert pred.report.over_inference == ("o1",)


async def test_report_none_preserved() -> None:
    """report=null の場合 IntentPrediction.report は None。"""
    text = (
        '{"candidates":[{"text":"ask","level":"high","rationale":"r"}],'
        '"report":null,"metadata":null}'
    )
    gen = _make_generator(text)
    pred = await gen.generate(_ctx())
    assert pred.report is None


async def test_metadata_passthrough() -> None:
    """LLM 出力の metadata が IntentPrediction.metadata にそのまま反映される。"""
    text = (
        '{"candidates":[{"text":"ask","level":"high","rationale":"r"}],'
        '"report":null,"metadata":{"source":"llm","k":1}}'
    )
    gen = _make_generator(text)
    pred = await gen.generate(_ctx())
    assert pred.metadata is not None
    assert pred.metadata.get("source") == "llm"
    assert pred.metadata.get("k") == 1


async def test_metadata_none_when_llm_returns_null() -> None:
    """LLM の metadata が null なら結果 metadata も None（rejected 記録は無い）。"""
    text = (
        '{"candidates":['
        '{"text":"ask","level":"high","rationale":"a"},'
        '{"text":"unknown","level":"high","rationale":"b"}'
        '],"report":null,"metadata":null}'
    )
    gen = _make_generator(text)
    pred = await gen.generate(_ctx())
    assert pred.metadata is None


# ---- policy 制約 (allowlist / max_candidates) ----


async def test_out_of_allowlist_silent_removal() -> None:
    """policy.categories に無い text は silent 除外・metadata に rejected キーは入らない。"""
    text = (
        '{"candidates":['
        '{"text":"ask","level":"high","rationale":"a"},'
        '{"text":"unknown","level":"high","rationale":"b"}'
        '],"report":null,"metadata":null}'
    )
    gen = _make_generator(text)
    pred = await gen.generate(_ctx())
    assert [c.text for c in pred.candidates] == ["ask"]
    # metadata が None または dict でも "rejected" キーは存在しない（pass-through 仕様）
    if pred.metadata is not None:
        assert "rejected" not in pred.metadata


async def test_all_removed_returns_empty_candidates() -> None:
    """全候補が allowlist 外 → candidates=() で IntentPrediction を返す（metadata 汚染なし）。"""
    text = (
        '{"candidates":['
        '{"text":"unknown1","level":"high","rationale":"a"},'
        '{"text":"unknown2","level":"high","rationale":"b"}'
        '],"report":null,"metadata":null}'
    )
    gen = _make_generator(text)
    pred = await gen.generate(_ctx())
    assert pred.candidates == ()
    # LLM が null を返しているので pass-through で None のまま
    assert pred.metadata is None


async def test_out_of_allowlist_emits_warning(caplog: pytest.LogCaptureFixture) -> None:
    """allowlist 除外があると `_llm` の logger が WARNING を発火する。"""
    text = (
        '{"candidates":['
        '{"text":"ask","level":"high","rationale":"a"},'
        '{"text":"unknown","level":"high","rationale":"b"},'
        '{"text":"another_unknown","level":"medium","rationale":"c"}'
        '],"report":null,"metadata":null}'
    )
    gen = _make_generator(text)
    with caplog.at_level(logging.WARNING, logger="oai_agentspec.runtime.intent._llm"):
        pred = await gen.generate(_ctx())
    assert [c.text for c in pred.candidates] == ["ask"]
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) >= 1
    msg = warnings[-1].getMessage()
    # 除外件数と除外テキスト一覧が含まれる（両方 pin する）
    assert "2" in msg
    assert "unknown" in msg
    assert "another_unknown" in msg


async def test_no_warning_when_no_removal(caplog: pytest.LogCaptureFixture) -> None:
    """全候補が allowlist 内なら WARNING は 1 件も発火しない。"""
    text = (
        '{"candidates":['
        '{"text":"ask","level":"high","rationale":"a"},'
        '{"text":"chitchat","level":"medium","rationale":"b"}'
        '],"report":null,"metadata":null}'
    )
    gen = _make_generator(text)
    with caplog.at_level(logging.WARNING, logger="oai_agentspec.runtime.intent._llm"):
        await gen.generate(_ctx())
    warnings = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and r.name == "oai_agentspec.runtime.intent._llm"
    ]
    assert warnings == []


async def test_max_candidates_truncates_after_sort() -> None:
    """max_candidates=2 で 5 候補 → レベル降順ソート後の上位 2 件だけ残る。"""
    text = (
        '{"candidates":['
        '{"text":"ask","level":"medium","rationale":"a"},'
        '{"text":"chitchat","level":"certain","rationale":"b"},'
        '{"text":"task","level":"high","rationale":"c"},'
        '{"text":"ask","level":"low","rationale":"d"},'
        '{"text":"chitchat","level":"speculative","rationale":"e"}'
        '],"report":null,"metadata":null}'
    )
    gen = _make_generator(text, policy=_policy(max_candidates=2))
    pred = await gen.generate(_ctx())
    assert len(pred.candidates) == 2
    assert [c.level for c in pred.candidates] == [
        ConfidenceLevel.CERTAIN,
        ConfidenceLevel.HIGH,
    ]


# ---- adapter signature / system prompt 注入 ----


async def test_include_policy_in_system_true_injects_render_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """include_policy_in_system=True (デフォルト) → system 引数に policy.render_prompt() が渡る。"""
    response = (
        '{"candidates":[{"text":"ask","level":"high","rationale":"r"}],'
        '"report":null,"metadata":null}'
    )
    cap = _Capturer(response)
    _patch_run_intent_prompt(monkeypatch, cap)
    policy = _policy()
    gen = _make_generator(response, policy=policy, include_policy_in_system=True)
    await gen.generate(_ctx())
    assert len(cap.calls) == 1
    assert cap.calls[0]["system"] == policy.render_prompt()


async def test_include_policy_in_system_false_uses_empty_system(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """include_policy_in_system=False → system 引数は空文字。"""
    response = (
        '{"candidates":[{"text":"ask","level":"high","rationale":"r"}],'
        '"report":null,"metadata":null}'
    )
    cap = _Capturer(response)
    _patch_run_intent_prompt(monkeypatch, cap)
    gen = _make_generator(response, include_policy_in_system=False)
    await gen.generate(_ctx())
    assert len(cap.calls) == 1
    assert cap.calls[0]["system"] == ""


async def test_prompt_callable_result_passed_as_user_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """prompt callable が呼ばれ、その戻り値が adapter の user_content に渡る。"""
    response = (
        '{"candidates":[{"text":"ask","level":"high","rationale":"r"}],'
        '"report":null,"metadata":null}'
    )
    cap = _Capturer(response)
    _patch_run_intent_prompt(monkeypatch, cap)

    seen: list[IntentContext[Any]] = []

    def prompt(ctx: IntentContext[Any]) -> str:
        seen.append(ctx)
        return f"PROMPTED::{ctx.utterance}"

    gen = _make_generator(response, prompt=prompt)
    ctx = _ctx(utterance="hello")
    await gen.generate(ctx)

    assert seen == [ctx]
    assert cap.calls[0]["user_content"] == "PROMPTED::hello"


async def test_history_items_passed_to_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    """context.history_items が adapter の history_items 引数へ tuple で渡る。"""
    response = (
        '{"candidates":[{"text":"ask","level":"high","rationale":"r"}],'
        '"report":null,"metadata":null}'
    )
    cap = _Capturer(response)
    _patch_run_intent_prompt(monkeypatch, cap)
    items: tuple[Mapping[str, Any], ...] = (
        {"role": "user", "content": "prev-1"},
        {"role": "assistant", "content": "prev-2"},
    )
    gen = _make_generator(response)
    await gen.generate(_ctx(history_items=items))
    assert len(cap.calls) == 1
    assert cap.calls[0]["history_items"] == items
    assert isinstance(cap.calls[0]["history_items"], tuple)


async def test_run_context_forwarded_to_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    """context.run_context が adapter の context= キーワード引数に渡る。"""
    response = (
        '{"candidates":[{"text":"ask","level":"high","rationale":"r"}],'
        '"report":null,"metadata":null}'
    )
    cap = _Capturer(response)
    _patch_run_intent_prompt(monkeypatch, cap)
    sentinel = object()
    gen = _make_generator(response)
    await gen.generate(_ctx(run_context=sentinel))
    assert len(cap.calls) == 1
    assert cap.calls[0]["context"] is sentinel


async def test_history_only_classification_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    """履歴のみ（utterance="" + history_items あり）の generate が動作する（Issue #24）。

    prompt callable が ctx.utterance を素通しする最小構成で、adapter 差し替え先の
    `_Capturer` が `user_content == ""` と履歴 tuple をそのまま受けることを pin する
    （adapter は差し替え済みなので ValueError にはならない統合視点の確認）。
    """
    response = (
        '{"candidates":[{"text":"ask","level":"high","rationale":"r"}],'
        '"report":null,"metadata":null}'
    )
    cap = _Capturer(response)
    _patch_run_intent_prompt(monkeypatch, cap)
    items: tuple[Mapping[str, Any], ...] = ({"role": "user", "content": "過去発話"},)
    gen = _make_generator(response, prompt=lambda ctx: ctx.utterance)
    pred = await gen.generate(_ctx(utterance="", history_items=items))
    assert isinstance(pred, IntentPrediction)
    assert len(cap.calls) == 1
    assert cap.calls[0]["user_content"] == ""
    assert cap.calls[0]["history_items"] == items


# ---- pydantic 型検査失敗 ----


async def test_invalid_json_raises() -> None:
    """LLM が不正 JSON を返す → 構造破綻として ValidationError が伝播する。"""
    from pydantic import ValidationError

    gen = _make_generator("this is not json")
    with pytest.raises(ValidationError):
        await gen.generate(_ctx())


async def test_unknown_confidence_level_raises_validation_error() -> None:
    """LLM が未知の ConfidenceLevel を返す → pydantic ValidationError（構造破綻）。"""
    from pydantic import ValidationError

    text = (
        '{"candidates":[{"text":"ask","level":"unknown","rationale":"r"}],'
        '"report":null,"metadata":null}'
    )
    gen = _make_generator(text)
    with pytest.raises(ValidationError):
        await gen.generate(_ctx())


# ---- _LEVEL_ORDER ----


def test_level_order_derived_from_enum_declaration() -> None:
    """_LEVEL_ORDER の keys 順は ConfidenceLevel の宣言順と一致する。"""
    from oai_agentspec.runtime.intent._llm import _LEVEL_ORDER

    assert list(_LEVEL_ORDER) == list(ConfidenceLevel)


def test_level_order_covers_all_confidence_levels() -> None:
    """_LEVEL_ORDER は ConfidenceLevel の全メンバーを漏れなくカバーする。"""
    from oai_agentspec.runtime.intent._llm import _LEVEL_ORDER

    assert set(_LEVEL_ORDER) == set(ConfidenceLevel)


def test_level_order_values_are_contiguous_from_zero() -> None:
    """_LEVEL_ORDER の値は 0 から始まる連番。"""
    from oai_agentspec.runtime.intent._llm import _LEVEL_ORDER

    assert list(_LEVEL_ORDER.values()) == list(range(len(ConfidenceLevel)))


# ---------------------------------------------------------------------------
# _strip_code_fence / コードフェンス耐性 (Issue #24: 低精度・高速モデル対応)
# ---------------------------------------------------------------------------


def test_strip_code_fence_pins() -> None:
    """_strip_code_fence の入出力を直接 pin する。"""
    from oai_agentspec.runtime.intent._llm import _strip_code_fence

    assert _strip_code_fence('```json\n{"a":1}\n```') == '{"a":1}'
    assert _strip_code_fence('```\n{"a":1}\n```') == '{"a":1}'
    assert _strip_code_fence('{"a":1}') == '{"a":1}'
    assert _strip_code_fence('  {"a":1}  ') == '{"a":1}'
    assert _strip_code_fence("```") == "```"


async def test_fenced_json_response_is_parsed() -> None:
    """```json ... ``` で包まれた応答でも IntentPrediction へ正常にパースされる。"""
    body = (
        '{"candidates":[{"text":"ask","level":"high","rationale":"r"}],'
        '"report":null,"metadata":null}'
    )
    text = f"```json\n{body}\n```"
    gen = _make_generator(text)
    pred = await gen.generate(_ctx())
    assert isinstance(pred, IntentPrediction)
    assert len(pred.candidates) == 1
    assert pred.candidates[0].text == "ask"


async def test_fenced_json_without_lang_tag_is_parsed() -> None:
    """言語タグなしの ``` ... ``` で包まれた応答でも正常にパースされる。"""
    body = (
        '{"candidates":[{"text":"ask","level":"high","rationale":"r"}],'
        '"report":null,"metadata":null}'
    )
    text = f"```\n{body}\n```"
    gen = _make_generator(text)
    pred = await gen.generate(_ctx())
    assert isinstance(pred, IntentPrediction)
    assert len(pred.candidates) == 1
    assert pred.candidates[0].text == "ask"


async def test_fence_only_response_raises_validation_error() -> None:
    """フェンス記号のみの応答は素通しされ、JSON パース失敗として ValidationError。"""
    from pydantic import ValidationError

    gen = _make_generator("```")
    with pytest.raises(ValidationError):
        await gen.generate(_ctx())


def test_strip_code_fence_single_line_variants() -> None:
    """単一行フェンス・言語タグ直後に本文が続く形もフェンスを剥がせる。"""
    from oai_agentspec.runtime.intent._llm import _strip_code_fence

    assert _strip_code_fence('```json {"x":1}```') == '{"x":1}'
    assert _strip_code_fence('```json{"a":1}\n```') == '{"a":1}'
    assert _strip_code_fence('```{"a":1}```') == '{"a":1}'
