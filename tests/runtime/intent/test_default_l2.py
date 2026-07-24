"""L2: `runtime.intent._default` の DefaultContextBuilder / DefaultIntentClassifier 挙動網羅。

ContextBuilder が IntentQuery.history から get_items(limit=...) を呼び、取得した
history アイテムを加工せず tuple 化して pass-through すること、history=None を空
tuple で扱うこと、DefaultIntentClassifier が context_builder → generator を順に
呼び結果を素通しすることを検証する。実 SDK / 実 LLM は使わず Fake で検証する。
"""

from __future__ import annotations

from typing import Any

import pytest

from oai_agentspec.runtime.intent._default import (
    DefaultContextBuilder,
    DefaultIntentClassifier,
)
from oai_agentspec.runtime.intent.types import (
    ConfidenceLevel,
    IntentCandidate,
    IntentContext,
    IntentPrediction,
    IntentQuery,
)

pytestmark = pytest.mark.integration


class _FakeHistory:
    """agents.Session 互換の最小 Fake。get_items(limit=...) の呼び出し引数を記録する。"""

    def __init__(self, items: list[dict[str, Any]] | None) -> None:
        self._items = items
        self.limit_calls: list[int | None] = []

    async def get_items(self, limit: int | None = None) -> list[dict[str, Any]] | None:
        self.limit_calls.append(limit)
        return self._items


class _FakeContextBuilder:
    """ContextBuilder Protocol の Fake。build() の呼び出しを記録し固定 ctx を返す。"""

    def __init__(self, ctx: IntentContext[Any]) -> None:
        self._ctx = ctx
        self.calls: list[IntentQuery[Any]] = []
        self.trace: list[str] = []

    async def build(self, query: IntentQuery[Any]) -> IntentContext[Any]:
        self.calls.append(query)
        self.trace.append("build")
        return self._ctx


class _FakeCandidateGenerator:
    """CandidateGenerator Protocol の Fake。generate() の呼び出しを記録し固定予測を返す。"""

    def __init__(self, prediction: IntentPrediction) -> None:
        self._prediction = prediction
        self.calls: list[IntentContext[Any]] = []
        self.trace: list[str] = []

    async def generate(self, context: IntentContext[Any]) -> IntentPrediction:
        self.calls.append(context)
        self.trace.append("generate")
        return self._prediction


def _make_prediction() -> IntentPrediction:
    """テスト用の固定 IntentPrediction。"""
    return IntentPrediction(
        candidates=(IntentCandidate(text="ask", level=ConfidenceLevel.HIGH, rationale="r"),),
    )


# ---- DefaultContextBuilder ----


async def test_history_limit_default_20() -> None:
    """DefaultContextBuilder() のデフォルト history_limit は 20。"""
    b = DefaultContextBuilder()
    assert b.history_limit == 20


async def test_custom_history_limit() -> None:
    """history_limit=5 を渡すとその値が保持される。"""
    b = DefaultContextBuilder(history_limit=5)
    assert b.history_limit == 5


def test_default_context_builder_rejects_zero_history_limit() -> None:
    """history_limit=0 は ValueError（メッセージに history_limit を含む）。"""
    with pytest.raises(ValueError, match="history_limit") as exc_info:
        DefaultContextBuilder(history_limit=0)
    assert "history_limit" in str(exc_info.value)


def test_default_context_builder_rejects_negative_history_limit() -> None:
    """history_limit=-1 は ValueError。"""
    with pytest.raises(ValueError, match="history_limit"):
        DefaultContextBuilder(history_limit=-1)


def test_default_context_builder_accepts_history_limit_one() -> None:
    """history_limit=1 は下限値として正常に受け入れられる。"""
    b = DefaultContextBuilder(history_limit=1)
    assert b.history_limit == 1


async def test_history_none_yields_empty_history_items() -> None:
    """history=None のとき history_items は空 tuple。"""
    b = DefaultContextBuilder()
    ctx = await b.build(IntentQuery(utterance="hi", history=None))
    assert ctx.utterance == "hi"
    assert ctx.history_items == ()
    assert ctx.run_context is None


async def test_history_items_pass_through_from_session() -> None:
    """history から取得したアイテムがそのまま history_items に格納される。"""
    items = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "bye"},
    ]
    hist = _FakeHistory(items)
    b = DefaultContextBuilder()
    ctx = await b.build(IntentQuery(utterance="q", history=hist))
    assert ctx.history_items == tuple(items)


async def test_history_get_items_called_with_history_limit() -> None:
    """history_limit=5 のとき get_items(limit=5) として呼ばれる。"""
    hist = _FakeHistory([])
    b = DefaultContextBuilder(history_limit=5)
    await b.build(IntentQuery(utterance="hi", history=hist))
    assert hist.limit_calls == [5]


async def test_history_get_items_called_with_default_limit() -> None:
    """history_limit を指定しない場合、get_items(limit=20) として呼ばれる。"""
    hist = _FakeHistory([])
    b = DefaultContextBuilder()
    await b.build(IntentQuery(utterance="hi", history=hist))
    assert hist.limit_calls == [20]


async def test_history_items_are_tuple_type() -> None:
    """history_items の型は tuple そのもの。"""
    hist = _FakeHistory([{"role": "user", "content": "hello"}])
    b = DefaultContextBuilder()
    ctx = await b.build(IntentQuery(utterance="q", history=hist))
    assert type(ctx.history_items) is tuple


async def test_utterance_and_run_context_pass_through() -> None:
    """utterance / run_context は IntentContext にそのまま素通しされる。"""
    b = DefaultContextBuilder()
    run_ctx = {"k": "v"}
    ctx = await b.build(IntentQuery(utterance="hi", run_context=run_ctx))
    assert ctx.utterance == "hi"
    assert ctx.run_context == run_ctx


async def test_history_get_items_returning_none_yields_empty_tuple() -> None:
    """get_items が None を返した場合、history_items は空 tuple。"""
    hist = _FakeHistory(None)
    b = DefaultContextBuilder()
    ctx = await b.build(IntentQuery(utterance="q", history=hist))
    assert ctx.history_items == ()


async def test_history_items_are_not_transformed() -> None:
    """history アイテムは加工されず、渡した dict の内容がそのまま出てくる。"""
    items = [{"role": "user", "content": [{"type": "text", "text": "part1"}]}]
    hist = _FakeHistory(items)
    b = DefaultContextBuilder()
    ctx = await b.build(IntentQuery(utterance="q", history=hist))
    assert ctx.history_items[0] == items[0]


async def test_build_returns_exactly_intent_context_type() -> None:
    """戻り値の型は IntentContext そのもの（サブクラス等を返さない）。"""
    b = DefaultContextBuilder()
    ctx = await b.build(IntentQuery(utterance="hi"))
    assert type(ctx) is IntentContext


async def test_history_only_query_builds_context_with_empty_utterance() -> None:
    """utterance 省略 + history のみの IntentQuery から IntentContext を構築できる（Issue #24）。

    utterance は空文字のまま素通しされ、履歴アイテムが history_items に格納される。
    """
    items = [{"role": "user", "content": "過去発話"}]
    hist = _FakeHistory(items)
    b = DefaultContextBuilder()
    ctx = await b.build(IntentQuery(history=hist))
    assert ctx.utterance == ""
    assert ctx.history_items == tuple(items)


# ---- DefaultIntentClassifier ----


async def test_classifier_stores_dependencies() -> None:
    """コンストラクタが context_builder / generator を保持する。"""
    cb = _FakeContextBuilder(IntentContext(utterance="hi"))
    gen = _FakeCandidateGenerator(_make_prediction())
    clf = DefaultIntentClassifier(context_builder=cb, generator=gen)
    assert clf.context_builder is cb
    assert clf.generator is gen


async def test_classify_calls_context_builder_build_with_query() -> None:
    """classify(query) が context_builder.build(query) を呼ぶ。"""
    fixed_ctx = IntentContext(utterance="hi")
    cb = _FakeContextBuilder(fixed_ctx)
    gen = _FakeCandidateGenerator(_make_prediction())
    clf = DefaultIntentClassifier(context_builder=cb, generator=gen)
    query = IntentQuery(utterance="hi")
    await clf.classify(query)
    assert cb.calls == [query]


async def test_classify_passes_built_context_to_generator() -> None:
    """context_builder.build の戻り値がそのまま generator.generate に渡る。"""
    fixed_ctx = IntentContext(
        utterance="hello", history_items=({"role": "user", "content": "prev"},)
    )
    cb = _FakeContextBuilder(fixed_ctx)
    gen = _FakeCandidateGenerator(_make_prediction())
    clf = DefaultIntentClassifier(context_builder=cb, generator=gen)
    await clf.classify(IntentQuery(utterance="hello"))
    assert gen.calls == [fixed_ctx]


async def test_classify_returns_generator_prediction() -> None:
    """最終戻り値は generator.generate の戻り値。"""
    prediction = _make_prediction()
    cb = _FakeContextBuilder(IntentContext(utterance="hi"))
    gen = _FakeCandidateGenerator(prediction)
    clf = DefaultIntentClassifier(context_builder=cb, generator=gen)
    result = await clf.classify(IntentQuery(utterance="hi"))
    assert result is prediction


async def test_classify_calls_build_before_generate() -> None:
    """呼び出し順序は build → generate（duck-typed 実装でも動く）。"""
    ctx_holder = IntentContext(utterance="hi")
    trace: list[str] = []

    class _DuckBuilder:
        async def build(self, query: IntentQuery[Any]) -> IntentContext[Any]:
            trace.append("build")
            return ctx_holder

    class _DuckGenerator:
        async def generate(self, context: IntentContext[Any]) -> IntentPrediction:
            trace.append("generate")
            return _make_prediction()

    clf = DefaultIntentClassifier(context_builder=_DuckBuilder(), generator=_DuckGenerator())
    await clf.classify(IntentQuery(utterance="hi"))
    assert trace == ["build", "generate"]


async def test_classify_empty_query_raises_value_error_end_to_end() -> None:
    """utterance と history の両方が空の IntentQuery は classify 経由で ValueError。

    adapter 単体の fail-fast pin とは別に、利用者が実際に踏む公開経路
    （classify 起点・実 adapter 経由）での伝播を固定する。モデルは呼ばれない。
    """
    from oai_agentspec.runtime.intent._llm import LLMCandidateGenerator
    from oai_agentspec.runtime.intent.types import IntentCategory, IntentPolicy

    from _helpers.intent_fakes import RecordingFakeModel

    model = RecordingFakeModel()
    clf = DefaultIntentClassifier(
        context_builder=DefaultContextBuilder(),
        generator=LLMCandidateGenerator(
            model,
            lambda ctx: ctx.utterance,
            policy=IntentPolicy(categories=(IntentCategory(name="a", description="d"),)),
        ),
    )

    with pytest.raises(ValueError, match="utterance"):
        await clf.classify(IntentQuery())

    assert model.calls == []
