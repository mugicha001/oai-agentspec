"""L2: `_ml.py` の `MLCandidateGenerator` の統合契約 pin。

FR-3 `MLCandidateGenerator` は `CandidateGenerator` Protocol を満たす推論ラッパで、
利用側の推論 callable（同期 / 非同期）を呼び、返った (label, score) 列を FR-2
`prediction_from_scored_labels` に渡して `IntentPrediction` を組み立てる。本 L2 は
以下を統合契約として pin する:

- 非同期 / 同期 inference の両対応（await した inference 結果が prediction に反映）。
- 同期 inference はイベントループをブロックしない（別スレッド実行）。
- inference 例外は握り潰さず `generate()` から伝播する（同期 / 非同期の両方）。
- `CandidateGenerator` Protocol 適合（@runtime_checkable）。
- 既存 factory `intent_classifier_from_generator` との統合で分類器が組み上がる。
- policy / mapper が keyword-only である契約。

まだ `MLCandidateGenerator` は未実装のため import 自体が失敗する (RED) のが期待挙動。
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any

import pytest

from oai_agentspec.runtime.intent import (
    CandidateGenerator,
    ConfidenceLevel,
    IntentCategory,
    IntentContext,
    IntentPolicy,
    IntentPrediction,
    IntentQuery,
    intent_classifier_from_generator,
)
from oai_agentspec.runtime.intent._ml import (
    MLCandidateGenerator,
    confidence_mapper_from_thresholds,
)

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# 共有ヘルパ
# ---------------------------------------------------------------------------


def _policy(max_candidates: int = 3) -> IntentPolicy:
    """テスト用の最小 IntentPolicy を返す (refund / cancel / other)。"""
    return IntentPolicy(
        categories=(
            IntentCategory(name="refund", description="返金"),
            IntentCategory(name="cancel", description="解約"),
            IntentCategory(name="other", description="その他"),
        ),
        max_candidates=max_candidates,
    )


def _mapper() -> Callable[[float], ConfidenceLevel]:
    """設計方針準拠の既定閾値マッパを返す。"""
    return confidence_mapper_from_thresholds(
        certain=0.90,
        high=0.75,
        medium=0.50,
        low=0.25,
        speculative=0.0,
    )


def _ctx(utt: str = "返金してほしい") -> IntentContext[Any]:
    """テスト用の整形済み IntentContext を返す。"""
    return IntentContext(utterance=utt, history_items=(), run_context=None)


# ---------------------------------------------------------------------------
# FR-3: MLCandidateGenerator（非同期 inference）
# ---------------------------------------------------------------------------


async def test_async_inference_produces_prediction() -> None:
    """非同期 inference の戻り値が FR-2 統合パスを通り prediction に反映される。"""

    async def infer(ctx: IntentContext[Any]) -> list[tuple[str, float]]:
        return [("refund", 0.93)]

    gen = MLCandidateGenerator(infer, policy=_policy(), mapper=_mapper())
    prediction = await gen.generate(_ctx())

    assert isinstance(prediction, IntentPrediction)
    texts = [c.text for c in prediction.candidates]
    assert "refund" in texts
    refund = next(c for c in prediction.candidates if c.text == "refund")
    assert refund.level is ConfidenceLevel.CERTAIN


async def test_async_inference_receives_the_context() -> None:
    """非同期 inference は generate に渡した context をそのまま受け取る。"""
    seen: list[IntentContext[Any]] = []

    async def infer(ctx: IntentContext[Any]) -> list[tuple[str, float]]:
        seen.append(ctx)
        return [("refund", 0.93)]

    gen = MLCandidateGenerator(infer, policy=_policy(), mapper=_mapper())
    ctx = _ctx("解約したい")
    await gen.generate(ctx)

    assert len(seen) == 1
    assert seen[0] is ctx


# ---------------------------------------------------------------------------
# FR-3: MLCandidateGenerator（同期 inference）
# ---------------------------------------------------------------------------


async def test_sync_inference_does_not_block_event_loop() -> None:
    """同期 inference はイベントループをブロックしない (別スレッド実行)。

    generate() の実行中も並行 asyncio タスクが進行し tick を刻めることを確認する。
    もし同期 inference をインラインで呼びブロックしていれば tick は進まない。
    """
    sleep_s = 0.1

    def sync_infer(ctx: IntentContext[Any]) -> list[tuple[str, float]]:
        time.sleep(sleep_s)
        return [("refund", 0.93)]

    gen = MLCandidateGenerator(sync_infer, policy=_policy(), mapper=_mapper())

    done = asyncio.Event()
    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while not done.is_set():
            ticks += 1
            await asyncio.sleep(0.01)

    async def run_gen() -> IntentPrediction:
        try:
            return await gen.generate(_ctx())
        finally:
            done.set()

    prediction, _ = await asyncio.gather(run_gen(), ticker())

    # ブロックしていなければ sleep_s の間に複数回 tick できる (0.1/0.01=10 回目安)。
    assert ticks >= 3
    assert isinstance(prediction, IntentPrediction)


async def test_sync_inference_return_value_reflected_in_prediction() -> None:
    """同期 inference の戻り値がそのまま prediction に反映される。"""

    def sync_infer(ctx: IntentContext[Any]) -> list[tuple[str, float]]:
        return [("refund", 0.95), ("cancel", 0.60)]

    gen = MLCandidateGenerator(sync_infer, policy=_policy(), mapper=_mapper())
    prediction = await gen.generate(_ctx())

    texts = [c.text for c in prediction.candidates]
    levels = [c.level for c in prediction.candidates]
    assert texts == ["refund", "cancel"]
    assert levels == [ConfidenceLevel.CERTAIN, ConfidenceLevel.MEDIUM]


# ---------------------------------------------------------------------------
# FR-3: 例外伝播（握り潰さない）
# ---------------------------------------------------------------------------


async def test_sync_inference_exception_propagates() -> None:
    """同期 inference の送出例外は generate() から伝播する (握り潰さない)。"""

    def sync_infer(ctx: IntentContext[Any]) -> list[tuple[str, float]]:
        raise RuntimeError("boom")

    gen = MLCandidateGenerator(sync_infer, policy=_policy(), mapper=_mapper())
    with pytest.raises(RuntimeError, match="boom"):
        await gen.generate(_ctx())


async def test_async_inference_exception_propagates() -> None:
    """非同期 inference の送出例外は generate() から伝播する (握り潰さない)。"""

    async def infer(ctx: IntentContext[Any]) -> list[tuple[str, float]]:
        raise RuntimeError("boom")

    gen = MLCandidateGenerator(infer, policy=_policy(), mapper=_mapper())
    with pytest.raises(RuntimeError, match="boom"):
        await gen.generate(_ctx())


# ---------------------------------------------------------------------------
# FR-3: Protocol 適合 / factory 統合 / keyword-only 契約
# ---------------------------------------------------------------------------


async def test_satisfies_candidate_generator_protocol() -> None:
    """MLCandidateGenerator は CandidateGenerator Protocol (@runtime_checkable) を満たす。"""

    async def infer(ctx: IntentContext[Any]) -> list[tuple[str, float]]:
        return [("refund", 0.93)]

    gen = MLCandidateGenerator(infer, policy=_policy(), mapper=_mapper())
    assert isinstance(gen, CandidateGenerator)


async def test_integrates_with_intent_classifier_from_generator() -> None:
    """既存 factory 経由で DefaultIntentClassifier が組み上がり classify が動く。"""

    def sync_infer(ctx: IntentContext[Any]) -> list[tuple[str, float]]:
        return [("refund", 0.93)]

    gen = MLCandidateGenerator(sync_infer, policy=_policy(), mapper=_mapper())
    clf = intent_classifier_from_generator(gen)
    prediction = await clf.classify(IntentQuery(utterance="返金して"))

    assert isinstance(prediction, IntentPrediction)
    texts = [c.text for c in prediction.candidates]
    assert "refund" in texts


def test_policy_and_mapper_are_keyword_only() -> None:
    """policy / mapper は keyword-only (位置引数で渡すと TypeError)。"""

    async def infer(ctx: IntentContext[Any]) -> list[tuple[str, float]]:
        return [("refund", 0.93)]

    with pytest.raises(TypeError):
        MLCandidateGenerator(infer, _policy(), _mapper())  # type: ignore[misc]


# ---------------------------------------------------------------------------
# FR-3: MLCandidateGenerator（awaitable を返す callable の汎用対応）
# ---------------------------------------------------------------------------


async def test_async_callable_object_is_awaited() -> None:
    """async `__call__` を持つ callable オブジェクトも非同期として正しく await される。

    `inspect.iscoroutinefunction` は callable オブジェクトの async `__call__` を
    非同期関数として判定できないため、同期パスに落ちて `to_thread` で呼ばれた
    結果が coroutine のまま `prediction_from_scored_labels` に渡ってしまうバグを pin する
    (Codex review P2 指摘)。実装は呼び出し後に `inspect.isawaitable(result)` を確認して
    await する必要がある。
    """

    class AsyncCallable:
        async def __call__(self, ctx: IntentContext[Any]) -> list[tuple[str, float]]:
            return [("refund", 0.93)]

    gen = MLCandidateGenerator(AsyncCallable(), policy=_policy(), mapper=_mapper())
    prediction = await gen.generate(_ctx())

    assert isinstance(prediction, IntentPrediction)
    texts = [c.text for c in prediction.candidates]
    assert "refund" in texts


async def test_sync_callable_returning_coroutine_is_awaited() -> None:
    """同期関数が coroutine を返す場合も戻り値が await される。

    型注釈 `Sequence[...] | Awaitable[Sequence[...]]` はこのパターンも許容するため、
    to_thread で呼んだ戻り値が awaitable なら await するのが契約。
    """

    async def _delegate(ctx: IntentContext[Any]) -> list[tuple[str, float]]:
        return [("cancel", 0.80)]

    def wrapper(ctx: IntentContext[Any]) -> Any:
        return _delegate(ctx)

    gen = MLCandidateGenerator(wrapper, policy=_policy(), mapper=_mapper())
    prediction = await gen.generate(_ctx())

    assert isinstance(prediction, IntentPrediction)
    texts = [c.text for c in prediction.candidates]
    assert "cancel" in texts
