"""L1: `runtime.intent.factories.intent_classifier_from_ml_inference` の組み立て契約。

`intent_classifier_from_model` の対称形として、mapper / thresholds の排他検証、
`TrainedIntentEstimator` 直渡し対応、`DefaultIntentClassifier`
(`DefaultContextBuilder` + `MLCandidateGenerator`) の組み立てをピン留めする。
実 sklearn / 実 LLM は呼ばない。
"""

from __future__ import annotations

from typing import Any

import pytest

from oai_agentspec.runtime.intent._default import (
    DefaultContextBuilder,
    DefaultIntentClassifier,
)
from oai_agentspec.runtime.intent._ml import MLCandidateGenerator
from oai_agentspec.runtime.intent._ml_training import TrainedIntentEstimator
from oai_agentspec.runtime.intent.factories import intent_classifier_from_ml_inference
from oai_agentspec.runtime.intent.types import (
    ConfidenceLevel,
    IntentCategory,
    IntentContext,
    IntentPolicy,
    IntentQuery,
)

pytestmark = pytest.mark.unit


def _policy() -> IntentPolicy:
    """テスト用の最小 IntentPolicy を返す。"""
    return IntentPolicy(
        categories=(
            IntentCategory(name="refund", description="返金"),
            IntentCategory(name="chitchat", description="雑談"),
        ),
    )


def _mapper(score: float) -> ConfidenceLevel:
    """テスト用の固定 mapper（常に CERTAIN を返す）。"""
    return ConfidenceLevel.CERTAIN


def _thresholds() -> dict[str, float]:
    """confidence_mapper_from_thresholds 展開用のテスト閾値。"""
    return {
        "certain": 0.9,
        "high": 0.7,
        "medium": 0.5,
        "low": 0.3,
        "speculative": 0.0,
    }


def _inference(context: IntentContext[Any]) -> list[tuple[str, float]]:
    """テスト用の同期推論 callable。"""
    return [("refund", 0.95)]


def test_from_ml_inference_with_mapper_builds_classifier() -> None:
    """callable + mapper で DefaultIntentClassifier が返る。"""
    clf = intent_classifier_from_ml_inference(_inference, policy=_policy(), mapper=_mapper)
    assert isinstance(clf, DefaultIntentClassifier)


def test_from_ml_inference_with_thresholds_expands_mapper() -> None:
    """thresholds を渡すと内部で mapper が構築され動作する。"""
    clf = intent_classifier_from_ml_inference(
        _inference, policy=_policy(), thresholds=_thresholds()
    )
    assert isinstance(clf, DefaultIntentClassifier)
    assert isinstance(clf.generator, MLCandidateGenerator)


def test_from_ml_inference_raises_when_both_mapper_and_thresholds() -> None:
    """mapper と thresholds を両方指定すると ValueError。"""
    with pytest.raises(ValueError, match="同時指定"):
        intent_classifier_from_ml_inference(
            _inference, policy=_policy(), mapper=_mapper, thresholds=_thresholds()
        )


def test_from_ml_inference_raises_when_neither_mapper_nor_thresholds() -> None:
    """mapper も thresholds も指定しないと ValueError。"""
    with pytest.raises(ValueError, match="いずれかを指定"):
        intent_classifier_from_ml_inference(_inference, policy=_policy())


def test_from_ml_inference_accepts_trained_intent_estimator_directly() -> None:
    """TrainedIntentEstimator を第 1 引数に渡すと内部で .inference を取り出す。"""
    trained = TrainedIntentEstimator(inference=_inference)
    clf = intent_classifier_from_ml_inference(trained, policy=_policy(), mapper=_mapper)
    assert isinstance(clf.generator, MLCandidateGenerator)
    assert clf.generator._inference is _inference


def test_from_ml_inference_history_limit_defaults_to_20() -> None:
    """history_limit のデフォルトは 20。"""
    clf = intent_classifier_from_ml_inference(_inference, policy=_policy(), mapper=_mapper)
    assert isinstance(clf.context_builder, DefaultContextBuilder)
    assert clf.context_builder.history_limit == 20


def test_from_ml_inference_history_limit_customizable() -> None:
    """history_limit=5 を渡すと反映される。"""
    clf = intent_classifier_from_ml_inference(
        _inference, policy=_policy(), mapper=_mapper, history_limit=5
    )
    assert isinstance(clf.context_builder, DefaultContextBuilder)
    assert clf.context_builder.history_limit == 5


def test_from_ml_inference_policy_is_keyword_only() -> None:
    """policy は keyword-only（位置引数で渡すと TypeError）。"""
    with pytest.raises(TypeError):
        intent_classifier_from_ml_inference(_inference, _policy(), mapper=_mapper)  # type: ignore[misc]


def test_from_ml_inference_uses_ml_candidate_generator_internally() -> None:
    """generator は MLCandidateGenerator インスタンスになる。"""
    clf = intent_classifier_from_ml_inference(_inference, policy=_policy(), mapper=_mapper)
    assert isinstance(clf.generator, MLCandidateGenerator)


async def test_from_ml_inference_integrates_with_classify() -> None:
    """classify() を実際に呼ぶと IntentPrediction 相当の候補が返る。"""
    clf = intent_classifier_from_ml_inference(
        _inference, policy=_policy(), thresholds=_thresholds()
    )
    prediction = await clf.classify(IntentQuery(utterance="返金してほしい"))
    assert len(prediction.candidates) == 1
    assert prediction.candidates[0].text == "refund"


async def test_from_ml_inference_resolves_policy_from_trained_estimator() -> None:
    """policy 入りの `TrainedIntentEstimator` を渡すと FR-5 の policy 省略時でも
    自動解決され classify が動く（FR-5 自動解決）。"""
    trained = TrainedIntentEstimator(inference=_inference, policy=_policy())

    clf = intent_classifier_from_ml_inference(trained, mapper=_mapper)

    prediction = await clf.classify(IntentQuery(utterance="返金してほしい"))
    assert len(prediction.candidates) == 1
    assert prediction.candidates[0].text == "refund"


def test_from_ml_inference_explicit_policy_overrides_estimator_policy() -> None:
    """FR-5 の policy を明示指定すると、成果物側の policy より優先される。"""
    estimator_policy = _policy()
    explicit_policy = _policy()
    trained = TrainedIntentEstimator(inference=_inference, policy=estimator_policy)

    clf = intent_classifier_from_ml_inference(trained, policy=explicit_policy, mapper=_mapper)

    assert clf.generator._policy is explicit_policy


def test_from_ml_inference_raises_when_no_policy_available() -> None:
    """callable 直渡し + policy 省略の場合、policy を解決できず ValueError。"""
    with pytest.raises(ValueError):
        intent_classifier_from_ml_inference(_inference, mapper=_mapper)
