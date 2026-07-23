"""L1: _ml_training.py (FR-4a 最小学習契約) の契約 pin。

`TrainedIntentEstimator` (frozen dataclass) の frozen 性・デフォルト値、
`make_trained_estimator` builder の等価な組み立て・keyword-only 強制、
`IntentTrainer` 型エイリアスの import 可能性を pin する。
"""

from __future__ import annotations

import dataclasses

import pytest

from oai_agentspec.runtime.intent._ml_training import (
    IntentTrainer,
    TrainedIntentEstimator,
    make_trained_estimator,
)
from oai_agentspec.runtime.intent.types import IntentCategory, IntentPolicy

pytestmark = pytest.mark.unit


def _inference(context: object) -> list[tuple[str, float]]:
    """テスト用の最小推論 callable（固定候補を返す）。"""
    return [("refund", 0.9)]


def _policy() -> IntentPolicy:
    """テスト用の最小 IntentPolicy を返す。"""
    return IntentPolicy(categories=(IntentCategory(name="refund", description="返金"),))


def test_trained_intent_estimator_is_frozen() -> None:
    """`TrainedIntentEstimator` は frozen dataclass で属性再代入が拒否される。"""
    est = TrainedIntentEstimator(inference=_inference)

    with pytest.raises(dataclasses.FrozenInstanceError):
        est.inference = _inference  # type: ignore[misc]


def test_trained_intent_estimator_defaults() -> None:
    """`estimator` / `decoder` / `policy` の既定値は None。"""
    est = TrainedIntentEstimator(inference=_inference)

    assert est.estimator is None
    assert est.decoder is None
    assert est.policy is None


def test_make_trained_estimator_returns_equivalent_instance() -> None:
    """`make_trained_estimator(inference=...)` は同等の `TrainedIntentEstimator` を返す。"""
    est = make_trained_estimator(inference=_inference)

    assert est == TrainedIntentEstimator(inference=_inference)
    assert isinstance(est, TrainedIntentEstimator)


def test_make_trained_estimator_preserves_policy() -> None:
    """`make_trained_estimator(policy=...)` は渡した policy を同一性を保って保持する。"""
    policy = _policy()

    est = make_trained_estimator(inference=_inference, policy=policy)

    assert est.policy is policy


def test_make_trained_estimator_rejects_positional_args() -> None:
    """`make_trained_estimator` は keyword-only のため位置引数は TypeError。"""
    with pytest.raises(TypeError):
        make_trained_estimator(_inference)  # type: ignore[misc]


def test_intent_trainer_type_alias_is_importable() -> None:
    """`IntentTrainer` 型エイリアスが公開され import できる。"""
    assert IntentTrainer is not None
