"""L1: `_ml_training.py` のチューニング成果物型（`TunedIntentEstimator`）の契約 pin。

`TunedIntentEstimator` は `TrainedIntentEstimator` の frozen サブクラスであり、
CV 探索の副産物（`best_params` / `best_score` / `cv_results`）を keyword-only の
追加フィールドとして保持する。本 L1 は以下を型契約として pin する:

- `tune_ml_estimator` のシグネチャに探索固有の引数が現れない（NFR-1）。
- 親と等価な同一性契約（`compare=False` により `hash` / `==` が親と同一挙動・FR-2b/LSP）。
- frozen 継承・keyword-only の必須フィールド・親フィールドの位置引数束縛の維持。
- 任意副産物の `None` 既定と、`repr` への副産物の露出（診断性）。

各テスト内でのローカル import は、RED 先行フェーズで「未実装なら import 自体が失敗する」
形を取るために導入したものをそのまま維持している（`test_ml_training_l2.py` と同じ作法）。
"""

from __future__ import annotations

import dataclasses
import inspect
from typing import Any

import pytest

from oai_agentspec.runtime.intent.types import IntentCategory, IntentContext, IntentPolicy

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# 共有ヘルパ
# ---------------------------------------------------------------------------


def _inference(context: IntentContext[Any]) -> list[tuple[str, float]]:
    """テスト用の最小推論 callable（固定候補を返す）。"""
    return [("refund", 0.9)]


def _policy() -> IntentPolicy:
    """テスト用の最小 IntentPolicy を返す。"""
    return IntentPolicy(categories=(IntentCategory(name="refund", description="返金"),))


class _AmbiguousTruth:
    """`numpy.ndarray` 相当のスタブ（numpy に依存せず挙動だけを再現する）。

    `__eq__` は bool ではなく自分自身と同型のオブジェクトを返し、その真偽値化は
    `ValueError` を送出する（ndarray の "truth value ... is ambiguous"）。
    ndarray と同じく unhashable でもある。
    """

    def __eq__(self, other: object) -> _AmbiguousTruth:  # type: ignore[override]
        return _AmbiguousTruth()

    def __bool__(self) -> bool:
        raise ValueError("truth value of an array with more than one element is ambiguous")

    __hash__ = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# NFR-1: 探索アルゴリズムを識別する分岐引数がシグネチャに現れない
# ---------------------------------------------------------------------------


def test_tune_ml_estimator_のシグネチャに探索固有の引数が現れない() -> None:
    """パラメータ名集合が固定 6 件と一致し、探索器固有の設定引数を含まない。"""
    from oai_agentspec.runtime.intent._ml_training import tune_ml_estimator

    names = set(inspect.signature(tune_ml_estimator).parameters)

    assert names == {"search", "x_train", "y_train", "policy", "transform", "label_encoding"}
    assert not names & {"param_grid", "cv", "scoring", "n_iter", "refit", "n_jobs"}


# ---------------------------------------------------------------------------
# FR-2b / LSP: 同一性契約が親と等価
# ---------------------------------------------------------------------------


def test_hash_は親と同値で成立する() -> None:
    """副産物は `compare=False` のためハッシュに寄与せず、親と同じ値になる。"""
    from oai_agentspec.runtime.intent._ml_training import (
        TrainedIntentEstimator,
        TunedIntentEstimator,
    )

    policy = _policy()
    parent = TrainedIntentEstimator(inference=_inference, policy=policy)
    tuned = TunedIntentEstimator(
        inference=_inference,
        policy=policy,
        best_params={"clf__C": 10.0},
        best_score=0.83,
        cv_results={"mean_test_score": [0.75, 0.83]},
    )

    # best_params に dict（unhashable）を入れても hash() は TypeError にならない。
    assert hash(tuned) == hash(parent)


def test_ndarray_を含む_cv_results_でも等価比較が例外にならない() -> None:
    """`cv_results` に真偽値化できない値を含んでも `==` は例外を送出せず成立する。"""
    from oai_agentspec.runtime.intent._ml_training import TunedIntentEstimator

    policy = _policy()

    def _build() -> Any:
        # dict の比較は値の同一性で短絡するため、両者に別インスタンスを入れる。
        return TunedIntentEstimator(
            inference=_inference,
            policy=policy,
            best_params={"clf__C": 10.0},
            cv_results={"mean_test_score": _AmbiguousTruth()},
        )

    left = _build()
    right = _build()

    assert left == right


# ---------------------------------------------------------------------------
# dataclass 継承の機構的契約（frozen / kw_only / 位置引数束縛）
# ---------------------------------------------------------------------------


def test_frozen_は子でも維持される() -> None:
    """`TunedIntentEstimator` のフィールドへの再代入は FrozenInstanceError。"""
    from oai_agentspec.runtime.intent._ml_training import TunedIntentEstimator

    tuned = TunedIntentEstimator(inference=_inference, best_params={"clf__C": 1.0})

    with pytest.raises(dataclasses.FrozenInstanceError):
        tuned.best_params = {"clf__C": 2.0}  # type: ignore[misc]


def test_best_params_の省略は_TypeError() -> None:
    """`best_params` は既定値を持たない必須フィールド（FR-2a の fail-fast）。"""
    from oai_agentspec.runtime.intent._ml_training import TunedIntentEstimator

    with pytest.raises(TypeError):
        TunedIntentEstimator(inference=_inference)  # type: ignore[call-arg]


def test_best_params_を位置引数で渡すと_TypeError() -> None:
    """副産物 3 フィールドは kw_only。位置引数では束縛できない。"""
    from oai_agentspec.runtime.intent._ml_training import TunedIntentEstimator

    with pytest.raises(TypeError):
        TunedIntentEstimator(_inference, None, None, None, {"clf__C": 1.0})  # type: ignore[misc]


def test_親フィールドの位置引数束縛が維持される() -> None:
    """親 4 フィールドは位置引数で束縛でき、順序も親と同じ。"""
    from oai_agentspec.runtime.intent._ml_training import TunedIntentEstimator

    estimator = object()
    policy = _policy()

    def decoder(value: Any) -> str:
        return str(value)

    tuned = TunedIntentEstimator(
        _inference, estimator, decoder, policy, best_params={"clf__C": 1.0}
    )

    assert tuned.inference is _inference
    assert tuned.estimator is estimator
    assert tuned.decoder is decoder
    assert tuned.policy is policy


def test_親のサブクラスとして_isinstance_を通過する() -> None:
    """`intent_classifier_from_ml_inference` の isinstance 判定を通る（FR-2b）。"""
    from oai_agentspec.runtime.intent._ml_training import (
        TrainedIntentEstimator,
        TunedIntentEstimator,
    )

    tuned = TunedIntentEstimator(inference=_inference, best_params={"clf__C": 1.0})

    assert isinstance(tuned, TrainedIntentEstimator)


# ---------------------------------------------------------------------------
# FR-2a: 任意副産物の None 既定 / repr への露出
# ---------------------------------------------------------------------------


def test_best_score_と_cv_results_の既定は_None() -> None:
    """任意副産物は省略時に None（欠落を例外にしない）。"""
    from oai_agentspec.runtime.intent._ml_training import TunedIntentEstimator

    tuned = TunedIntentEstimator(inference=_inference, best_params={"clf__C": 1.0})

    assert tuned.best_score is None
    assert tuned.cv_results is None


def test_repr_に副産物が現れる() -> None:
    """`compare=False` は repr に影響しない（診断性を失わない）。"""
    from oai_agentspec.runtime.intent._ml_training import TunedIntentEstimator

    tuned = TunedIntentEstimator(
        inference=_inference,
        best_params={"clf__C": 10.0},
        best_score=0.83,
        cv_results={"mean_test_score": [0.75, 0.83]},
    )

    text = repr(tuned)

    assert "best_params={'clf__C': 10.0}" in text
    assert "best_score=0.83" in text
    assert "cv_results=" in text
