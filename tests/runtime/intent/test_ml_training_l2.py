"""L2: `_ml_training.py` の `ml_inference_from_estimator` (FR-4c) 統合契約 pin。

FR-4c `ml_inference_from_estimator` は学習済み estimator（`predict_proba` /
`classes_` を持つ sklearn 互換オブジェクト）から推論 callable を組み立てる薄い
factory（fit は駆動しない）。本 L2 は以下を統合契約として pin する:

- 構築時の属性検査（`predict_proba` / `classes_` 欠如は TypeError/AttributeError 相当）。
- 既定 transform（`[ctx.utterance]` 単一サンプル列）と custom transform の適用。
- `classes_` とスコアのペア化（順序保存）。
- decoder によるラベル復号（既定は恒等）。
- policy / mapper に相当する引数の keyword-only 契約。
- FR-3 `MLCandidateGenerator` との統合（得た推論 callable がそのまま合成できる）。

まだ `ml_inference_from_estimator` は未実装のため import 自体が失敗する (RED) のが
期待挙動。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from oai_agentspec.runtime.intent import (
    IntentCategory,
    IntentContext,
    IntentPolicy,
)
from oai_agentspec.runtime.intent._ml import (
    MLCandidateGenerator,
    confidence_mapper_from_thresholds,
)
from oai_agentspec.runtime.intent._ml_training import ml_inference_from_estimator

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# 共有ヘルパ
# ---------------------------------------------------------------------------


class FakeEstimator:
    """duck-typed の学習済み estimator。`predict_proba` の入力を記録する。

    `predict_proba` は 1 サンプル分の 2D 配列 `[[p1, p2, ...]]` を返す想定。
    """

    def __init__(self, classes: tuple[Any, ...], proba_rows: list[list[float]]) -> None:
        self.classes_ = classes
        self._proba = proba_rows
        self.received_x: list[Any] = []

    def predict_proba(self, X: Any) -> list[list[float]]:
        self.received_x.append(X)
        return self._proba


def _ctx(utt: str = "返金") -> IntentContext[Any]:
    """テスト用の整形済み IntentContext を返す。"""
    return IntentContext(utterance=utt, history_items=(), run_context=None)


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


def _mapper() -> Callable[[float], Any]:
    """設計方針準拠の既定閾値マッパを返す。"""
    return confidence_mapper_from_thresholds(
        certain=0.90,
        high=0.75,
        medium=0.50,
        low=0.25,
        speculative=0.0,
    )


# ---------------------------------------------------------------------------
# 受け入れ基準 1-2: 構築時の属性検査
# ---------------------------------------------------------------------------


def test_rejects_estimator_without_predict_proba() -> None:
    """`predict_proba` を持たない estimator は構築時に拒否される。"""

    class BadEst:
        classes_ = ("a",)

    with pytest.raises((TypeError, AttributeError)):
        ml_inference_from_estimator(BadEst())


def test_rejects_estimator_without_classes() -> None:
    """`classes_` を持たない estimator は構築時に拒否される。"""

    class BadEst2:
        def predict_proba(self, X: Any) -> list[list[float]]:
            return [[1.0]]

    with pytest.raises((TypeError, AttributeError)):
        ml_inference_from_estimator(BadEst2())


# ---------------------------------------------------------------------------
# 受け入れ基準 3-4: transform の適用
# ---------------------------------------------------------------------------


def test_default_transform_wraps_utterance_as_single_sample() -> None:
    """既定 transform は utterance を 1 要素リストに包んで estimator へ渡す
    （sklearn のサンプル列契約に整合）。"""
    est = FakeEstimator(classes=("refund", "cancel"), proba_rows=[[0.7, 0.3]])
    inference = ml_inference_from_estimator(est)

    inference(_ctx(utt="返金"))

    assert est.received_x == [["返金"]]


def test_custom_transform_is_applied() -> None:
    """custom transform の戻り値がそのまま predict_proba に渡る。"""
    est = FakeEstimator(classes=("refund", "cancel"), proba_rows=[[0.6, 0.4]])
    sentinel = object()
    inference = ml_inference_from_estimator(est, transform=lambda ctx: sentinel)

    inference(_ctx(utt="返金"))

    assert est.received_x == [sentinel]


# ---------------------------------------------------------------------------
# 受け入れ基準 5: classes_ とスコアのペア化（順序保存）
# ---------------------------------------------------------------------------


def test_pairs_classes_with_scores_in_order() -> None:
    """`classes_` の順で (label, score) 列を返す。"""
    est = FakeEstimator(
        classes=("refund", "cancel", "other"),
        proba_rows=[[0.7, 0.2, 0.1]],
    )
    inference = ml_inference_from_estimator(est)

    result = list(inference(_ctx()))

    assert result == [("refund", 0.7), ("cancel", 0.2), ("other", 0.1)]


# ---------------------------------------------------------------------------
# 受け入れ基準 6-7: decoder
# ---------------------------------------------------------------------------


def test_decoder_decodes_labels() -> None:
    """decoder を渡すと `classes_` の各値が復号ラベルに変換される。"""
    est = FakeEstimator(classes=(0, 1), proba_rows=[[0.8, 0.2]])
    decoder = {0: "refund", 1: "cancel"}
    inference = ml_inference_from_estimator(est, decoder=lambda x: decoder[x])

    result = list(inference(_ctx()))

    assert result == [("refund", 0.8), ("cancel", 0.2)]


def test_default_decoder_is_identity() -> None:
    """decoder=None は `classes_` の文字列ラベルをそのまま反映する（恒等）。"""
    est = FakeEstimator(classes=("refund", "cancel"), proba_rows=[[0.9, 0.1]])
    inference = ml_inference_from_estimator(est)

    result = list(inference(_ctx()))

    assert result == [("refund", 0.9), ("cancel", 0.1)]


# ---------------------------------------------------------------------------
# 受け入れ基準 8: keyword-only
# ---------------------------------------------------------------------------


def test_rejects_positional_optional_args() -> None:
    """transform / decoder は keyword-only。位置引数は TypeError。"""
    est = FakeEstimator(classes=("refund",), proba_rows=[[1.0]])

    with pytest.raises(TypeError):
        ml_inference_from_estimator(est, lambda ctx: ctx.utterance, lambda x: x)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 受け入れ基準 9: FR-3 MLCandidateGenerator との統合
# ---------------------------------------------------------------------------


async def test_integrates_with_ml_candidate_generator() -> None:
    """得た推論 callable を `MLCandidateGenerator` に渡すと分類が組み上がる。"""
    est = FakeEstimator(
        classes=("refund", "cancel", "other"),
        proba_rows=[[0.95, 0.03, 0.02]],
    )
    inference = ml_inference_from_estimator(est)
    gen = MLCandidateGenerator(inference, policy=_policy(), mapper=_mapper())

    prediction = await gen.generate(_ctx(utt="返金"))

    assert prediction.candidates
    assert prediction.candidates[0].text == "refund"


# ===========================================================================
# FR-4b `fit_ml_estimator`（T5・build-don't-run 唯一の駆動点）
#
# `fit_ml_estimator` は未実装のため、各テスト内でのローカル import が ImportError で
# 失敗する (RED) のが期待挙動。既存の FR-4c テスト（本ファイル上部の 9 件）は
# モジュール収集を壊さないよう、ローカル import に閉じている。
# ===========================================================================


class FakeFittableEstimator:
    """`fit` を記録する duck-typed の学習可能 estimator。

    `fit` は sklearn 慣行に従い自身を返し、`(X, y)` の参照を `fit_calls` に記録する。
    `predict_proba` は 1 サンプル分の 2D 配列 `[[p1, p2, ...]]` を返す。
    """

    def __init__(self, classes: tuple[Any, ...], proba_rows: list[list[float]]) -> None:
        self.classes_ = classes
        self._proba = proba_rows
        self.fit_calls: list[tuple[Any, Any]] = []

    def fit(self, X: Any, y: Any) -> FakeFittableEstimator:
        self.fit_calls.append((X, y))
        return self

    def predict_proba(self, X: Any) -> list[list[float]]:
        return self._proba


# ---------------------------------------------------------------------------
# 受け入れ基準: fit 呼び出し（エンコードなし = y 素通し）
# ---------------------------------------------------------------------------


def test_fit_calls_estimator_fit_with_raw_y_when_no_encoding() -> None:
    """label_encoding=None は `estimator.fit(x, y)` を y 素通しで 1 回駆動する。"""
    from oai_agentspec.runtime.intent._ml_training import fit_ml_estimator

    est = FakeFittableEstimator(classes=("refund", "cancel"), proba_rows=[[0.7, 0.3]])
    x_train = [[0.0], [1.0], [0.0]]
    y_train = ["refund", "cancel", "refund"]

    fit_ml_estimator(est, x_train=x_train, y_train=y_train, policy=_policy())

    assert len(est.fit_calls) == 1
    called_x, called_y = est.fit_calls[0]
    assert called_x is x_train
    assert called_y is y_train


# ---------------------------------------------------------------------------
# 受け入れ基準: label_encoding による y のエンコード
# ---------------------------------------------------------------------------


def test_fit_encodes_y_with_label_encoding_before_calling_fit() -> None:
    """label_encoding 指定時、fit の 2 引数目は写像で置換した数値列（元の y は不変）。"""
    from oai_agentspec.runtime.intent._ml_training import fit_ml_estimator

    est = FakeFittableEstimator(classes=(0, 1), proba_rows=[[0.6, 0.4]])
    x_train = [[0.0], [1.0], [0.0]]
    y_train = ["refund", "cancel", "refund"]

    fit_ml_estimator(
        est,
        x_train=x_train,
        y_train=y_train,
        policy=_policy(),
        label_encoding={"refund": 0, "cancel": 1},
    )

    called_x, called_y = est.fit_calls[0]
    assert called_x is x_train
    assert list(called_y) == [0, 1, 0]
    # 元の y_train は破壊的変更を受けない。
    assert y_train == ["refund", "cancel", "refund"]


# ---------------------------------------------------------------------------
# 受け入れ基準: 戻り値 TrainedIntentEstimator（同一 estimator 保持）
# ---------------------------------------------------------------------------


def test_fit_returns_trained_intent_estimator_with_original_estimator() -> None:
    """戻り値は `TrainedIntentEstimator` で、`.estimator` は fit 済みの同一オブジェクト。"""
    from oai_agentspec.runtime.intent._ml_training import (
        TrainedIntentEstimator,
        fit_ml_estimator,
    )

    est = FakeFittableEstimator(classes=("refund", "cancel"), proba_rows=[[0.7, 0.3]])

    trained = fit_ml_estimator(
        est,
        x_train=[[0.0]],
        y_train=["refund"],
        policy=_policy(),
    )

    assert isinstance(trained, TrainedIntentEstimator)
    assert trained.estimator is est


# ---------------------------------------------------------------------------
# 受け入れ基準: 推論での復号（エンコードした場合は文字列ラベルへ復号）
# ---------------------------------------------------------------------------


def test_fit_inference_returns_decoded_string_labels() -> None:
    """label_encoding 指定時、推論 callable の戻り値ラベルは復号された文字列。"""
    from oai_agentspec.runtime.intent._ml_training import fit_ml_estimator

    est = FakeFittableEstimator(classes=(0, 1, 2), proba_rows=[[0.7, 0.2, 0.1]])

    trained = fit_ml_estimator(
        est,
        x_train=[[0.0]],
        y_train=["refund", "cancel", "other"],
        policy=_policy(),
        label_encoding={"refund": 0, "cancel": 1, "other": 2},
    )

    result = list(trained.inference(_ctx(utt="返金")))

    assert result == [("refund", 0.7), ("cancel", 0.2), ("other", 0.1)]


def test_fit_inference_without_encoding_is_identity() -> None:
    """label_encoding=None なら decoder=None・classes_ の値がそのまま戻り値ラベル。"""
    from oai_agentspec.runtime.intent._ml_training import fit_ml_estimator

    est = FakeFittableEstimator(
        classes=("refund", "cancel", "other"),
        proba_rows=[[0.9, 0.07, 0.03]],
    )

    trained = fit_ml_estimator(
        est,
        x_train=[[0.0]],
        y_train=["refund"],
        policy=_policy(),
    )

    assert trained.decoder is None
    result = list(trained.inference(_ctx()))
    assert result == [("refund", 0.9), ("cancel", 0.07), ("other", 0.03)]


# ---------------------------------------------------------------------------
# 受け入れ基準: fit 属性検査
# ---------------------------------------------------------------------------


def test_fit_raises_when_estimator_has_no_fit() -> None:
    """`fit` を持たない estimator は構築時に TypeError/AttributeError で拒否される。"""
    from oai_agentspec.runtime.intent._ml_training import fit_ml_estimator

    class NoFitEst:
        classes_ = (0, 1)

        def predict_proba(self, X: Any) -> list[list[float]]:
            return [[0.5, 0.5]]

    with pytest.raises((TypeError, AttributeError)):
        fit_ml_estimator(
            NoFitEst(),
            x_train=[[0.0]],
            y_train=["refund"],
            policy=_policy(),
        )


# ---------------------------------------------------------------------------
# 受け入れ基準: keyword-only
# ---------------------------------------------------------------------------


def test_fit_arguments_are_keyword_only() -> None:
    """x_train 以降は keyword-only。位置引数で渡すと TypeError。"""
    from oai_agentspec.runtime.intent._ml_training import fit_ml_estimator

    est = FakeFittableEstimator(classes=("refund",), proba_rows=[[1.0]])

    with pytest.raises(TypeError):
        fit_ml_estimator(est, [[0.0]], ["refund"])  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 受け入れ基準: FR-3 MLCandidateGenerator との統合
# ---------------------------------------------------------------------------


def test_fit_returns_trained_intent_estimator_with_policy_preserved() -> None:
    """戻り値の `TrainedIntentEstimator.policy` は渡した policy と同一。"""
    from oai_agentspec.runtime.intent._ml_training import fit_ml_estimator

    est = FakeFittableEstimator(classes=("refund", "cancel"), proba_rows=[[0.7, 0.3]])
    policy = _policy()

    trained = fit_ml_estimator(
        est,
        x_train=[[0.0]],
        y_train=["refund"],
        policy=policy,
    )

    assert trained.policy is policy


async def test_fit_result_integrates_with_ml_candidate_generator() -> None:
    """戻り値の inference を `MLCandidateGenerator` に渡すと分類が組み上がる。"""
    from oai_agentspec.runtime.intent._ml_training import fit_ml_estimator

    est = FakeFittableEstimator(classes=(0, 1, 2), proba_rows=[[0.95, 0.03, 0.02]])
    trained = fit_ml_estimator(
        est,
        x_train=[[0.0]],
        y_train=["refund", "cancel", "other"],
        policy=_policy(),
        label_encoding={"refund": 0, "cancel": 1, "other": 2},
    )
    gen = MLCandidateGenerator(trained.inference, policy=_policy(), mapper=_mapper())

    prediction = await gen.generate(_ctx(utt="返金"))

    assert prediction.candidates
    assert prediction.candidates[0].text == "refund"
