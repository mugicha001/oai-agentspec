"""L1: _ml.py (推論側支援関数) の契約 pin。

`confidence_mapper_from_thresholds` (FR-1): 5 段階の閾値境界から float スコアを
`ConfidenceLevel` へ変換する呼び出し可能オブジェクトを構築する。境界包含 (`>=`)・
単調性検証・許容範囲外の error/clamp 挙動を pin する。
`prediction_from_scored_labels` (FR-2): (ラベル, スコア) 列を IntentPrediction へ
変換する。降順整列・allowlist フィルタ・重複集約・max_candidates truncate・
空入力の非例外挙動を pin する。実 SDK / 実 LLM は呼ばない。

まだ `_ml.py` は存在しないため import 自体が失敗する (RED) のが期待挙動。
"""

from __future__ import annotations

import pytest

from oai_agentspec.runtime.intent import (
    ConfidenceLevel,
    IntentCategory,
    IntentPolicy,
    IntentPrediction,
)
from oai_agentspec.runtime.intent._ml import (
    confidence_mapper_from_thresholds,
    prediction_from_scored_labels,
)

pytestmark = pytest.mark.unit


def _policy(max_candidates: int = 3) -> IntentPolicy:
    """テスト用の最小 IntentPolicy を返す (refund / cancel / other・既定 max_candidates=3)。"""
    return IntentPolicy(
        categories=(
            IntentCategory(name="refund", description="返金"),
            IntentCategory(name="cancel", description="キャンセル"),
            IntentCategory(name="other", description="その他"),
        ),
        max_candidates=max_candidates,
    )


def _mapper():
    """設計方針 §9 例1 準拠の既定閾値マッパを返す。"""
    return confidence_mapper_from_thresholds(
        certain=0.90,
        high=0.75,
        medium=0.50,
        low=0.25,
        speculative=0.0,
    )


# ---------------------------------------------------------------------------
# FR-1: confidence_mapper_from_thresholds
# ---------------------------------------------------------------------------


def test_mapper_high_score_maps_to_certain() -> None:
    """certain 下限以上のスコアは CERTAIN に変換される。"""
    mapper = _mapper()
    assert mapper(0.93) is ConfidenceLevel.CERTAIN


def test_mapper_boundary_scores_are_included_in_upper_level() -> None:
    """下限に等しい境界値スコアは上位側の ConfidenceLevel に含める (>= 境界)。"""
    mapper = _mapper()
    assert mapper(0.90) is ConfidenceLevel.CERTAIN
    assert mapper(0.75) is ConfidenceLevel.HIGH
    assert mapper(0.50) is ConfidenceLevel.MEDIUM
    assert mapper(0.25) is ConfidenceLevel.LOW


def test_mapper_low_score_maps_to_speculative() -> None:
    """いずれの下限にも満たないスコアは SPECULATIVE。speculative=0.0 で 0.0 も SPECULATIVE。"""
    mapper = _mapper()
    assert mapper(0.10) is ConfidenceLevel.SPECULATIVE
    assert mapper(0.0) is ConfidenceLevel.SPECULATIVE


def test_mapper_rejects_non_monotonic_thresholds() -> None:
    """閾値境界が単調非増加でない場合は構築時に ValueError。"""
    with pytest.raises(ValueError):
        confidence_mapper_from_thresholds(
            certain=0.5,
            high=0.7,
            medium=0.50,
            low=0.25,
            speculative=0.0,
        )


def test_mapper_out_of_range_score_raises_by_default() -> None:
    """許容範囲 (既定 0.0〜1.0) 外のスコアは既定 on_out_of_range='error' で ValueError。"""
    mapper = _mapper()
    with pytest.raises(ValueError):
        mapper(1.2)
    with pytest.raises(ValueError):
        mapper(-0.5)


def test_mapper_out_of_range_score_clamps_when_requested() -> None:
    """on_out_of_range='clamp' 指定時のみ許容範囲へ clamp する (1.2→CERTAIN, -0.5→SPECULATIVE)。"""
    mapper = confidence_mapper_from_thresholds(
        certain=0.90,
        high=0.75,
        medium=0.50,
        low=0.25,
        speculative=0.0,
        on_out_of_range="clamp",
    )
    assert mapper(1.2) is ConfidenceLevel.CERTAIN
    assert mapper(-0.5) is ConfidenceLevel.SPECULATIVE


def test_mapper_thresholds_are_keyword_only() -> None:
    """閾値引数は keyword-only (位置引数で渡すと TypeError)。"""
    with pytest.raises(TypeError):
        confidence_mapper_from_thresholds(0.90, 0.75, 0.50, 0.25, 0.0)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# FR-2: prediction_from_scored_labels
# ---------------------------------------------------------------------------


def test_prediction_converts_scored_labels_in_descending_order() -> None:
    """(ラベル, スコア) 列を IntentPrediction に変換し、candidates は ConfidenceLevel 降順。"""
    policy = _policy()
    mapper = _mapper()
    prediction = prediction_from_scored_labels(
        [("cancel", 0.60), ("refund", 0.95)],
        policy=policy,
        mapper=mapper,
    )
    assert isinstance(prediction, IntentPrediction)
    texts = [c.text for c in prediction.candidates]
    levels = [c.level for c in prediction.candidates]
    assert texts == ["refund", "cancel"]
    assert levels == [ConfidenceLevel.CERTAIN, ConfidenceLevel.MEDIUM]


def test_prediction_filters_labels_outside_allowlist() -> None:
    """policy.categories の name 集合に無いラベルは除外される。"""
    policy = _policy()
    mapper = _mapper()
    prediction = prediction_from_scored_labels(
        [("refund", 0.95), ("unknown", 0.80)],
        policy=policy,
        mapper=mapper,
    )
    texts = [c.text for c in prediction.candidates]
    assert texts == ["refund"]
    assert "unknown" not in texts


def test_prediction_aggregates_duplicate_labels_to_highest_score() -> None:
    """同一ラベルが複数出現した場合は最高スコアの 1 件のみ採用する。"""
    policy = _policy()
    mapper = _mapper()
    prediction = prediction_from_scored_labels(
        [("refund", 0.30), ("refund", 0.95)],
        policy=policy,
        mapper=mapper,
    )
    refund_candidates = [c for c in prediction.candidates if c.text == "refund"]
    assert len(refund_candidates) == 1
    assert refund_candidates[0].level is ConfidenceLevel.CERTAIN


def test_prediction_truncates_to_max_candidates() -> None:
    """候補数が policy.max_candidates を超える場合、上位から max_candidates 件に truncate。"""
    policy = _policy(max_candidates=2)
    mapper = _mapper()
    prediction = prediction_from_scored_labels(
        [("refund", 0.95), ("cancel", 0.80), ("other", 0.55)],
        policy=policy,
        mapper=mapper,
    )
    texts = [c.text for c in prediction.candidates]
    assert len(prediction.candidates) == 2
    assert texts == ["refund", "cancel"]


def test_prediction_empty_input_returns_empty_prediction() -> None:
    """入力が空の場合、空 candidates を持つ IntentPrediction を返す (例外を送出しない)。"""
    policy = _policy()
    mapper = _mapper()
    prediction = prediction_from_scored_labels([], policy=policy, mapper=mapper)
    assert isinstance(prediction, IntentPrediction)
    assert prediction.candidates == ()


def test_prediction_all_filtered_returns_empty_prediction() -> None:
    """allowlist 適用後に候補が 0 件になる場合も、空 candidates の IntentPrediction を返す。"""
    policy = _policy()
    mapper = _mapper()
    prediction = prediction_from_scored_labels(
        [("unknown", 0.95), ("nope", 0.80)],
        policy=policy,
        mapper=mapper,
    )
    assert isinstance(prediction, IntentPrediction)
    assert prediction.candidates == ()
