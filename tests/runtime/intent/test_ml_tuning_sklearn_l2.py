"""L2: 実 scikit-learn を用いた `tune_ml_estimator` の end-to-end 検証（FR-5 後半）。

`GridSearchCV` + `TfidfVectorizer` + `LogisticRegression` の小規模 `Pipeline` を用い、
`tune_ml_estimator` -> `intent_classifier_from_ml_inference` -> `classify` まで実 sklearn
を通して検証する。fake 探索器（`test_ml_tuning_l2.py`）が担う契約全体の検証とは別に、
実装詳細に依存しないことの裏取りとして 1 本だけ持つ（ADR 0039 6.2）。

`scikit-learn` は開発依存グループ（`[dependency-groups].dev`）としてのみ導入され、lib 本体・
配布物の依存には含めない。
"""

from __future__ import annotations

import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from oai_agentspec.runtime.intent._ml_training import tune_ml_estimator
from oai_agentspec.runtime.intent.factories import intent_classifier_from_ml_inference
from oai_agentspec.runtime.intent.types import (
    IntentCategory,
    IntentContext,
    IntentPolicy,
    IntentQuery,
)

pytestmark = pytest.mark.integration

_X_TRAIN = [
    "返金してほしい",
    "返金をお願いします",
    "お金を返してください",
    "解約したい",
    "解約の手続きを教えてください",
    "契約をやめたい",
]
_Y_TRAIN = ["refund", "refund", "refund", "cancel", "cancel", "cancel"]

_PARAM_GRID = {"clf__C": [1.0, 10.0]}


def _policy() -> IntentPolicy:
    """返金 / 解約の 2 カテゴリからなる最小 `IntentPolicy` を返す。"""
    return IntentPolicy(
        categories=(
            IntentCategory(name="refund", description="返金"),
            IntentCategory(name="cancel", description="解約"),
        ),
        max_candidates=2,
    )


def _thresholds() -> dict[str, float]:
    """`intent_classifier_from_ml_inference` へ渡す閾値マッピング。"""
    return {"certain": 0.90, "high": 0.75, "medium": 0.50, "low": 0.25, "speculative": 0.0}


async def test_gridsearchcv_で調整した分類器が意図を分類できる() -> None:
    """実 `GridSearchCV` の探索結果から組んだ分類器が `classify` まで動く。"""
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3))),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    search = GridSearchCV(pipeline, param_grid=_PARAM_GRID, cv=2)

    tuned = tune_ml_estimator(search, x_train=_X_TRAIN, y_train=_Y_TRAIN, policy=_policy())

    assert tuned.best_params in ({"clf__C": 1.0}, {"clf__C": 10.0})
    assert isinstance(tuned.best_score, float)

    # 推論器の生スコアで pin する。`classes_` は y_train の出現順ではなくアルファベット順
    # （`['cancel' 'refund']`）になるため、ラベルと `predict_proba` の列対応が逆転すると
    # ここで落ちる。候補列側の `in {...}` 照合は policy の allowlist が同じ 2 件のため
    # ほぼ恒真で、この対応ズレを捕捉できない。
    context = IntentContext(utterance="返金してほしいです", history_items=(), run_context=None)
    scored = dict(tuned.inference(context))

    assert set(scored) == {"refund", "cancel"}
    assert scored["refund"] > scored["cancel"]

    clf = intent_classifier_from_ml_inference(tuned, thresholds=_thresholds())
    prediction = await clf.classify(IntentQuery(utterance="返金してほしいです"))

    assert prediction.candidates
    assert prediction.candidates[0].text == "refund"
