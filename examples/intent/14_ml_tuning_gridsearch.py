"""CV 探索器（`GridSearchCV`）を `tune_ml_estimator` に渡すハイパーパラメータ探索の例。

08 は単一の estimator を `fit_ml_estimator`（FR-4b）へ渡すゼロコード fit だったが、本例は
`TfidfVectorizer` + `LogisticRegression` の `Pipeline` を包む `GridSearchCV` を
`tune_ml_estimator` へ渡し、探索・交差検証・`best_estimator_` の決定まで含めて 1 呼び出しで
完結させる。探索・交差検証（`scoring` / `cv` / `param_grid`）はすべて探索器インスタンス側の
設定であり、`tune_ml_estimator` のシグネチャには一切現れない。本例は評価指標に `f1_macro` を
明示指定してこの境界を示す（未指定なら estimator の `score()` = 分類器では accuracy）。
`best_score` / `cv_results` の値が何の指標かは探索器の `scoring` が決め、lib は解釈も変換も
せずそのまま成果物へ載せる。lib は探索アルゴリズムを持たず `search.fit()` を 1 回駆動する
だけの薄い結線（ADR 0004 / ADR 0039）。

推論器・成果物の `estimator` は探索器そのものではなく fit 後の `best_estimator_` に束縛される。
`refit` を無効にした探索器（sklearn の `refit=False` 等）は `best_estimator_` を生成しないため
本経路では使えず、明示的な `AttributeError` になる。

返る `TunedIntentEstimator` は `TrainedIntentEstimator` のサブクラスであり、
`intent_classifier_from_ml_inference`（FR-5）へそのまま渡せる（policy は自動解決）。
`GridSearchCV` を `RandomizedSearchCV` / `HalvingGridSearchCV` / 自作の CV 探索器に差し替えても
呼び出し側（`tune_ml_estimator` 以降のコード）は変わらない。

注意: 実行時間を数秒以内に収めるため学習データは 15 件・`cv=2` の最小構成にしている。

環境変数不要で実行:
    uv run --group examples python examples/intent/14_ml_tuning_gridsearch.py
"""

from __future__ import annotations

import asyncio
import sys
import time

try:
    import sklearn  # noqa: F401
except ModuleNotFoundError:
    print(
        "ModuleNotFoundError: scikit-learn is required for this example.\n"
        "Install it via:  uv sync --group examples",
        file=sys.stderr,
    )
    sys.exit(1)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from oai_agentspec.runtime.intent import (
    IntentCategory,
    IntentPolicy,
    IntentQuery,
    intent_classifier_from_ml_inference,
    tune_ml_estimator,
)

POLICY = IntentPolicy(
    categories=(
        IntentCategory(name="refund", description="返金・返品に関する問い合わせ"),
        IntentCategory(name="cancel", description="解約・キャンセルに関する問い合わせ"),
        IntentCategory(name="other", description="上記に当てはまらない問い合わせ"),
    ),
    max_candidates=3,
)

# 学習データ: (発話, ラベル) の対で保持する（順序と対応関係を局所化する。08 参照）。
# 各カテゴリ 5 件・計 15 件（cv=2 の各 fold で汎化できる最小規模）。
_TRAINING_DATA: list[tuple[str, str]] = [
    ("商品を返品して返金してほしい", "refund"),
    ("購入した商品の返金を希望します", "refund"),
    ("壊れていたので返品したい", "refund"),
    ("届いた商品を返品したいので返金してください", "refund"),
    ("不良品だったため返金をお願いします", "refund"),
    ("サブスクリプションを解約したい", "cancel"),
    ("契約をキャンセルしたいです", "cancel"),
    ("定期購入を今すぐ止めてほしい", "cancel"),
    ("月額プランを解約する方法を教えてください", "cancel"),
    ("自動更新をキャンセルしたい", "cancel"),
    ("営業時間を教えてください", "other"),
    ("アカウントのパスワードを変更したい", "other"),
    ("対応言語は何がありますか", "other"),
    ("アプリの使い方が分かりません", "other"),
    ("ログイン方法を知りたいです", "other"),
]
_TRAIN_TEXTS: list[str] = [text for text, _ in _TRAINING_DATA]
_TRAIN_LABELS: list[str] = [label for _, label in _TRAINING_DATA]

_THRESHOLDS: dict[str, float] = {
    "certain": 0.90,
    "high": 0.75,
    "medium": 0.50,
    "low": 0.25,
    "speculative": 0.0,
}

# 探索対象パラメータグリッド: LogisticRegression の正則化強度 C のみ 3 通り。
# cv=2 と合わせて実行時間を数秒以内に抑える。
_PARAM_GRID: dict[str, list[float]] = {"clf__C": [0.1, 1.0, 10.0]}

# 評価指標。探索器側の設定であり tune_ml_estimator には渡さない。既定（未指定）は
# estimator の score()（分類器なら accuracy）で、明示すると best_score / cv_results の
# 値の意味が変わる。lib はこの値を解釈も変換もせず、そのまま成果物へ載せる。
_SCORING = "f1_macro"

# held-out テストセット: (発話, 正解ラベル) の対。学習データとは異なる言い回しを使い、
# 3 カテゴリすべてで分類が正しく機能することを示す。
_TEST_DATA: list[tuple[str, str]] = [
    ("商品が届いたのですが返金してもらえますか", "refund"),
    ("プレミアムプランをやめたいのですが", "cancel"),
    ("休業日はいつですか", "other"),
]


def _print_cv_results(cv_results: dict[str, object]) -> None:
    """`cv_results_` から候補ごとの平均スコアと順位を整形して stdout へ表示する。

    pandas は依存を増やさないため使わず、`cv_results` の並列配列（`params` /
    `mean_test_score` / `rank_test_score`）を zip して手組みで整形する。

    Args:
        cv_results: `TunedIntentEstimator.cv_results`（sklearn の `cv_results_` 由来）。
    """
    params = cv_results["params"]
    mean_scores = cv_results["mean_test_score"]
    ranks = cv_results["rank_test_score"]
    rows = sorted(zip(ranks, mean_scores, params, strict=True), key=lambda row: row[0])
    for rank, mean_score, candidate_params in rows:
        print(f"  rank={rank} mean_test_score={mean_score:.4f} params={candidate_params}")


async def main() -> None:
    """GridSearchCV でハイパーパラメータ探索を行い、探索副産物と分類結果を表示する。"""
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3))),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    # scoring / cv / param_grid は探索器側の設定。tune_ml_estimator のシグネチャには現れない。
    search = GridSearchCV(pipeline, param_grid=_PARAM_GRID, cv=2, scoring=_SCORING)

    tune_start = time.perf_counter()
    # transform は省略可: 既定で [ctx.utterance]（単一サンプル列）が best_estimator_ に渡る。
    tuned = tune_ml_estimator(
        search,
        x_train=_TRAIN_TEXTS,
        y_train=_TRAIN_LABELS,
        policy=POLICY,
    )
    tune_ms = (time.perf_counter() - tune_start) * 1000

    print(f"[tune] {len(_TRAINING_DATA)} 件 x {len(_PARAM_GRID['clf__C'])} 候補: {tune_ms:.2f} ms")
    print(f"[best_params] {tuned.best_params}")
    # best_score / cv_results の値が何の指標かは探索器の scoring が決める。lib は解釈しない
    # ため、意味を知りたければ探索器側（sklearn なら scorer_）を見る。
    print(
        f"[best_score] {tuned.best_score:.4f} (scoring={_SCORING})"
        if tuned.best_score is not None
        else "[best_score] None"
    )
    print("[cv_results] 候補ごとの平均スコアと順位:")
    if tuned.cv_results is not None:
        _print_cv_results(dict(tuned.cv_results))

    # policy は省略可: tuned（TunedIntentEstimator は TrainedIntentEstimator のサブクラス）が
    # 保持する policy から自動解決される。探索器を差し替えても intent_classifier_from_ml_inference
    # 以降の呼び出しは一切変わらない（推論器は探索器ではなく best_estimator_ に束縛されるため）。
    classifier = intent_classifier_from_ml_inference(
        tuned,
        thresholds=_THRESHOLDS,
    )

    correct = 0
    for utt, expected in _TEST_DATA:
        prediction = await classifier.classify(IntentQuery(utterance=utt))
        predicted = prediction.candidates[0].text if prediction.candidates else None
        level = prediction.candidates[0].level if prediction.candidates else "-"
        mark = "OK" if predicted == expected else "NG"
        if predicted == expected:
            correct += 1
        print(f"[{mark}] {utt} -> 予測: {predicted}/正解: {expected} ({level})")

    print(f"[accuracy] {correct}/{len(_TEST_DATA)} ({100 * correct / len(_TEST_DATA):.1f}%)")


if __name__ == "__main__":
    asyncio.run(main())
