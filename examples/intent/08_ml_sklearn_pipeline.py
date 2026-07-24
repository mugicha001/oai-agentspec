"""生テキスト -> sklearn `Pipeline` を `fit_ml_estimator` に渡すゼロコード fit の例。

`TfidfVectorizer` + `LogisticRegression` を束ねた sklearn `Pipeline` を学習データで
`fit_ml_estimator`（FR-4b）に 1 回だけ fit させ、返る `TrainedIntentEstimator` を
`intent_classifier_from_ml_inference`（FR-5）で `DefaultIntentClassifier` に組み上げる。
LLM を一切呼ばないため API キー・環境変数は不要。

環境変数不要で実行:
    uv run --group examples python examples/intent/08_ml_sklearn_pipeline.py
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
from sklearn.pipeline import Pipeline

from oai_agentspec.runtime.intent import (
    IntentCategory,
    IntentPolicy,
    IntentQuery,
    fit_ml_estimator,
    intent_classifier_from_ml_inference,
)

POLICY = IntentPolicy(
    categories=(
        IntentCategory(name="refund", description="返金・返品に関する問い合わせ"),
        IntentCategory(name="cancel", description="解約・キャンセルに関する問い合わせ"),
        IntentCategory(name="other", description="上記に当てはまらない問い合わせ"),
    ),
    max_candidates=3,
)

# 学習データ: (発話, ラベル) の対で保持する（順序と対応関係を局所化し、
# 追加・並び替え・シャッフル時のバグを防ぐ）。
_TRAINING_DATA: list[tuple[str, str]] = [
    ("商品を返品して返金してほしい", "refund"),
    ("購入した商品の返金を希望します", "refund"),
    ("領収書と違う金額が請求されたので返金してください", "refund"),
    ("壊れていたので返品したい", "refund"),
    ("返金の手続き方法を教えてください", "refund"),
    ("支払った代金を返してほしい", "refund"),
    ("サイズが合わなかったので返品したい", "refund"),
    ("誤って二重決済されたので払い戻してほしい", "refund"),
    ("サブスクリプションを解約したい", "cancel"),
    ("契約をキャンセルしたいです", "cancel"),
    ("定期購入を今すぐ止めてほしい", "cancel"),
    ("会員登録を解除してください", "cancel"),
    ("来月からの契約を解約したい", "cancel"),
    ("予約をキャンセルしたい", "cancel"),
    ("自動更新を止めてほしい", "cancel"),
    ("利用プランを解約する方法は？", "cancel"),
    ("営業時間を教えてください", "other"),
    ("アカウントのパスワードを変更したい", "other"),
    ("対応言語は何がありますか", "other"),
    ("アプリの使い方が分かりません", "other"),
    ("新しい機能について知りたい", "other"),
    ("サポートセンターの電話番号は？", "other"),
    ("他の商品と比較したい", "other"),
    ("メールアドレスを変更したい", "other"),
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

# held-out テストセット: (発話, 正解ラベル) の対。学習データ (_TRAINING_DATA) とは
# 異なる言い回しを使い、精度評価が意味を持つようにする（学習データの類義文の焼き直しは
# 「まぐれ当たり」を精度に見せかけてしまうため避ける）。
# 注意: 学習データは各カテゴリ 8 件のごく小規模なミニ辞書のため、held-out（未見の
# 言い回し）に対する精度は高くない（実測 30〜50% 程度）。これは実装の不具合ではなく
# 「サンプルとして動かせる最小規模のデータ」の限界である。実運用では数百〜数千件規模の
# 学習データを用意すること。
_TEST_DATA: list[tuple[str, str]] = [
    ("先月分の代金がまだ戻ってきていません", "refund"),
    ("届いた商品に不具合があったので払い戻してもらえますか", "refund"),
    ("プレミアムプランをやめたいのですが", "cancel"),
    ("毎月の引き落としを止める手続きを知りたい", "cancel"),
    ("休業日はいつですか", "other"),
    ("問い合わせ窓口の連絡先を教えてください", "other"),
]


async def main() -> None:
    """sklearn Pipeline を学習し、held-out データで分類精度を評価する。"""
    # 日本語は空白区切りでないため、既定の word analyzer では分割されない。
    # 文字 n-gram（char_wb）でトークン化することでキーワード部分文字列を捕捉する。
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3))),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )

    fit_start = time.perf_counter()
    # transform は省略可: 既定で [ctx.utterance]（単一サンプル列）が estimator に渡る。
    trained = fit_ml_estimator(
        pipeline,
        x_train=_TRAIN_TEXTS,
        y_train=_TRAIN_LABELS,
        policy=POLICY,
    )
    fit_ms = (time.perf_counter() - fit_start) * 1000

    # policy は省略可: trained（TrainedIntentEstimator）が保持する policy から自動解決される。
    classifier = intent_classifier_from_ml_inference(
        trained,
        thresholds=_THRESHOLDS,
    )

    print(f"[fit] {len(_TRAINING_DATA)} 件を学習: {fit_ms:.2f} ms")

    correct = 0
    for utt, expected in _TEST_DATA:
        classify_start = time.perf_counter()
        prediction = await classifier.classify(IntentQuery(utterance=utt))
        classify_ms = (time.perf_counter() - classify_start) * 1000
        predicted = prediction.candidates[0].text if prediction.candidates else None
        mark = "✓" if predicted == expected else "✗"
        if predicted == expected:
            correct += 1
        level = prediction.candidates[0].level if prediction.candidates else "-"
        print(
            f"[{mark}] {utt} -> 予測: {predicted}/正解: {expected} ({level}) ({classify_ms:.2f} ms)"
        )

    print(f"[accuracy] {correct}/{len(_TEST_DATA)} ({100 * correct / len(_TEST_DATA):.1f}%)")


if __name__ == "__main__":
    asyncio.run(main())
