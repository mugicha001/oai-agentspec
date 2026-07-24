"""事前ベクトル化した数値特徴量経路で ML 推論 callable を組み立てる例。

**08 との違い**: 08 は `fit_ml_estimator`（FR-4b）で lib に fit を駆動させるが、
本例は **利用者側で fit を回した** vectorizer + estimator を lib に持ち込み、
`ml_inference_from_estimator`（FR-4c）で推論 callable にラップする（lib は
`fit` を一切駆動しない）。生テキスト -> 数値特徴量への変換は利用者責務であり、
本例では `vectorizer.transform` を推論時の `transform=` 引数で明示的に渡す
（08 は transform 省略時の既定 `[ctx.utterance]` をそのまま利用するのに対し、
本例は数値特徴量への変換が必須のため明示指定する）。

スコア -> `ConfidenceLevel` の変換は `confidence_mapper_from_thresholds`（FR-1）で
構築した mapper を `intent_classifier_from_ml_inference`（FR-5）に渡す（08 の
`thresholds=` 直接指定の対称形）。LLM を一切呼ばないため API キー・環境変数は不要。

環境変数不要で実行:
    uv run --group examples python examples/intent/09_ml_pretrained_features.py
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

from oai_agentspec.runtime.intent import (
    IntentCategory,
    IntentPolicy,
    IntentQuery,
    confidence_mapper_from_thresholds,
    intent_classifier_from_ml_inference,
    ml_inference_from_estimator,
)

POLICY = IntentPolicy(
    categories=(
        IntentCategory(name="refund", description="返金・返品に関する問い合わせ"),
        IntentCategory(name="cancel", description="解約・キャンセルに関する問い合わせ"),
        IntentCategory(name="other", description="上記に当てはまらない問い合わせ"),
    ),
    max_candidates=3,
)

# 学習データ: (発話, ラベル) の対で保持する（順序と対応関係を局所化する）。
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

# held-out テストセット: (発話, 正解ラベル) の対。08 と同一（学習データとは異なる
# 言い回しで held-out 精度を評価する。小規模データのため精度は高くない・詳細は 08 参照）。
_TEST_DATA: list[tuple[str, str]] = [
    ("先月分の代金がまだ戻ってきていません", "refund"),
    ("届いた商品に不具合があったので払い戻してもらえますか", "refund"),
    ("プレミアムプランをやめたいのですが", "cancel"),
    ("毎月の引き落としを止める手続きを知りたい", "cancel"),
    ("休業日はいつですか", "other"),
    ("問い合わせ窓口の連絡先を教えてください", "other"),
]


async def main() -> None:
    """利用者側で fit した vectorizer + estimator を推論 callable にラップして精度評価する。"""
    # 学習は lib の外で回す（本例の主眼: lib は fit を駆動せず・利用者持ち込みを受け取る）。
    # 日本語は空白区切りでないため char n-gram（char_wb）でトークン化する。
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3))
    estimator = LogisticRegression(max_iter=1000)

    fit_start = time.perf_counter()
    x_train = vectorizer.fit_transform(_TRAIN_TEXTS)
    estimator.fit(x_train, _TRAIN_LABELS)
    fit_ms = (time.perf_counter() - fit_start) * 1000

    inference = ml_inference_from_estimator(
        estimator,
        transform=lambda ctx: vectorizer.transform([ctx.utterance]),
    )
    mapper = confidence_mapper_from_thresholds(
        certain=0.90,
        high=0.75,
        medium=0.50,
        low=0.25,
        speculative=0.0,
    )

    classifier = intent_classifier_from_ml_inference(
        inference,
        policy=POLICY,
        mapper=mapper,
    )

    print(f"[fit] {len(_TRAINING_DATA)} 件を学習（lib 外・利用者側）: {fit_ms:.2f} ms")

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
