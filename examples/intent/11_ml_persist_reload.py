"""fit 済み estimator を pickle で永続化し、別プロセス想定で再接続する例。

`fit_ml_estimator`（FR-4b）で学習した estimator を `pickle.dumps` で bytes 化し、
「別プロセス」を想定して `pickle.loads` で復元したあと、`ml_inference_from_estimator`
（FR-4c）で推論 callable へ再接続する。

セキュリティ注意: pickle は信頼できないファイルを load しない（任意コード実行の
危険）。運用では joblib も選択肢になる。

注意: `TrainedIntentEstimator` を丸ごと pickle することはできない（`inference` が
クロージャのため）。保存対象は `.estimator` のみとし、リロード側では
`decoder=`（`label_encoding` 使用時のみ）と FR-5 の `policy=` を再指定する。

環境変数不要で実行:
    uv run --group examples python examples/intent/11_ml_persist_reload.py
"""

from __future__ import annotations

import asyncio
import pickle
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

_THRESHOLDS: dict[str, float] = {
    "certain": 0.90,
    "high": 0.75,
    "medium": 0.50,
    "low": 0.25,
    "speculative": 0.0,
}

# held-out テストセット: (発話, 正解ラベル) の対。08/09 と同一（詳細は 08 参照）。
_TEST_DATA: list[tuple[str, str]] = [
    ("先月分の代金がまだ戻ってきていません", "refund"),
    ("届いた商品に不具合があったので払い戻してもらえますか", "refund"),
    ("プレミアムプランをやめたいのですが", "cancel"),
    ("毎月の引き落としを止める手続きを知りたい", "cancel"),
    ("休業日はいつですか", "other"),
    ("問い合わせ窓口の連絡先を教えてください", "other"),
]


async def main() -> None:
    """estimator を学習・pickle 化し、別プロセス想定で復元して分類する。"""
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3))),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )

    fit_start = time.perf_counter()
    trained = fit_ml_estimator(
        pipeline,
        x_train=_TRAIN_TEXTS,
        y_train=_TRAIN_LABELS,
        policy=POLICY,
    )
    fit_ms = (time.perf_counter() - fit_start) * 1000
    print(f"[fit] {len(_TRAINING_DATA)} 件を学習: {fit_ms:.2f} ms")

    # 保存対象は estimator のみ（TrainedIntentEstimator 丸ごとは pickle 不可）。
    payload = pickle.dumps(trained.estimator)
    print(f"[persist] estimator を pickle 化: {len(payload)} bytes")

    # --- ここから別プロセス想定（実運用ではファイル/ストレージ経由で受け渡す） ---
    restored_estimator = pickle.loads(payload)  # noqa: S301

    # transform は省略可: 既定で [ctx.utterance] が estimator に渡る。
    inference = ml_inference_from_estimator(restored_estimator)
    # policy は callable 直渡しのため明示指定が必要（TrainedIntentEstimator からの
    # 自動解決は使えない）。
    classifier = intent_classifier_from_ml_inference(
        inference,
        policy=POLICY,
        thresholds=_THRESHOLDS,
    )

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
