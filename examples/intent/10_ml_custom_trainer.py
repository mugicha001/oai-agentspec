"""学習手段非依存の `IntentTrainer` 契約を標準ライブラリのみで直接適合する例。

sklearn 等の外部 ML ライブラリに依存せず、固定辞書ベースのキーワード分類手順を
`IntentTrainer`（FR-4a）のシグネチャ（`(...) -> TrainedIntentEstimator`）で実装し、
`make_trained_estimator` で `TrainedIntentEstimator` に包む。組み立てた推論 callable は
`intent_classifier_from_ml_inference`（FR-5）にそのまま渡せる。LLM も sklearn も
使わないため追加依存ゼロで動く。

環境変数不要 & sklearn 不要で実行:
    uv run python examples/intent/10_ml_custom_trainer.py
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence

from oai_agentspec.runtime.intent import (
    IntentCategory,
    IntentContext,
    IntentPolicy,
    IntentQuery,
    TrainedIntentEstimator,
    intent_classifier_from_ml_inference,
    make_trained_estimator,
)

POLICY = IntentPolicy(
    categories=(
        IntentCategory(name="refund", description="返金・返品に関する問い合わせ"),
        IntentCategory(name="cancel", description="解約・キャンセルに関する問い合わせ"),
        IntentCategory(name="other", description="上記に当てはまらない問い合わせ"),
    ),
    max_candidates=3,
)

# カテゴリ名 -> キーワード集合（ダミー学習手順が参照する固定辞書）。
_KEYWORDS: dict[str, frozenset[str]] = {
    "refund": frozenset({"返品", "返金", "払い戻し"}),
    "cancel": frozenset({"解約", "キャンセル", "解除"}),
}

_THRESHOLDS: dict[str, float] = {
    "certain": 0.90,
    "high": 0.75,
    "medium": 0.50,
    "low": 0.25,
    "speculative": 0.0,
}

_TEST_UTTERANCES: list[str] = [
    "商品を返品して返金してほしい",
    "サブスクリプションを解約したい",
    "営業時間を教えてください",
]


def _infer(context: IntentContext[None]) -> list[tuple[str, float]]:
    """固定辞書とのキーワードマッチでカテゴリごとのスコアを算出する。

    Args:
        context: 分類対象の文脈（`utterance` のみ参照する）。

    Returns:
        (カテゴリ名, スコア) の列。ヒットしたカテゴリのみを返す。
    """
    scores: list[tuple[str, float]] = []
    for category, keywords in _KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in context.utterance)
        if hits:
            scores.append((category, min(1.0, 0.5 + 0.25 * hits)))
    return scores


def train_dummy_classifier(
    texts: Sequence[str],
    labels: Sequence[str],
    policy: IntentPolicy,
) -> TrainedIntentEstimator:
    """固定辞書ベースのダミー学習手順（`IntentTrainer` 契約への直接適合）。

    実際の学習（パラメータ調整）は行わず、モジュール定数の `_KEYWORDS` を用いた
    キーワードマッチ推論 callable をそのまま包んで返す。`texts` / `labels` は
    `IntentTrainer` シグネチャ（学習データを受け取る形）の実演のためのみに存在し、
    本実装では参照しない。

    Args:
        texts: 学習発話列（本実装では未使用）。
        labels: 学習ラベル列（本実装では未使用）。
        policy: 分類器が守る契約（本実装では未使用・将来の拡張ポイント）。

    Returns:
        キーワードマッチ推論 callable を含む `TrainedIntentEstimator`。
    """
    del texts, labels, policy  # ダミー学習のため未使用（シグネチャ実演目的）
    return make_trained_estimator(inference=_infer)


async def main() -> None:
    """ダミー学習手順で組み立てた分類器で複数発話を分類する。"""
    fit_start = time.perf_counter()
    trained = train_dummy_classifier(texts=(), labels=(), policy=POLICY)
    fit_ms = (time.perf_counter() - fit_start) * 1000

    classifier = intent_classifier_from_ml_inference(
        trained,
        policy=POLICY,
        thresholds=_THRESHOLDS,
    )

    print(f"[fit] ダミー学習手順（キーワード辞書ベース・パラメータ調整なし）: {fit_ms:.2f} ms")

    for utt in _TEST_UTTERANCES:
        classify_start = time.perf_counter()
        prediction = await classifier.classify(IntentQuery(utterance=utt))
        classify_ms = (time.perf_counter() - classify_start) * 1000
        if not prediction.candidates:
            print(f"[?] {utt} -> 候補なし ({classify_ms:.2f} ms)")
            continue
        top = prediction.candidates[0]
        print(f"[{top.text}] {utt} -> {top.text} ({top.level}) ({classify_ms:.2f} ms)")


if __name__ == "__main__":
    asyncio.run(main())
