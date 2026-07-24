"""自作 `CandidateGenerator` を差し込む例（LLM 不使用・オフライン実行）。

`CandidateGenerator` Protocol を満たすキーワードマッチ分類器を自作し、
`intent_classifier_from_generator` の 1 行で `DefaultIntentClassifier` に束ねる。
LLM を一切呼ばないため API キー・環境変数なしで実行でき、コスト・レイテンシは
ゼロになる（embedding 分類など他方式も同じ差し込み口で実現できる）。

`IntentPolicy` の強制（allowlist / レベル降順 sort / max_candidates truncate）は
generator 実装の責務（`protocols.CandidateGenerator` の docstring 参照）。この例では
sort / truncate を自前適用する。allowlist は taxonomy のカテゴリ名しか候補に
生成しない構成のため自明に充足される。

環境変数不要で実行:
    uv run python examples/intent/07_custom_candidate_generator.py
"""

from __future__ import annotations

import asyncio
from typing import Any

from oai_agentspec.runtime.intent import (
    ConfidenceLevel,
    IntentCandidate,
    IntentCategory,
    IntentContext,
    IntentPolicy,
    IntentPrediction,
    IntentQuery,
    intent_classifier_from_generator,
)

POLICY = IntentPolicy(
    categories=(
        IntentCategory(name="billing", description="請求・支払いに関する問い合わせ"),
        IntentCategory(name="technical", description="技術的なトラブル・エラーの相談"),
    ),
    max_candidates=2,
)

# カテゴリ名 -> キーワード集合（taxonomy のカテゴリ名しかキーにしない = allowlist 自明充足）。
KEYWORDS: dict[str, frozenset[str]] = {
    "billing": frozenset({"請求", "支払い", "請求書", "領収書", "金額"}),
    "technical": frozenset({"エラー", "動かない", "落ちる", "遅い", "ログイン"}),
}

# ヒット数 -> ConfidenceLevel の写像（application 側のルール）。
_HITS_TO_LEVEL = {1: ConfidenceLevel.MEDIUM, 2: ConfidenceLevel.HIGH}
_CERTAIN_THRESHOLD = 3

# ConfidenceLevel の宣言順（certain -> speculative）を降順ソートキーに転用する。
_LEVEL_ORDER = {level: idx for idx, level in enumerate(ConfidenceLevel)}


def _match_keywords(utterance: str, keywords: frozenset[str]) -> list[str]:
    """最長一致 + 消費でキーワードを照合する（部分文字列の二重計上を防ぐ）。

    長いキーワードから順に照合し、マッチした区間を消費してから短いキーワードを
    照合する。「請求書」がヒットしたら内包される「請求」は同じ区間で再カウント
    されない（独立に両方出現すれば両方ヒットする）。
    """
    remaining = utterance
    hits: list[str] = []
    for kw in sorted(keywords, key=len, reverse=True):
        if kw in remaining:
            hits.append(kw)
            remaining = remaining.replace(kw, "\x00")  # マッチ区間を消費する
    return sorted(hits)


class KeywordCandidateGenerator:
    """キーワードマッチによる `CandidateGenerator` Protocol の自作実装。

    utterance に含まれるカテゴリ別キーワードのヒット数（最長一致・重複区間は
    二重計上しない）を ConfidenceLevel へ写像する（3 件以上 = certain /
    2 件 = high / 1 件 = medium・ヒット 0 は候補外）。
    継承は不要で、`async def generate` を持つだけで Protocol を満たす。
    """

    def __init__(self, policy: IntentPolicy, keywords: dict[str, frozenset[str]]) -> None:
        self._policy = policy
        self._keywords = keywords

    async def generate(self, context: IntentContext[Any]) -> IntentPrediction:
        matched: dict[str, list[str]] = {}
        candidates: list[IntentCandidate] = []
        for category in self._policy.categories:
            hits = _match_keywords(
                context.utterance, self._keywords.get(category.name, frozenset())
            )
            if not hits:
                continue  # ヒット 0 のカテゴリは候補に入れない
            matched[category.name] = hits
            level = (
                ConfidenceLevel.CERTAIN
                if len(hits) >= _CERTAIN_THRESHOLD
                else _HITS_TO_LEVEL[len(hits)]
            )
            candidates.append(IntentCandidate(text=category.name, level=level))

        # policy 強制は generator の責務: レベル降順 sort -> max_candidates truncate。
        candidates.sort(key=lambda c: _LEVEL_ORDER[c.level])
        candidates = candidates[: self._policy.max_candidates]

        return IntentPrediction(
            candidates=tuple(candidates),
            # matched_keywords は truncate 前の全カテゴリのマッチ観測値
            # （candidates の内訳ではない。切り落とされたカテゴリの内訳も残る）。
            metadata={"matched_keywords": matched},
        )


async def main() -> None:
    # 1 行ヘルパ: 自作 generator + DefaultContextBuilder を束ねる（LLM・model 不要）。
    classifier = intent_classifier_from_generator(
        KeywordCandidateGenerator(policy=POLICY, keywords=KEYWORDS),
    )

    for label, utt in [
        (
            "明確 billing（3 ヒット -> certain）",
            "先月の請求書と領収書を再発行して、支払い方法も変えたい",
        ),
        ("明確 technical（2 ヒット -> high）", "エラーで動かない"),
        ("複合（両カテゴリが候補・降順）", "請求画面がエラーで動かない"),
        ("ヒットなし（空 candidates）", "こんにちは"),
    ]:
        prediction = await classifier.classify(IntentQuery(utterance=utt))
        print("\n" + "=" * 60)
        print(f"[{label}]")
        print(f"[UTTERANCE] {utt}")
        for i, c in enumerate(prediction.candidates):
            print(f"[CANDIDATE] #{i + 1} {c.text} (level={c.level.value})")
        if not prediction.candidates:
            print("[CANDIDATE] なし（下流で fallback を判断する）")
        print(f"[METADATA]  {prediction.metadata}")


if __name__ == "__main__":
    asyncio.run(main())
