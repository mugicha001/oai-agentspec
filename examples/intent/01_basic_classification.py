"""意図予測の最小例（1 行ヘルパ・実 API）。

`intent_classifier_from_model` に model + prompt callable + IntentPolicy を渡し、
1 発話を分類する。分類実行時に LLM へ流れる合成プロンプト
（`policy.render_prompt()` = SYSTEM / `prompt(context)` = USER）も表示する。

Azure OpenAI の環境変数（examples/_shared/_azure.py 参照）を設定して実行:
    uv run python examples/intent/01_basic_classification.py

導入: pip install 'oai-agentspec[intent]'（pydantic のみ）。
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from oai_agentspec.runtime.intent import (
    IntentCategory,
    IntentContext,
    IntentPolicy,
    IntentQuery,
    intent_classifier_from_model,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))

from _timing import stopwatch  # noqa: E402

from _azure import azure_model  # noqa: E402


def build_prompt(context: IntentContext) -> str:
    """LLM に渡す user 入力を組み立てる（履歴は SDK が multi-turn として届ける）。"""
    return f"次の発話を分類してください:\n{context.utterance}"


async def main() -> None:
    policy = IntentPolicy(
        categories=(
            IntentCategory(name="ask_price", description="価格・料金に関する質問"),
            IntentCategory(name="ask_spec", description="仕様・機能に関する質問"),
            IntentCategory(name="chitchat", description="雑談・挨拶"),
        ),
        max_candidates=3,
    )

    classifier = intent_classifier_from_model(
        model=azure_model(),
        prompt=build_prompt,
        policy=policy,
    )

    query = IntentQuery(utterance="このプランは月いくらですか?")

    # 分類実行時に LLM へ流れる合成プロンプトを表示する。
    context = await classifier.context_builder.build(query)
    print("=" * 60)
    print("[SYSTEM] policy.render_prompt() の出力")
    print("=" * 60)
    print(policy.render_prompt())
    print()
    print("=" * 60)
    print("[USER] prompt(context) の出力")
    print("=" * 60)
    print(build_prompt(context))
    print()

    with stopwatch("classify"):
        prediction = await classifier.classify(query)

    print("=" * 60)
    print("[RESULT] IntentPrediction")
    print("=" * 60)
    # rationale は既定 (include_rationale_in_prompt=False) では生成されないため表示しない。
    # 判断理由が必要なら IntentPolicy(include_rationale_in_prompt=True) を指定する (例 04 参照)。
    for i, c in enumerate(prediction.candidates):
        print(f"  #{i + 1} text={c.text} level={c.level.value}")


if __name__ == "__main__":
    asyncio.run(main())
