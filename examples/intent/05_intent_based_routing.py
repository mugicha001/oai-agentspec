"""意図分類 → 下流エージェントへ dispatch する統合フロー例（実 API）。

分類結果の先頭候補で `AgentRegistry` の下流エージェントを選び、SDK `Runner` で
応答させる。lib 本体は実行分岐（PolicyEngine）を持たないため、分岐は利用側
application 責務であることを示す（コメント参照）。合成プロンプトの表示例は
例 01-04 を参照。

Azure OpenAI の環境変数（examples/_shared/_azure.py 参照）を設定して実行:
    uv run python examples/intent/05_intent_based_routing.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from agents import Runner

from oai_agentspec import AgentRegistry, AgentSpec
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

POLICY = IntentPolicy(
    categories=(
        IntentCategory(name="billing", description="請求・支払いに関する問い合わせ"),
        IntentCategory(name="technical", description="技術的なトラブル・エラーの相談"),
        IntentCategory(name="general", description="上記以外の一般的な問い合わせ"),
    ),
)


def build_prompt(context: IntentContext) -> str:
    return f"次の発話を分類してください:\n{context.utterance}"


def build_registry() -> AgentRegistry:
    """分類先エージェント群を宣言（下流の応答エージェント）。"""
    registry = AgentRegistry()
    registry.register(
        AgentSpec(
            name="billing",
            instructions="あなたは請求担当です。金額や請求書に関する質問へ簡潔に答えます。",
            model=azure_model(),
        )
    )
    registry.register(
        AgentSpec(
            name="technical",
            instructions="あなたは技術サポートです。トラブル切り分けの手順を提示します。",
            model=azure_model(),
        )
    )
    registry.register(
        AgentSpec(
            name="general",
            instructions="あなたは一般問い合わせ担当です。総合的に案内します。",
            model=azure_model(),
        )
    )
    registry.validate()
    return registry


async def classify_and_route(utterance: str) -> None:
    classifier = intent_classifier_from_model(
        model=azure_model(),
        prompt=build_prompt,
        policy=POLICY,
    )
    registry = build_registry()

    with stopwatch("classify"):
        prediction = await classifier.classify(IntentQuery(utterance=utterance))
    if not prediction.candidates:
        # 空 candidates の原因は戻り値からは区別できない（LLM が候補を返さなかった /
        # allowlist で全除外された）。除外の有無は logger.warning
        # （oai_agentspec.runtime.intent._llm）または SDK Span で確認する。
        print(f"\n[UTTERANCE] {utterance}\n  -> 有効候補なし（原因は logger.warning / Span 参照）")
        return

    # ライブラリは実行分岐を持たない。ここは application 側の dispatch。
    top = prediction.candidates[0]
    agent = registry.get(top.text)
    with stopwatch("dispatch (下流エージェント応答)"):
        result = await Runner.run(agent, input=utterance)

    print("\n" + "=" * 60)
    print(f"[UTTERANCE] {utterance}")
    print(f"[INTENT]    {top.text} (level={top.level.value})")
    print(f"[AGENT]     {agent.name}")
    print(f"[ANSWER]    {str(result.final_output)[:200]}")


async def main() -> None:
    for utt in [
        "先月の請求書のPDFが届かないので送ってほしい",
        "ログイン後に画面が真っ白になります",
    ]:
        await classify_and_route(utt)


if __name__ == "__main__":
    asyncio.run(main())
