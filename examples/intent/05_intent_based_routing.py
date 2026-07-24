"""信頼度で分岐する intent 分類の応用例（実 API）。

分類結果の信頼度（`ConfidenceLevel`）と複数候補を使い、確信が高ければ下流エージェント
へ dispatch し、低ければ実行せずに候補を提示して聞き返す。「実行しない判断」「信頼度
による分岐」「複数候補の提示」は handoff 機構では表現できない intent 分類固有の
ユースケース（入口ルーティングだけが目的なら `06_dynamic_edge_routing.py` の
dynamic_edge が SDK ネイティブで推奨）。

lib 本体は実行分岐（PolicyEngine）を持たないため、分岐は利用側 application 責務で
あることを示す（コメント参照）。

Azure OpenAI の環境変数（examples/_shared/_azure.py 参照）を設定して実行:
    uv run python examples/intent/05_intent_based_routing.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from agents import ModelSettings, Runner
from openai.types.shared import Reasoning

from oai_agentspec import AgentRegistry, AgentSpec
from oai_agentspec.runtime.intent import (
    ConfidenceLevel,
    IntentCategory,
    IntentContext,
    IntentPolicy,
    IntentQuery,
    intent_classifier_from_model,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
from _timing import stopwatch  # noqa: E402
from _warmup import warmup  # noqa: E402

from _azure import azure_model  # noqa: E402

# reasoning 系モデルの思考トークンを止めて分類レイテンシを最小化する（利用側 DI・分類器のみ適用）。
# 非 reasoning デプロイ（gpt-4.1-nano 等）では AZURE_OPENAI_REASONING=0 を設定すると
# reasoning / verbosity パラメータ自体を送らない（未対応モデルでの API エラーを回避）。
if os.environ.get("AZURE_OPENAI_REASONING", "1") != "0":
    MODEL_SETTINGS = ModelSettings(
        reasoning=Reasoning(effort="none"), verbosity="low", max_tokens=100
    )
else:
    MODEL_SETTINGS = ModelSettings(max_tokens=100)

# certain / high なら dispatch、それ未満なら実行せず聞き返す（application 側の閾値）。
DISPATCH_LEVELS = {ConfidenceLevel.CERTAIN, ConfidenceLevel.HIGH}

POLICY = IntentPolicy(
    categories=(
        IntentCategory(name="billing", description="請求・支払いに関する問い合わせ"),
        IntentCategory(name="technical", description="技術的なトラブル・エラーの相談"),
        IntentCategory(name="general", description="上記以外の一般的な問い合わせ"),
    ),
)

CATEGORY_LABELS = {c.name: c.description for c in POLICY.categories}


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


async def classify_then_decide(model: object, registry: AgentRegistry, utterance: str) -> None:
    classifier = intent_classifier_from_model(
        model=model,
        prompt=build_prompt,
        policy=POLICY,
        model_settings=MODEL_SETTINGS,
    )

    with stopwatch("classify"):
        prediction = await classifier.classify(IntentQuery(utterance=utterance))

    print("\n" + "=" * 60)
    print(f"[UTTERANCE] {utterance}")
    for i, c in enumerate(prediction.candidates):
        print(f"[CANDIDATE] #{i + 1} {c.text} (level={c.level.value})")

    if not prediction.candidates:
        # 空 candidates の原因は戻り値からは区別できない（LLM が候補を返さなかった /
        # allowlist で全除外された）。除外の有無は logger.warning
        # （oai_agentspec.runtime.intent._llm）または SDK Span で確認する。
        print("[DECISION]  有効候補なし（原因は logger.warning / Span 参照）")
        return

    # ここからは application 側の分岐（lib は実行分岐を持たない）。
    top = prediction.candidates[0]
    if top.level in DISPATCH_LEVELS:
        # 確信が高い: 下流エージェントへ dispatch する。
        agent = registry.get(top.text)
        with stopwatch("dispatch (下流エージェント応答)"):
            result = await Runner.run(agent, input=utterance)
        print(f"[DECISION]  dispatch -> {agent.name} (level={top.level.value})")
        print(f"[ANSWER]    {str(result.final_output)[:200]}")
    else:
        # 確信が低い: 下流を実行せず、候補を提示して聞き返す（LLM 呼び出しゼロ）。
        # 「実行しない」選択と複数候補の提示は分類結果をデータとして持つから可能になる。
        choices = " / ".join(
            f"{CATEGORY_LABELS[c.text]}（{c.text}）" for c in prediction.candidates
        )
        print(f"[DECISION]  確信度不足 (top={top.level.value}) -> dispatch せず聞き返す")
        print(f"[ASK]       ご用件は次のどれに近いですか? {choices}")


async def main() -> None:
    # 分類器用の model を 1 つだけ生成して共有し、起動時に warmup で接続と推論経路を温める
    # （下流 dispatch エージェントは別 client のため初回 dispatch は cold のまま）。
    model = azure_model()
    with stopwatch("warmup"):
        await warmup(model, MODEL_SETTINGS)
    registry = build_registry()

    for utt in [
        "先月の請求書のPDFが届かないので送ってほしい",  # 明確 -> dispatch される想定
        "なんかうまくいかない",  # 曖昧 -> 確信度不足で聞き返しになる想定
    ]:
        await classify_then_decide(model, registry, utt)


if __name__ == "__main__":
    asyncio.run(main())
