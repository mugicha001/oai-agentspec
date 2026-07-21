"""独自 `ContextBuilder` を差し替える例（Protocol DI・実 API）。

`DefaultIntentClassifier` を直接組み立て、`run_context`（ユーザー属性）を素通しする
`UserProfileContextBuilder` を差し込む。prompt callable 側で `context.run_context` を
参照し、LLM に渡す user_content へ反映する。1 行ヘルパ `intent_classifier_from_model`
を経由しないカスタム構成の実例。

Azure OpenAI の環境変数（examples/_shared/_azure.py 参照）を設定して実行:
    uv run python examples/intent/04_custom_context_builder.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from agents import ModelSettings
from openai.types.shared import Reasoning

from oai_agentspec.runtime.intent import (
    DefaultIntentClassifier,
    IntentCategory,
    IntentContext,
    IntentPolicy,
    IntentQuery,
    LLMCandidateGenerator,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
from _timing import stopwatch  # noqa: E402
from _warmup import warmup  # noqa: E402

from _azure import azure_model  # noqa: E402

# reasoning 系モデルの思考トークンを止めて分類レイテンシを最小化する（利用側 DI）。
# 本例は include_rationale_in_prompt=True で rationale も生成させるため、JSON が
# 途中で切れないよう max_tokens は他例 (100) より余裕を持たせる。
# 非 reasoning デプロイでは AZURE_OPENAI_REASONING=0 で reasoning / verbosity を送らない。
if os.environ.get("AZURE_OPENAI_REASONING", "1") != "0":
    MODEL_SETTINGS = ModelSettings(
        reasoning=Reasoning(effort="none"), verbosity="low", max_tokens=300
    )
else:
    MODEL_SETTINGS = ModelSettings(max_tokens=300)


@dataclass(frozen=True)
class UserProfile:
    """利用側の run_context として渡すユーザー属性。"""

    plan: str
    locale: str


class UserProfileContextBuilder:
    """`ContextBuilder` Protocol の実装。run_context をそのまま素通しする。"""

    async def build(self, query: IntentQuery[UserProfile]) -> IntentContext[UserProfile]:
        return IntentContext(
            utterance=query.utterance,
            run_context=query.run_context,
        )


POLICY = IntentPolicy(
    categories=(
        IntentCategory(name="upgrade", description="上位プランへの変更希望"),
        IntentCategory(name="cancel", description="解約希望"),
        IntentCategory(name="support", description="技術的な問い合わせ"),
    ),
    # 判断理由も生成させる（生成トークンと引き換え・既定 False）。結果表示の rationale が埋まる。
    include_rationale_in_prompt=True,
)


def build_prompt(context: IntentContext[UserProfile]) -> str:
    """`context.run_context`（UserProfile）を user_content に反映する。"""
    profile = context.run_context
    header = (
        f"[user_profile] plan={profile.plan} locale={profile.locale}"
        if profile is not None
        else "[user_profile] (unknown)"
    )
    return f"{header}\n\n次の発話を分類してください:\n{context.utterance}"


async def main() -> None:
    # warmup と classifier で同じ model インスタンスを共有する
    # （接続プールが client 単位のため）。
    model = azure_model()
    with stopwatch("warmup"):
        await warmup(model, MODEL_SETTINGS)

    # DefaultIntentClassifier を直接組み立て（Protocol 差し替えの例）。
    classifier = DefaultIntentClassifier(
        context_builder=UserProfileContextBuilder(),
        generator=LLMCandidateGenerator(
            model=model,
            prompt=build_prompt,
            policy=POLICY,
            model_settings=MODEL_SETTINGS,
        ),
    )

    query: IntentQuery[UserProfile] = IntentQuery(
        utterance="もっと使い倒したい。上限を上げられますか?",
        run_context=UserProfile(plan="basic", locale="ja"),
    )

    context = await classifier.context_builder.build(query)
    print("=" * 60)
    print("[SYSTEM] policy.render_prompt()")
    print("=" * 60)
    print(POLICY.render_prompt())
    print()
    print("=" * 60)
    print("[USER] prompt(context) -- build_prompt が context.run_context を反映")
    print("=" * 60)
    print(build_prompt(context))
    print()

    with stopwatch("classify"):
        prediction = await classifier.classify(query)

    print("=" * 60)
    print("[RESULT] profile を踏まえた分類（basic プランなので upgrade が有力）")
    print("=" * 60)
    for i, c in enumerate(prediction.candidates):
        print(f"  #{i + 1} text={c.text} level={c.level.value} rationale={c.rationale}")


if __name__ == "__main__":
    asyncio.run(main())
