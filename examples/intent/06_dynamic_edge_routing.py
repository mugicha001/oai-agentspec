"""dynamic_edge + intent 分類器による入口ルーティング例（RunContext で分類結果を伝搬・実 API）。

triage は `tool_choice="required"` で `route` ハンドオフ tool の呼び出しを強制され、
引数としてユーザー発話を分類しやすい 1 文にリライトする。resolver（async）はその
リライト文を `intent_classifier_from_model` の分類器に入れ、結果に応じて転送先を決める:

    state = RouteState()                          # run ごとに独立（並行実行でも競合しない）
    Runner.run(entry, input=発話, context=state)
      -> triage: route(utterance=<リライト文>)     # tool_choice="required" で強制
      -> resolver: classifier.classify(リライト文)
           分類候補を state に書き込み（RunContext 経由・SDK の公式データ搬送路）、
           - 先頭候補が certain / high -> そのカテゴリの担当へ
           - 候補なし / 信頼度不足 -> reception（受付）へ
      -> reception の instructions は callable（dynamic instructions）で、
         state の分類候補を読んで「有力候補: ...」を含む指示文を実行時に組み立てる

taxonomy（分類対象・`CATEGORIES`）と routing 候補（転送先・`CANDIDATES`）は分離する。
reception は分類対象ではなく転送先のみ（catch-all を taxonomy に混ぜると LLM が安易に
そこへ逃げて分類品質が下がるため）。reception は分類器が実際に迷った候補だけを提示して
確認質問を返す。

Azure OpenAI の環境変数（examples/_shared/_azure.py 参照）を設定して実行:
    uv run python examples/intent/06_dynamic_edge_routing.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agents import ModelSettings, Runner
from openai.types.shared import Reasoning
from pydantic import BaseModel, Field

from oai_agentspec import AgentRegistry, AgentSpec, HandoffGraph
from oai_agentspec.runtime.intent import (
    ConfidenceLevel,
    IntentCategory,
    IntentPolicy,
    IntentQuery,
    intent_classifier_from_model,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
from _timing import stopwatch  # noqa: E402
from _warmup import warmup  # noqa: E402

from _azure import azure_model  # noqa: E402

# reasoning 系モデルの思考トークンを止めて分類レイテンシを最小化する（利用側 DI・分類器に適用）。
# 非 reasoning デプロイ（gpt-4.1-nano 等）では AZURE_OPENAI_REASONING=0 を設定すると
# reasoning / verbosity パラメータ自体を送らない（未対応モデルでの API エラーを回避）。
if os.environ.get("AZURE_OPENAI_REASONING", "1") != "0":
    MODEL_SETTINGS = ModelSettings(
        reasoning=Reasoning(effort="none"), verbosity="low", max_tokens=100
    )
else:
    MODEL_SETTINGS = ModelSettings(max_tokens=100)

# taxonomy: 分類器が判定する純粋な業務カテゴリのみ（catch-all は入れない）。
CATEGORIES: tuple[IntentCategory, ...] = (
    IntentCategory(name="billing", description="請求・支払いに関する問い合わせ"),
    IntentCategory(name="technical", description="技術的なトラブル・エラーの相談"),
)
POLICY = IntentPolicy(categories=CATEGORIES, max_candidates=3)
_CATEGORY_LINES = "\n".join(f"- {c.name}: {c.description}" for c in CATEGORIES)

# routing: taxonomy + fallback 受付。reception は分類対象ではないが転送先ではある。
FALLBACK = "reception"
CANDIDATES = [c.name for c in CATEGORIES] + [FALLBACK]

# この信頼度以上なら担当へ dispatch、未満なら reception へ（application 側の閾値）。
DISPATCH_LEVELS = {ConfidenceLevel.CERTAIN, ConfidenceLevel.HIGH}


@dataclass
class RouteState:
    """run ごとに生成する共有 state（`Runner.run(context=...)` で渡す）。

    resolver が分類結果を書き込み、reception の dynamic instructions が読む。
    run 単位のスコープなので並行実行でも競合しない。
    """

    candidates: list[dict[str, str]] = field(default_factory=list)


class RouteInput(BaseModel):
    """`route` ハンドオフ tool が LLM に埋めさせる構造化引数。"""

    utterance: str = Field(
        description="ユーザーの依頼内容を、意図分類しやすい 1 文にリライトしたもの。",
    )


def build_resolver(classifier: Any) -> Any:
    """リライト文を intent 分類器へ入れ、分類結果を RouteState に書き込む async resolver。"""

    async def route_by_intent(context: Any, input_json: str) -> str:
        payload = json.loads(input_json) if input_json else {}
        rewritten = payload.get("utterance", "")
        print(f"[REWRITE]    {rewritten}")
        try:
            prediction = await classifier.classify(IntentQuery(utterance=rewritten))
        except ValueError:
            # triage が空 utterance を返した場合（分類対象なし）は crash させず受付へ。
            print("[RESOLVE]    utterance が空 -> reception")
            return FALLBACK

        # RunContextWrapper を開いて run スコープの state に分類結果を書き込む。
        state = context.context
        if isinstance(state, RouteState):
            state.candidates = [
                {"text": c.text, "level": c.level.value} for c in prediction.candidates
            ]

        top = prediction.candidates[0] if prediction.candidates else None
        if top is None:
            # 候補なし: LLM が空を返した / allowlist で全除外された -> 受付へ。
            print("[RESOLVE]    候補なし -> reception")
            return FALLBACK
        if top.level not in DISPATCH_LEVELS:
            # 信頼度不足: 転送はするが担当ではなく受付で聞き返す。
            print(f"[RESOLVE]    信頼度不足 ({top.text} {top.level.value}) -> reception")
            return FALLBACK
        print(f"[RESOLVE]    {top.text} ({top.level.value})")
        return top.text  # allowlist 済みのため必ず taxonomy 内 = CANDIDATES 内。

    return route_by_intent


def reception_instructions(ctx: Any, agent: Any) -> str:
    """reception の dynamic instructions。分類器が迷った候補だけを提示させる。

    `ctx` は RunContextWrapper で、`.context` が `Runner.run(context=...)` に渡した
    RouteState。候補が取れない場合は taxonomy 全件の提示にフォールバックする。
    """
    state = ctx.context
    base = (
        "あなたは総合受付です。ここに来るのは意図を絞り込めなかった問い合わせです。"
        "候補を提示し、どれに近いかを 1 文で確認してください。\n"
    )
    if isinstance(state, RouteState) and state.candidates:
        lines = "\n".join(f"- {c['text']} (確信度: {c['level']})" for c in state.candidates)
        return base + f"意図分類の候補（確信度順）:\n{lines}"
    return base + f"カテゴリ候補:\n{_CATEGORY_LINES}"


def build_registry(model: Any) -> tuple[AgentRegistry, HandoffGraph]:
    registry = AgentRegistry()

    classifier = intent_classifier_from_model(
        model=model,
        prompt=lambda ctx: ctx.utterance,
        policy=POLICY,
        model_settings=MODEL_SETTINGS,
    )

    registry.register(
        AgentSpec(
            name="triage",
            instructions=(
                "あなたは問い合わせの受付担当。必ず `route` tool を呼ぶこと。"
                "utterance 引数には、ユーザーの依頼内容を意図分類しやすい 1 文に"
                "リライトして渡す（分類自体は後段が行うので転送先は考えなくてよい）。"
                "リライト文は、依頼内容の要点を簡潔にまとめ、曖昧な表現を避ける。"
            ),
            model=model,
            # route の呼び出しを強制する（instructions 頼みの判断ブレを排除）。
            model_settings=ModelSettings(tool_choice="required"),
        )
    )
    for cat in CATEGORIES:
        registry.register(
            AgentSpec(
                name=cat.name,
                instructions=f"あなたは{cat.description}の担当です。簡潔に回答します。",
                model=model,
            )
        )
    registry.register(
        AgentSpec(
            name=FALLBACK,
            instructions=reception_instructions,  # callable = 実行時に state から組み立てる
            model=model,
        )
    )

    graph = HandoffGraph(entry="triage")
    graph.dynamic_edge(
        "triage",
        CANDIDATES,
        build_resolver(classifier),
        tool_name="route",
        description="リライトされた依頼内容を意図分類し、担当エージェントを実行時に決定する",
        input_type=RouteInput,
    )
    graph.apply(registry)
    registry.validate()
    return registry, graph


async def main() -> None:
    # warmup と registry 内の全エージェント・分類器で同じ model インスタンスを共有する
    # （接続プールが client 単位のため）。
    model = azure_model()
    with stopwatch("warmup"):
        await warmup(model, MODEL_SETTINGS)

    registry, graph = build_registry(model)
    print("--- handoff graph ---")
    print(graph.mermaid())
    print("---------------------")

    entry = graph.entry_agent(registry)

    for label, utt in [
        ("明確（billing へ遷移する想定）", "先月の請求書のPDFが届かないので送ってほしい"),
        ("曖昧（信頼度不足 -> reception が分類候補を提示する想定）", "なんかうまくいかない"),
    ]:
        print(f"\n=== {label} ===")
        print(f"[UTTERANCE]  {utt}")
        state = RouteState()  # run ごとに新規生成（並行実行でも競合しない）
        with stopwatch("run"):
            result = await Runner.run(entry, input=utt, context=state)
        print(f"[STATE]      candidates={state.candidates}")
        print(f"[LAST AGENT] {result.last_agent.name}")
        print(f"[ANSWER]     {str(result.final_output)[:200]}")


if __name__ == "__main__":
    asyncio.run(main())
