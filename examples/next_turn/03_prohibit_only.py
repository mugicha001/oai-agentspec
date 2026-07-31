"""Next-Turn Agent Override の 1 本例: 禁止のみルール（たらい回しの遮断だけを行う）。

次ターン指定（`next_agent`）を持たず到達時ハンドオフ禁止だけを宣言する形。本例では
以下を示す:

- `NextTurnRule(no_handoff_on_arrival=True)` のみのルールは、ハンドオフで当該エージェント
  へ到達したターンでその全 handoff を無効化し、自分で回答を終えさせる（たらい回しの遮断）。
- 次ターン指定を持たないため `resolve_next_agent` は常に None（上書きなし）を返す。
  「禁止は働いたのに上書きされない」のは仕様で、次ターンは SDK 標準の last_agent 継続になる。
- `next_turn_agent` は上書きなしのとき `result.last_agent` をそのまま返す（registry を
  経由した正規化はしない）。
- 到達記録は run 内の一時状態なので次ターンには持ち越されず、次ターンの当該エージェントは
  元の handoff 構成で動く。
- 禁止はハンドオフによる到達でのみ発動する。同じエージェントをターン開始エージェントとして
  直接使うターンでは発動しない。

次ターン指定を伴う主経路は `01_next_turn_override.py`、単体 API での自前分岐は
`02_resolve_only.py` を参照。

Azure OpenAI の環境変数（AZURE_OPENAI_*・examples/_shared/_azure.py 参照）を設定して実行:

    uv run python examples/next_turn/03_prohibit_only.py

本例は実 API を呼ぶ（合計 1 回）。
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from agents import Runner

from oai_agentspec import (
    AgentRegistry,
    AgentSpec,
    HandoffGraph,
    NextTurnPolicy,
    NextTurnRule,
    PromptLayout,
    PromptStore,
    apply_next_turn_policy,
    next_turn_agent,
    resolve_next_agent,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
from _azure import azure_model  # noqa: E402
from _run_path import print_run_path  # noqa: E402

PROMPT_VARS = {"company": "AgentSpec Inc."}
LAYOUT = PromptLayout(base="base", parts="parts", agents="agents")

# 次ターン指定を持たない = 上書きはせず、到達時ハンドオフ禁止だけを行う宣言。
POLICY = NextTurnPolicy(rules={"billing": NextTurnRule(no_handoff_on_arrival=True)})


def build_registry() -> AgentRegistry:
    """triage <-> billing / support のハンドオフを持つ registry を組む。

    Returns:
        登録済み registry。
    """
    store = PromptStore(Path(__file__).resolve().parent.parent / "prompts", LAYOUT)
    registry = AgentRegistry()
    model = azure_model()

    for name in ("triage", "billing", "support"):
        registry.register(
            AgentSpec(
                name=name,
                instructions=store.compose(
                    agent=name, base="main", parts=["style", "safety"], vars=PROMPT_VARS
                ),
                model=model,
            )
        )

    graph = HandoffGraph(entry="triage")
    graph.edge("triage", "billing", description="請求・支払いの問い合わせは請求担当へ")
    graph.edge("triage", "support", description="技術的な問い合わせはサポートへ")
    # billing の出辺。billing へハンドオフで到達したターンだけ、この辺が無効化される。
    graph.edge("billing", "support", description="技術的な内容はサポートへ")
    graph.edge("support", "triage", description="担当外は振り分けへ戻す")
    graph.apply(registry)
    registry.validate()

    return registry


async def main() -> None:
    """禁止のみルール: たらい回しを止め、次ターンは last_agent 継続のままにする。"""
    runtime_registry = apply_next_turn_policy(POLICY, build_registry())

    result = await Runner.run(
        runtime_registry.get("triage"), input="請求の内訳と、ついでに障害情報も知りたいです"
    )
    # billing へハンドオフで到達した場合、このターンは billing -> support へ回せず自分で答える。
    print_run_path(result)
    print(f"--- resolve_next_agent: {resolve_next_agent(POLICY, result)!r}（上書きなし）---")

    # 上書きなしのため last_agent がそのまま返る。禁止は run を跨がないため、次ターンの
    # 当該エージェントは元の handoff 構成で動く。
    agent = next_turn_agent(POLICY, result, runtime_registry)
    if agent is None:
        print("次ターンの開始エージェントを決定できませんでした")
        return
    print(f"--- 次ターンの開始エージェント: {agent.name} ---")


if __name__ == "__main__":
    asyncio.run(main())
