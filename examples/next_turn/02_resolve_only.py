"""Next-Turn Agent Override の 1 本例: 単体 API（`resolve_next_agent` で自前分岐）。

組み立てヘルパ `next_turn_agent` を使わず、解決結果の名前だけを受け取って分岐する形。
本例では以下を示す:

- `resolve_next_agent(policy, result)` は「上書き先の名前（str）」または
  「上書きなし（None）」だけを返す副作用のない純関数で、registry も結果も変更しない。
- 分岐は利用者コードが書く: 名前があれば `registry.get(name)`、None なら
  `result.last_agent` を次ターンの開始エージェントにする（= SDK 標準の last_agent 継続）。
- `resolve_next_agent` の None は「上書きなし（正常系）」で、`next_turn_agent` の None
  （開始エージェント決定不能）とは意味が異なる。
- 発動条件は AND: 「ターン内にハンドオフ遷移が 1 件以上」かつ「最終回答者名が宣言のキー」。
  本例は `"support": "triage"` の str 略記エントリが triage -> support の到達で発動する。

到達時ハンドオフ禁止を併用する主経路は `01_next_turn_override.py`、禁止のみルールは
`03_prohibit_only.py` を参照。

Azure OpenAI の環境変数（AZURE_OPENAI_*・examples/_shared/_azure.py 参照）を設定して実行:

    uv run python examples/next_turn/02_resolve_only.py

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
    resolve_next_agent,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
from _azure import azure_model  # noqa: E402
from _run_path import print_run_path  # noqa: E402

PROMPT_VARS = {"company": "AgentSpec Inc."}
LAYOUT = PromptLayout(base="base", parts="parts", agents="agents")

POLICY = NextTurnPolicy(
    rules={
        "billing": (
            NextTurnRule(next_agent="triage", no_handoff_on_arrival=True, source="triage"),
            NextTurnRule(no_handoff_on_arrival=True),
        ),
        "support": "triage",
    }
)


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
    graph.edge("billing", "support", description="技術的な内容はサポートへ")
    graph.edge("support", "triage", description="担当外は振り分けへ戻す")
    graph.apply(registry)
    registry.validate()

    return registry


async def main() -> None:
    """`resolve_next_agent` の戻り名で次ターンの開始エージェントを自前で決める。"""
    runtime_registry = apply_next_turn_policy(POLICY, build_registry())

    result = await Runner.run(runtime_registry.get("triage"), input="ログインでエラー E42 が出ます")
    print_run_path(result)

    # str = 上書き先の名前 / None = 上書きなし（正常系。last_agent 継続）。
    name = resolve_next_agent(POLICY, result)
    print(f"--- resolve_next_agent: {name!r} ---")

    agent = runtime_registry.get(name) if name is not None else result.last_agent
    if agent is None:
        print("次ターンの開始エージェントを決定できませんでした")
        return
    print(f"--- 次ターンの開始エージェント: {agent.name} ---")


if __name__ == "__main__":
    asyncio.run(main())
