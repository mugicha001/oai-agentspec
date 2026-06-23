"""相互ハンドオフ（循環）を解決する例（Azure OpenAI）。

triage -> support と support -> triage の双方向エッジ（A <-> B の循環）を宣言する。
素朴に構築すると「相手がまだ存在しない」ため詰むが、AgentRegistry は局所 2 パスの
遅延バインド（handoffs 空で全エージェントを構築 -> 後から相互に結線）で循環を解決する。

Azure OpenAI の環境変数（AZURE_OPENAI_* 。examples/_azure.py 参照）を設定して実行:
    uv run python examples/cyclic_handoff.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from agents import Runner

from oai_agentspec import AgentRegistry, AgentSpec, HandoffGraph, PromptLayout, PromptStore

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
from _azure import azure_model  # noqa: E402
from _run_path import print_run_path  # noqa: E402

PROMPT_VARS = {"company": "AgentSpec Inc."}
LAYOUT = PromptLayout(base="base", parts="parts", agents="agents")


def build_registry() -> tuple[AgentRegistry, HandoffGraph]:
    store = PromptStore(Path(__file__).resolve().parent.parent / "prompts", LAYOUT)
    registry = AgentRegistry()
    model = azure_model()

    for name in ("triage", "support"):
        registry.register(
            AgentSpec(
                name=name,
                instructions=store.compose(
                    agent=name, base="main", parts=["style", "safety"], vars=PROMPT_VARS
                ),
                model=model,
            )
        )

    # 双方向エッジ = 循環。triage <-> support。
    graph = HandoffGraph(entry="triage")
    graph.edge("triage", "support", description="技術的な問い合わせはサポートへ")
    graph.edge("support", "triage", description="技術以外は振り分けへ戻す")
    graph.apply(registry)
    registry.validate()

    return registry, graph


async def main() -> None:
    registry, graph = build_registry()
    print("--- handoff graph（循環）---")
    print(graph.mermaid())
    print("-------------------------")

    # 2 パス遅延バインドにより、両者が互いへのハンドオフ tool を持つ。
    triage = registry.get("triage")
    support = registry.get("support")
    print("triage の handoffs:", [h.tool_name for h in triage.handoffs])
    print("support の handoffs:", [h.tool_name for h in support.handoffs])

    result = await Runner.run(triage, input="ログインできません。エラーコードは E42 です")
    print_run_path(result)  # triage -> support のハンドオフが handoff として現れる
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
