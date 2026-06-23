"""triage / billing / support を Azure OpenAI で宣言的に組み立てる例。

合成済み instructions を `AgentSpec.instructions` に渡す（SDK の Agent と同じ使い心地）。

Azure OpenAI の環境変数（AZURE_OPENAI_* 。examples/_azure.py 参照）を設定して実行:
    uv run python examples/basic.py
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
    graph.edge("triage", "billing", description="請求関連")
    graph.edge("triage", "support", description="技術問い合わせ")
    graph.apply(registry)
    registry.validate()  # handoffs 参照のタイポを run 前に検出

    return registry, graph


async def main() -> None:
    registry, graph = build_registry()
    print("--- handoff graph ---")
    print(graph.mermaid())
    print("---------------------")

    entry = graph.entry_agent(registry)
    result = await Runner.run(entry, input="先月の請求書のPDFが欲しいです")
    print_run_path(result)  # どのエージェントを経由して回答に至ったか
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
