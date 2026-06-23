"""サブエージェント（agent as tool）でオーケストレーションする例（Azure OpenAI）。

orchestrator が researcher / writer をツールとして呼び、結果を受け取って自分で
最終応答を組み立てる（handoff と異なり制御がメインへ戻る）。各 instructions は
PromptStore.compose で base + agent を合成して生成する。

Azure OpenAI の環境変数（AZURE_OPENAI_* 。examples/_azure.py 参照）を設定して実行:
    uv run python examples/sub_agents.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from agents import Runner

from oai_agentspec import AgentRegistry, AgentSpec, PromptLayout, PromptStore

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
from _azure import azure_model  # noqa: E402
from _run_path import print_run_path  # noqa: E402

PROMPT_VARS = {"company": "AgentSpec Inc."}
LAYOUT = PromptLayout(base="base", parts="parts", agents="agents")


def build_registry() -> AgentRegistry:
    store = PromptStore(Path(__file__).resolve().parent.parent / "prompts", LAYOUT)
    registry = AgentRegistry()
    model = azure_model()

    # サブエージェント（base="sub" 共通ベース + agents/<name>.md）。
    for name in ("researcher", "writer"):
        registry.register(
            AgentSpec(
                name=name,
                instructions=store.compose(agent=name, base="sub", vars=PROMPT_VARS),
                model=model,
            )
        )

    # メイン（base="main" + agents/orchestrator.md）。sub_agents が as_tool 注入される。
    registry.register(
        AgentSpec(
            name="orchestrator",
            instructions=store.compose(agent="orchestrator", base="main", vars=PROMPT_VARS),
            sub_agents=["researcher", "writer"],
            model=model,
        )
    )
    registry.validate()  # sub_agents 参照のタイポを run 前に検出
    return registry


async def main() -> None:
    registry = build_registry()
    orchestrator = registry.get("orchestrator")

    print("--- orchestrator tools (as_tool) ---")
    print([tool.name for tool in orchestrator.tools])

    result = await Runner.run(orchestrator, input="新製品の紹介記事を書いて")
    print_run_path(result)  # サブエージェント（as tool）の呼び出しが tool_call として現れる
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
