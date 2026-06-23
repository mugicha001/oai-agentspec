"""RunContext に応じて instructions を動的生成する例（Azure OpenAI）。

`PromptStore.compose` に `vars` を callable（ctx -> dict）で渡すと 2 引数 callable が
返り、各 run で context に応じて agents/concierge.md がレンダリングされる。

Azure OpenAI の環境変数（AZURE_OPENAI_* 。examples/_azure.py 参照）を設定して実行:
    uv run python examples/dynamic_context.py
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path

from agents import RunContextWrapper, Runner

from oai_agentspec import AgentRegistry, AgentSpec, PromptLayout, PromptStore

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
from _azure import azure_model  # noqa: E402
from _run_path import print_run_path  # noqa: E402

LAYOUT = PromptLayout(base="base", parts="parts", agents="agents")


@dataclass
class SupportContext:
    """利用側が定義する任意のコンテキスト型。"""

    user_name: str
    plan: str


def extract_vars(context: RunContextWrapper[SupportContext]) -> dict[str, str]:
    """RunContextWrapper から agents/concierge.md の ${var} へ渡す値を抽出する。

    SDK が instructions callable に渡すのは RunContextWrapper であり、利用側の
    context オブジェクトは `context.context` で取り出す。
    """
    ctx = context.context
    return {
        "tier": "VIP" if ctx.plan == "premium" else "標準",
        "user_name": ctx.user_name,
    }


def build_registry() -> AgentRegistry:
    store = PromptStore(Path(__file__).resolve().parent.parent / "prompts", LAYOUT)
    registry = AgentRegistry()
    registry.register(
        AgentSpec(
            name="concierge",
            instructions=store.compose(agent="concierge", vars=extract_vars),
            model=azure_model(),
        )
    )
    return registry


async def main() -> None:
    registry = build_registry()
    concierge = registry.get("concierge")

    for ctx in (
        SupportContext(user_name="Mugi", plan="premium"),
        SupportContext(user_name="Cha", plan="free"),
    ):
        result = await Runner.run(concierge, input="状況を教えてください", context=ctx)
        print(f"=== {ctx.user_name} ({ctx.plan}) ===")
        print_run_path(result)
        print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
