"""動的ハンドオフ（dynamic_edge）の例（モデル呼び出しなし）。

通常のエッジは固定 1 ターゲットだが、dynamic_edge は resolver が候補から転送先を
実行時に選ぶ（SDK の Handoff.on_invoke_handoff を内部で生成）。ここでは resolver を
ランダム選択にして、生成された on_invoke_handoff を直接呼び、毎回どの担当へ転送されるかを示す
（本番では Runner.run 中に LLM がこのハンドオフ tool を呼ぶと発火する）。

実行:
    uv run python examples/dynamic_edge.py
"""

from __future__ import annotations

import asyncio
import random
from typing import Any

from oai_agentspec import AgentRegistry, AgentSpec, HandoffGraph

CANDIDATES = ["billing", "support", "sales"]


def route_randomly(context: Any, input_json: Any) -> str:
    """候補からランダムに転送先名を選ぶ resolver（候補内に限る）。"""
    return random.choice(CANDIDATES)


def build_registry() -> tuple[AgentRegistry, HandoffGraph]:
    registry = AgentRegistry()
    registry.register(AgentSpec(name="triage", instructions="振り分け担当。"))
    for name in CANDIDATES:
        registry.register(AgentSpec(name=name, instructions=f"{name} 担当。"))

    graph = HandoffGraph(entry="triage")
    graph.dynamic_edge(
        "triage", CANDIDATES, route_randomly, tool_name="route", description="動的に担当を決定"
    )
    graph.apply(registry)
    registry.validate()  # 候補名の未登録（タイポ）を検出
    return registry, graph


async def main() -> None:
    random.seed(0)  # 表示を再現可能にする
    registry, graph = build_registry()
    print(graph.mermaid())  # 破線で動的エッジが描かれる

    triage = registry.get("triage")
    dynamic = next(h for h in triage.handoffs if getattr(h, "tool_name", "") == "route")

    print("--- on_invoke を 5 回呼んだ転送先 ---")
    for _ in range(5):
        target = await dynamic.on_invoke_handoff(None, None)
        print("  ->", target.name)


if __name__ == "__main__":
    asyncio.run(main())
