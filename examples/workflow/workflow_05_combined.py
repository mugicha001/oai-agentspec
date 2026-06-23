"""ワークフロー入門 5: 組み合わせ（並列 + 合流 + 条件分岐）。

実際のワークフローは複数パターンの組み合わせになる。ここでは
「並列で集計 -> 合流 -> 結果で条件分岐」を 1 つのグラフにまとめる。

    uv run python examples/workflow_05_combined.py
"""

from __future__ import annotations

import asyncio
from typing import Any

from agents import Runner, set_tracing_disabled

from oai_agentspec import END, START, AgentRegistry, WorkflowGraph

set_tracing_disabled(True)


def summarize(inputs: dict[str, Any], ctx: Any) -> dict[str, int]:
    # fan-in 合流: {"chars": .., "words": ..} を 1 つにまとめる
    return {"chars": inputs["chars"], "words": inputs["words"]}


def route(stats: dict[str, int], ctx: Any) -> str:
    return "big" if stats["chars"] >= 10 else "small"


def build() -> WorkflowGraph:
    wf = WorkflowGraph("combined")
    wf.add_function_node("ingest", fn=lambda msg, ctx: msg.strip())
    # 並列で 2 つ集計（fan-out）
    wf.add_function_node("chars", fn=lambda msg, ctx: len(msg))
    wf.add_function_node("words", fn=lambda msg, ctx: len(msg.split()))
    # 合流（fan-in）
    wf.add_function_node("summarize", fn=summarize)
    # 条件分岐の各ハンドラ
    wf.add_function_node("flag_big", fn=lambda s, ctx: f"大きい入力です ({s})")
    wf.add_function_node("flag_small", fn=lambda s, ctx: f"小さい入力です ({s})")

    wf.add_edge(START, "ingest")
    wf.add_edge("ingest", "chars")
    wf.add_edge("ingest", "words")
    wf.add_fan_in_edge(["chars", "words"], "summarize")
    wf.add_conditional_edges("summarize", route, {"big": "flag_big", "small": "flag_small"})
    wf.add_edge("flag_big", END)
    wf.add_edge("flag_small", END)
    return wf


async def main() -> None:
    wf = build()
    registry = AgentRegistry()
    wf.validate(registry)
    registry.register(wf.as_agent_spec("combined_flow", registry=registry))

    print(wf.mermaid())
    for text in ("これはそこそこ長い入力テキストです", "短い文"):
        result = await Runner.run(registry.get("combined_flow"), input=text)
        print(f"input={text!r} -> {result.final_output}")


if __name__ == "__main__":
    asyncio.run(main())
