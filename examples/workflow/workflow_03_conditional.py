"""ワークフロー入門 3: 条件分岐（conditional edges）。

router(msg, ctx) -> 判定キー の戻り値を mapping で次ノードへ解決する。
ここでは入力の長さで long / short を分け、別々の整形ノードへ振り分ける。

    uv run python examples/workflow_03_conditional.py
"""

from __future__ import annotations

import asyncio
from typing import Any

from agents import Runner, set_tracing_disabled

from oai_agentspec import END, START, AgentRegistry, WorkflowGraph

set_tracing_disabled(True)


def route(msg: str, ctx: Any) -> str:
    return "long" if len(msg) >= 10 else "short"


def build() -> WorkflowGraph:
    wf = WorkflowGraph("conditional")
    wf.add_function_node("classify", fn=lambda msg, ctx: msg)
    wf.add_function_node("summarize", fn=lambda msg, ctx: f"要約: {msg[:8]}…")
    wf.add_function_node("keep", fn=lambda msg, ctx: f"そのまま: {msg}")

    wf.add_edge(START, "classify")
    # 条件分岐: route の戻りキー -> ノード名
    wf.add_conditional_edges("classify", route, {"long": "summarize", "short": "keep"})
    wf.add_edge("summarize", END)
    wf.add_edge("keep", END)
    return wf


async def main() -> None:
    wf = build()
    registry = AgentRegistry()
    wf.validate(registry)
    registry.register(wf.as_agent_spec("cond_flow", registry=registry))

    print(wf.mermaid())
    for text in ("これは長めの入力テキストです", "短い"):
        result = await Runner.run(registry.get("cond_flow"), input=text)
        print(f"input={text!r} -> {result.final_output}")


if __name__ == "__main__":
    asyncio.run(main())
