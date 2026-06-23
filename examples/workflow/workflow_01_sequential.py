"""ワークフロー入門 1: 順次（sequential）。

最も基本の形。START -> normalize -> shout -> END とノードを一直線につなぐ。
全ノードが関数なので LLM を呼ばず offline で動く（Runner.run で実行）。

    uv run python examples/workflow_01_sequential.py
"""

from __future__ import annotations

import asyncio

from agents import Runner, set_tracing_disabled

from oai_agentspec import END, START, AgentRegistry, WorkflowGraph

set_tracing_disabled(True)


def build() -> WorkflowGraph:
    wf = WorkflowGraph("sequential")
    # ノードを定義（関数ノード: fn(msg, ctx) -> 出力。msg は上流の出力）
    wf.add_function_node("normalize", fn=lambda msg, ctx: msg.strip().lower())
    wf.add_function_node("shout", fn=lambda msg, ctx: f"{msg}!")
    # エッジを定義（START -> normalize -> shout -> END）
    wf.add_edge(START, "normalize")
    wf.add_edge("normalize", "shout")
    wf.add_edge("shout", END)
    return wf


async def main() -> None:
    wf = build()
    registry = AgentRegistry()
    wf.validate(registry)
    registry.register(wf.as_agent_spec("seq_flow", registry=registry))

    print(wf.mermaid())
    result = await Runner.run(registry.get("seq_flow"), input="  Hello World  ")
    print("output:", result.final_output)  # -> "hello world!"


if __name__ == "__main__":
    asyncio.run(main())
