"""ワークフロー入門 2: 並列（fan-out + fan-in）。

split から 2 ノードへ分岐（fan-out）して並行実行し、merge で合流（fan-in）する。
fan-in の合流先（merge）は FUNCTION ノードで、入力 msg は {ソースノード名: 出力} の dict。

    uv run python examples/workflow_02_parallel.py
"""

from __future__ import annotations

import asyncio
from typing import Any

from agents import Runner, set_tracing_disabled

from oai_agentspec import END, START, AgentRegistry, WorkflowGraph

set_tracing_disabled(True)


def count_chars(msg: str, ctx: Any) -> int:
    return len(msg)


def count_words(msg: str, ctx: Any) -> int:
    return len(msg.split())


def merge(inputs: dict[str, Any], ctx: Any) -> str:
    # fan-in の合流先: inputs = {"chars": 11, "words": 2}
    return f"chars={inputs['chars']} words={inputs['words']}"


def build() -> WorkflowGraph:
    wf = WorkflowGraph("parallel")
    wf.add_function_node("split", fn=lambda msg, ctx: msg)  # そのまま下流へ
    wf.add_function_node("chars", fn=count_chars)
    wf.add_function_node("words", fn=count_words)
    wf.add_function_node("merge", fn=merge)

    wf.add_edge(START, "split")
    # fan-out: split から 2 本のエッジ -> chars / words が並行実行
    wf.add_edge("split", "chars")
    wf.add_edge("split", "words")
    # fan-in: 両方の完了を待って merge へ（merge は dict を受ける）
    wf.add_fan_in_edge(["chars", "words"], "merge")
    wf.add_edge("merge", END)
    return wf


async def main() -> None:
    wf = build()
    registry = AgentRegistry()
    wf.validate(registry)
    registry.register(wf.as_agent_spec("parallel_flow", registry=registry))

    print(wf.mermaid())
    result = await Runner.run(registry.get("parallel_flow"), input="hello world")
    print("output:", result.final_output)  # -> "chars=11 words=2"


if __name__ == "__main__":
    asyncio.run(main())
