"""ワークフロー入門 4: ループ（エッジ + 条件 + recursion_limit）。

専用の loop API は無い。ノードへ戻るエッジと条件分岐で繰り返しを表し、
無限ループは recursion_limit（既定 25・超過で実行時エラー）で防ぐ。
ここでは末尾に "*" を 3 個付くまで tick を繰り返す。

    uv run python examples/workflow_04_loop.py
"""

from __future__ import annotations

import asyncio
from typing import Any

from agents import Runner, set_tracing_disabled

from oai_agentspec import END, START, AgentRegistry, WorkflowGraph

set_tracing_disabled(True)


def tick(msg: str, ctx: Any) -> str:
    return msg + "*"


def again(msg: str, ctx: Any) -> str:
    # 継続条件: "*" が 3 個未満なら tick へ戻る、満ちたら END
    return "loop" if msg.count("*") < 3 else "done"


def build() -> WorkflowGraph:
    wf = WorkflowGraph("loop", recursion_limit=10)
    wf.add_function_node("tick", fn=tick)
    wf.add_edge(START, "tick")
    # tick の後で条件判定: 未達なら tick へ戻る（ループ）、達したら END
    wf.add_conditional_edges("tick", again, {"loop": "tick", "done": END})
    return wf


async def main() -> None:
    wf = build()
    registry = AgentRegistry()
    wf.validate(registry)
    registry.register(wf.as_agent_spec("loop_flow", registry=registry))

    print(wf.mermaid())
    result = await Runner.run(registry.get("loop_flow"), input="x")
    print("output:", result.final_output)  # -> "x***"


if __name__ == "__main__":
    asyncio.run(main())
