"""ワークフロー入門 7: 経路D（決定論ファサード・context 透過・LLM 0 回）。

経路C（as_agent_spec）はワークフローを 1 Agent 化できるが、外側 run の context を内部ノードへ
渡せない（SDK の Model.get_response が context を受け取れないハード制約・C-11）。経路A
（as_facade_spec の既定 LLM 入口）は context を透過できるが実 LLM を 1 回呼ぶ。

経路D = `as_facade_spec(mode=FacadeMode.DETERMINISTIC)` は、入口に決定論ステートレスモデルを
据えてワークフロー tool を強制発火する。これにより「決定論 + 外側 context 透過 + 実 LLM 0 回」を
同時に満たす。tool 経由のため context が内部の関数ノードまで届く（ctx.context で取り出す）。

実 LLM 0 回の主因は入口の決定論モデル（DeterministicToolCallModel）が LLM を呼ばずに毎回
ワークフロー tool を発火するからで、加えて本例は AGENT ノードも無いため完全に offline で動く
（Runner.run で実行）。経路C との違いは「context が内部ノードへ届くか」（経路C は届かない）。

    uv run python examples/workflow_07_deterministic_context.py
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from agents import Runner, set_tracing_disabled

from oai_agentspec import END, START, AgentRegistry, FacadeMode, WorkflowGraph

set_tracing_disabled(True)


@dataclass
class AppContext:
    """外側 run から渡すアプリ context（ユーザー情報など）。"""

    user_id: str
    locale: str


def greet(msg: str, ctx: Any) -> str:
    # ctx は RunContextWrapper。自分のオブジェクトは ctx.context で得る（経路D は透過する）。
    app: AppContext = ctx.context
    hello = "こんにちは" if app.locale == "ja" else "Hello"
    return f"{hello}, {app.user_id}: {msg}"


def build() -> WorkflowGraph:
    wf = WorkflowGraph("greeting")
    wf.add_function_node("normalize", fn=lambda msg, ctx: msg.strip())
    wf.add_function_node("greet", fn=greet)
    wf.add_edge(START, "normalize")
    wf.add_edge("normalize", "greet")
    wf.add_edge("greet", END)
    return wf


async def main() -> None:
    wf = build()
    registry = AgentRegistry()
    wf.validate(registry)
    # 経路D: 決定論ファサードとして登録（実 LLM を呼ばず context を内部へ透過）。
    registry.register(wf.as_facade_spec("greeting_flow", mode=FacadeMode.DETERMINISTIC))

    print(wf.mermaid())
    print("---")
    agent = registry.get("greeting_flow")

    # 外側 run の context が内部の関数ノード（greet）へ届く。
    r1 = await Runner.run(agent, input="  ようこそ  ", context=AppContext("u_123", "ja"))
    r2 = await Runner.run(agent, input="  welcome  ", context=AppContext("u_999", "en"))
    print("ja:", r1.final_output)  # -> こんにちは, u_123: ようこそ
    print("en:", r2.final_output)  # -> Hello, u_999: welcome


if __name__ == "__main__":
    asyncio.run(main())
