"""ワークフロー入門 6: 条件 fan-out（データ依存で並列対象が変わる）+ 動的 fan-in。

通常の fan-out は「常に全部」並列だが、条件 fan-out は router が
「走らせるノードのリスト」を返すことで、入力に応じて 0〜N 個を動的に並列起動する。
合流（fan-in）は実際に走った枝だけを待ち、merge には {走ったノード名: 出力} の dict が渡る
（走らなかった枝はキーごと現れない）。

題材: 投稿モデレーション。画像があれば image_check、リンクがあれば link_check を走らせる。

    uv run python examples/workflow_06_conditional_fanout.py
"""

from __future__ import annotations

import asyncio
from typing import Any

from agents import Runner, set_tracing_disabled

from oai_agentspec import END, START, AgentRegistry, WorkflowGraph

set_tracing_disabled(True)


def route(msg: str, ctx: Any) -> list[Any]:
    """走らせる検査ノードのリストを返す（0 個なら no_checks 終端で判定文に整形する）。"""
    targets: list[Any] = []
    if "image" in msg:
        targets.append("image_check")
    if "link" in msg:
        targets.append("link_check")
    # 検査対象が無いときも生入力をそのまま返さず、判定文を出す終端へ寄せる（出力の一貫性）。
    return targets or ["no_checks"]


def merge(inputs: dict[str, Any], ctx: Any) -> str:
    # fan-in: inputs は「実際に走った検査だけ」。走らなかった検査はキーごと無い。
    ran = ", ".join(f"{name}={result}" for name, result in sorted(inputs.items()))
    return f"判定: {ran}"


def build() -> WorkflowGraph:
    wf = WorkflowGraph("moderation")
    wf.add_function_node("classify", fn=lambda msg, ctx: msg)
    wf.add_function_node("image_check", fn=lambda msg, ctx: "OK")
    wf.add_function_node("link_check", fn=lambda msg, ctx: "安全")
    wf.add_function_node("merge", fn=merge)
    # 検査対象 0 件のときの終端（判定文に整形して一貫した出力にする）。
    # no_checks は fan-in(merge) を通らず単独で END へ抜ける終端枝。merge は実際に走った
    # 検査枝（activated なソース）だけを待つため、検査が 0 件のこの枝は merge に合流しない。
    wf.add_function_node("no_checks", fn=lambda msg, ctx: "判定: 検査対象なし")

    wf.add_edge(START, "classify")
    # 条件 fan-out: route が返したノードだけを並行起動。candidates で可能な行き先を宣言
    # （validate の到達性・mermaid 可視化に使う。route は動的に部分集合を返す）。
    wf.add_conditional_edges(
        "classify", route, candidates=["image_check", "link_check", "no_checks"]
    )
    # 動的 fan-in: 実際に走った枝だけ待って合流（合流先は FUNCTION 必須）
    wf.add_fan_in_edge(["image_check", "link_check"], "merge")
    wf.add_edge("merge", END)
    wf.add_edge("no_checks", END)
    return wf


async def main() -> None:
    wf = build()
    registry = AgentRegistry()
    wf.validate(registry)
    registry.register(wf.as_agent_spec("moderation_flow", registry=registry))

    print(wf.mermaid())
    print("---")
    for text in (
        "image付きの投稿",
        "link付きの投稿",
        "imageもlinkもある投稿",
        "テキストだけの投稿",
    ):
        result = await Runner.run(registry.get("moderation_flow"), input=text)
        print(f"input={text!r} -> {result.final_output}")


if __name__ == "__main__":
    asyncio.run(main())
