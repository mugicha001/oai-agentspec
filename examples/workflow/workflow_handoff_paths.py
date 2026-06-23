"""ワークフローへの handoff 流入 3 経路を比較する例（Azure OpenAI）。

題材は「記事作成パイプライン」。AGENT ノード（researcher -> writer）と、その間に
入力を整える FUNCTION ノード（brief）を 1 つ挟んだ node/edge ワークフローを、上流の
triage エージェントから呼び出す 3 経路で構築し、決定性 / context 透過 / LLM 層数 の違いを
示す。

ワークフローのトポロジ（公開 API・node/edge 方式）::

    START -> research[researcher] -> brief(整形) -> compose[writer] -> END

  経路C（as_agent_spec）: ワークフローを「本物の Agent」として registry 登録し、
    triage から handoff の直接ターゲットにする。WorkflowModel が LLM を呼ばずに
    エンジンを回すため流入は決定論的（追加 LLM 層 0）。ただし外側 run の共有 context は
    ワークフロー内ノードへ伝播しない。

  経路A（as_facade_spec）: ワークフローを tool として持つファサード Agent を作り、
    triage から handoff する。外側の共有 context をワークフロー内へ透過できる代わりに、
    流入時にファサードが LLM を 1 回必ず呼ぶ（tool_choice='required' で強制・非決定・
    追加 LLM 層 1）。

  経路B（HandoffGraph.edge で entry 相当へ直接）: ワークフローを介さず、triage から
    ワークフローの先頭 AGENT ノードに対応する素のエージェント（researcher）へ直接
    handoff する。ワークフローのトポロジ（researcher -> brief -> writer）は適用されず、
    以降は通常の handoff 連鎖に委ねられる。最も単純だが「ワークフローとして」の決定論的
    逐次実行は得られない。

違いの要約:
  - 決定性:   C=決定論 / A=非決定（流入 LLM 依存）/ B=非決定（通常 handoff）
  - context:  C=非透過 / A=透過 / B=通常 handoff の context（ワークフロー外）
  - LLM 層数: C=+0    / A=+1（ファサード）/ B=+0（ただしワークフロー実行はされない）

Azure OpenAI の環境変数（AZURE_OPENAI_* 。examples/_azure.py 参照）を設定して実行:
    uv run python examples/workflow_handoff_paths.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

from agents import Runner

from oai_agentspec import (
    END,
    START,
    AgentRegistry,
    AgentSpec,
    HandoffGraph,
    PromptLayout,
    PromptStore,
    WorkflowGraph,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
from _azure import azure_model  # noqa: E402
from _run_path import print_run_path  # noqa: E402

PROMPT_VARS = {"company": "AgentSpec Inc."}
LAYOUT = PromptLayout(base="base", parts="parts", agents="agents")


def build_base_registry() -> AgentRegistry:
    """triage / researcher / writer を登録した共有 registry を作る。"""
    store = PromptStore(Path(__file__).resolve().parent.parent / "prompts", LAYOUT)
    registry = AgentRegistry()
    model = azure_model()
    specs = {
        "triage": ("main", ["style", "safety"]),
        "researcher": ("sub", []),
        "writer": ("sub", []),
    }
    for name, (base, parts) in specs.items():
        registry.register(
            AgentSpec(
                name=name,
                instructions=store.compose(agent=name, base=base, parts=parts, vars=PROMPT_VARS),
                model=model,
            )
        )
    return registry


def make_brief(msg: Any, _ctx: Any) -> str:
    """researcher の出力を writer 向けの執筆ブリーフ文字列に整形する FUNCTION ノード。

    AGENT ノードへ渡す入力は文字列が必要なため、ここで明示的に str 化する
    （暗黙の str 化はライブラリ側では行わない）。

    Args:
        msg: 直前 research ノード（researcher）の最終出力。
        _ctx: 共有 context（本例では未使用）。

    Returns:
        writer への入力にする執筆ブリーフ文字列。
    """
    return f"以下の調査メモをもとに紹介記事を書いてください:\n{msg}"


def build_workflow() -> WorkflowGraph:
    """researcher -> brief(整形) -> writer の node/edge ワークフローを宣言する。

    Returns:
        AGENT ノード 2 つの間に FUNCTION ノードを挟んだ WorkflowGraph。
    """
    wf = WorkflowGraph(name="article")
    wf.add_agent_node("research", agent="researcher")
    wf.add_function_node("brief", fn=make_brief)
    wf.add_agent_node("compose", agent="writer")

    wf.add_edge(START, "research")
    wf.add_edge("research", "brief")
    wf.add_edge("brief", "compose")
    wf.add_edge("compose", END)
    return wf


def build_path_c(registry: AgentRegistry, wf: WorkflowGraph) -> HandoffGraph:
    """経路C: ワークフローを Agent 化して registry 登録し handoff ターゲットにする。"""
    wf.validate(registry)
    # WorkflowModel を据えた AgentSpec（決定論的に起動・LLM 層 +0）。
    registry.register(wf.as_agent_spec("article_workflow", registry=registry))
    graph = HandoffGraph(entry="triage")
    graph.edge("triage", "article_workflow", description="記事作成はワークフローへ")
    graph.apply(registry)
    registry.validate()
    return graph


def build_path_a(registry: AgentRegistry, wf: WorkflowGraph) -> HandoffGraph:
    """経路A: ワークフロー tool を持つファサード Agent を handoff ターゲットにする。"""
    wf.validate(registry)
    graph = HandoffGraph(entry="triage")
    # context 透過の代わりに流入時 LLM 1 回（tool_choice='required'・LLM 層 +1）。
    # connect_as_facade が registry 登録 + triage->facade エッジ結線を行い、handoff エッジに
    # 既定 input_filter（直近 1 件）を載せて流入履歴を有界化する（C-10）。
    # 経路A のファサードは LLM を 1 回呼ぶため実モデルが必須（未注入だと SDK デフォルトの
    # OpenAI クライアントにフォールバックする）。本体はモデル非同梱なので明示注入する。
    wf.connect_as_facade(
        registry,
        graph,
        "article_facade",
        "triage",
        description="記事作成はファサードへ",
        model=azure_model(),
    )
    graph.apply(registry)
    registry.validate()
    return graph


def build_path_b(registry: AgentRegistry) -> HandoffGraph:
    """経路B: ワークフローを介さず先頭 AGENT ノード相当（researcher）へ直接 handoff。"""
    # ワークフローのトポロジは適用されない（通常 handoff・非決定）。
    graph = HandoffGraph(entry="triage")
    graph.edge("triage", "researcher", description="まず調査担当へ（ワークフロー非経由）")
    graph.edge("researcher", "writer", description="調査後に執筆担当へ")
    graph.apply(registry)
    registry.validate()
    return graph


async def _run_one(label: str, registry: AgentRegistry, graph: HandoffGraph) -> None:
    """1 経路を実行して handoff の mermaid と run path を表示する。"""
    print(f"\n===== {label} =====")
    print(graph.mermaid())
    entry = graph.entry_agent(registry)
    result = await Runner.run(entry, input="新製品『麦茶 Pro』の紹介記事を書いて")
    print_run_path(result)
    print(result.final_output)


async def main() -> None:
    # ワークフローのトポロジ（公開 API で宣言した node/edge）を 1 度だけ可視化する。
    print("--- workflow mermaid ---")
    print(build_workflow().mermaid())
    print("------------------------")

    # 経路ごとに独立した registry を用意（handoff トポロジが互いに干渉しないように）。
    reg_c = build_base_registry()
    graph_c = build_path_c(reg_c, build_workflow())

    reg_a = build_base_registry()
    graph_a = build_path_a(reg_a, build_workflow())

    reg_b = build_base_registry()
    graph_b = build_path_b(reg_b)

    await _run_one("経路C: as_agent_spec（決定論・context 非透過・LLM +0）", reg_c, graph_c)
    await _run_one("経路A: as_facade_spec（非決定・context 透過・LLM +1）", reg_a, graph_a)
    await _run_one("経路B: HandoffGraph.edge で entry 直接（ワークフロー非経由）", reg_b, graph_b)


if __name__ == "__main__":
    asyncio.run(main())
