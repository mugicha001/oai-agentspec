"""ルーティングの正しさ（handoff_correctness）を横断評価する例（実 API）。

`HandoffGraph`（triage -> billing / tech）を registry（specs 登録済み）とともに渡し、
end-to-end で実行する。実際のルーティング経路を捕捉し、`expected_route`（期待経路）と
決定的に比較する。横断評価は specs を含む registry が必須（HandoffGraph は名前エッジのみ
保持し spec 実体を持たないため）。

Azure OpenAI の環境変数を設定して実行:
    uv run python examples/llmops/03_handoff_route_eval.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from oai_agentspec import AgentRegistry, AgentSpec, HandoffEdge, HandoffGraph
from oai_agentspec.runtime.llmops import EvalCase, HandoffRoute, Relevance, evaluate

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
from _azure import azure_model  # noqa: E402


def _build_registry() -> AgentRegistry:
    registry = AgentRegistry()
    registry.register(
        AgentSpec(
            name="triage",
            instructions=(
                "あなたは振り分け窓口です。請求・支払いの話題なら billing へ、"
                "技術的な不具合なら tech へハンドオフしてください。"
            ),
            model=azure_model(),
        )
    )
    registry.register(
        AgentSpec(
            name="billing",
            instructions="あなたは請求担当です。請求・支払いの質問に簡潔に答えます。",
            model=azure_model(),
        )
    )
    registry.register(
        AgentSpec(
            name="tech",
            instructions="あなたは技術サポートです。技術的な不具合に簡潔に答えます。",
            model=azure_model(),
        )
    )
    return registry


def _print_result(result) -> None:  # noqa: ANN001
    print(f"\n=== 評価対象: {result.target_id} / 統合 verdict: {result.verdict.value} ===")
    for i, case in enumerate(result.cases):
        route = [s.agent for s in case.observation.route.steps] if case.observation else []
        last = case.observation.route.last_agent if case.observation else "?"
        print(f"\n[case {i}] input={case.case_input!r}")
        print(f"  観測経路={route} / last_agent={last}")
        print(f"  output: {case.output[:80]}")
        for c in case.criteria:
            print(f"  - {c.criterion:18s} {c.status.value:14s}  {c.rationale[:60]}")


async def main() -> None:
    registry = _build_registry()
    graph = HandoffGraph(
        entry="triage",
        edges=[HandoffEdge("triage", "billing"), HandoffEdge("triage", "tech")],
    )

    # expected_route は起点を含むフルパスで書く（triage で受けて billing へ handoff = 経由順）。
    dataset = [
        EvalCase("請求書の金額が間違っているのですが", expected_route=["triage", "billing"]),
    ]

    # HandoffRoute() を criteria に入れたときだけルーティングを評価する（横断専用・決定的比較）。
    criteria = [Relevance(), HandoffRoute()]

    result = await evaluate(
        graph, dataset, judge=azure_model(), criteria=criteria, registry=registry
    )
    _print_result(result)


if __name__ == "__main__":
    asyncio.run(main())
