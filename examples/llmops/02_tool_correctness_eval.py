"""ツール使用の正しさ（tool_correctness）を評価する例（実行トレース捕捉・実 API）。

ツールを持つエージェントを実行し、実際に呼ばれたツール列を捕捉して `expected_tools`
（期待ツール）と DeepEval ToolCorrectnessMetric で決定的に比較する。実行トレース捕捉は
`_adapters` 内で生 RunResult を消費して plain な観測へ変換する（SDK 型を外へ出さない）。

Azure OpenAI の環境変数を設定して実行:
    uv run python examples/llmops/02_tool_correctness_eval.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from oai_agentspec import AgentSpec, function_tool
from oai_agentspec.runtime.llmops import EvalCase, Relevance, ToolUse, evaluate

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
from _azure import azure_model  # noqa: E402


@function_tool
def get_weather(city: str) -> str:
    """指定都市の天気を返す（example 用のダミー実装）。"""
    return f"{city}の天気: 晴れ、22度"


def _print_result(result) -> None:  # noqa: ANN001
    print(f"\n=== 評価対象: {result.target_id} / 統合 verdict: {result.verdict.value} ===")
    for i, case in enumerate(result.cases):
        observed = [t.tool for t in case.observation.tool_calls] if case.observation else []
        print(f"\n[case {i}] input={case.case_input!r} / 観測ツール={observed}")
        print(f"  output: {case.output[:80]}")
        for c in case.criteria:
            score = "" if c.score is None else f" score={c.score:.2f}"
            print(f"  - {c.criterion:18s} {c.status.value:14s}{score}  {c.rationale[:60]}")


async def main() -> None:
    target = AgentSpec(
        name="weather-agent",
        instructions=(
            "あなたは天気アシスタントです。天気を聞かれたら必ず get_weather ツールを"
            "使って調べてから答えてください。"
        ),
        tools=[get_weather],
        model=azure_model(),
    )

    dataset = [
        # 期待ツールを呼ぶべきケース。
        EvalCase("東京の天気を教えて", expected_tools=["get_weather"]),
    ]

    # ToolUse() を criteria に入れたときだけツール使用を評価する（明示・自動付与しない）。
    criteria = [Relevance(), ToolUse()]

    result = await evaluate(target, dataset, judge=azure_model(), criteria=criteria)
    _print_result(result)


if __name__ == "__main__":
    asyncio.run(main())
