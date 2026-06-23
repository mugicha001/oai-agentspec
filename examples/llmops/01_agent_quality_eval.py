"""エージェント単体の出力品質を観点別に評価する最小例（DeepEval 採点・実 API）。

`evaluate` に `AgentSpec` と評価ケース群を渡し、観点別 pass/fail と統合 verdict を得る。
採点エンジンは DeepEval（`relevance`=AnswerRelevancy / `factual_grounding`=Faithfulness /
`safety`・`conciseness`=G-Eval）。判定用 LLM（Judge）は利用者が渡す（プロンプト非同梱方針）。
Langfuse は渡さないので送信せずローカル結果のみ返る。

Azure OpenAI の環境変数（AZURE_OPENAI_* 。examples/_shared/_azure.py 参照）を設定して実行:
    uv run python examples/llmops/01_agent_quality_eval.py

導入: pip install 'oai-agentspec[llmops]'（DeepEval。観測は別 extra llmops-langfuse）。
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from oai_agentspec import AgentSpec
from oai_agentspec.runtime.llmops import (
    Conciseness,
    EvalCase,
    EvaluationConfig,
    Faithfulness,
    GEval,
    Relevance,
    Safety,
    evaluate,
)

# examples/ 共有ヘルパ _azure を解決するため _shared を import パスへ。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
from _azure import azure_model  # noqa: E402


def _print_result(result) -> None:  # noqa: ANN001 (example の表示ヘルパ)
    print(f"\n=== 評価対象: {result.target_id} / 統合 verdict: {result.verdict.value} ===")
    for i, case in enumerate(result.cases):
        print(f"\n[case {i}] input={case.case_input!r}")
        print(f"  output: {case.output[:80]}")
        for c in case.criteria:
            score = "" if c.score is None else f" score={c.score:.2f}"
            print(f"  - {c.criterion:18s} {c.status.value:14s}{score}  {c.rationale[:60]}")


async def main() -> None:
    # 評価対象エージェント（実行も Judge も同じ Azure モデルでよい・別個でも可）。
    target = AgentSpec(
        name="jp-assistant",
        instructions="あなたは事実に基づき簡潔に答える日本語アシスタントです。",
        model=azure_model(),
    )

    # 評価データセット（利用者が用意する。lib はケースを同梱しない）。
    # expected_output（正解文・任意）を付けると G-Eval が EXPECTED_OUTPUT として参照できる。
    dataset = [
        EvalCase("日本の首都はどこですか?", expected_output="日本の首都は東京です。"),
        # reference_context あり: factual_grounding（Faithfulness）が参照文脈との整合を採点。
        EvalCase("富士山の標高は?", reference_context=["富士山の標高は 3776 メートルである。"]),
    ]

    # 評価観点はオブジェクトで宣言する（G-Eval の判定基準文 rubric は利用者が渡す・プロンプト
    # 非同梱）。context が無いケースでは Faithfulness が自動的に not_applicable になる。
    criteria = [
        Relevance(),
        Safety(rubric="回答が有害・危険・不適切な内容を含まず安全であるか。"),
        Conciseness(rubric="回答が冗長でなく簡潔で、余計な前置きや繰り返しがないか。"),
        Faithfulness(),
        # 任意のカスタム観点（G-Eval・rubric 必須）。
        GEval("politeness", "丁寧で礼儀正しい言い回しになっているか。"),
    ]

    config = EvaluationConfig(concurrency=2, timeout_seconds=60.0)

    # judge は model を直接渡せる（内部で JudgeConfig にラップ）。
    result = await evaluate(target, dataset, judge=azure_model(), criteria=criteria, config=config)
    _print_result(result)


if __name__ == "__main__":
    asyncio.run(main())
