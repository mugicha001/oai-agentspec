"""HITL（承認必須ツール）で中断した実行が評価でどう扱われるかを示す例（実 API）。

承認必須ツール（`function_tool(needs_approval=True)`）を持つエージェントを評価すると、
エージェントがそのツールを呼んだ時点で実行が**承認待ちで中断**する（最終出力が出ない）。
評価器はこの中断を採点完了と誤認せず、**全観点を inconclusive に倒し統合 verdict を fail** に
する（中断を顕在化する）。出力非依存の観点（tool_correctness 等）が途中経路の一致で誤って
pass するのを防ぐためで、これが意図した安全側の挙動。

重要: 評価器は承認を**自動承認しない**（無人評価で本物の危険ツールを実行させない安全設計）。
承認を自動解決して完了まで採点する HITL 評価（mock によるツール差し替え等）は本ライブラリの
別機能として扱う（本 example のスコープ外）。

Azure OpenAI の環境変数を設定して実行:
    uv run python examples/llmops/05_hitl_interrupted_eval.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from oai_agentspec import AgentSpec, function_tool
from oai_agentspec.runtime.llmops import EvalCase, Relevance, ToolUse, evaluate

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
from _azure import azure_model  # noqa: E402


@function_tool(needs_approval=True)
def delete_account(user_id: str) -> str:
    """指定ユーザーのアカウントを削除する（危険操作・承認必須・example 用ダミー）。"""
    return f"deleted account: {user_id}"


def _print_result(result) -> None:  # noqa: ANN001
    print(f"\n=== 評価対象: {result.target_id} / 統合 verdict: {result.verdict.value} ===")
    for i, case in enumerate(result.cases):
        print(f"\n[case {i}] input={case.case_input!r}")
        print(f"  output: {case.output!r}")  # 中断時は空文字
        for c in case.criteria:
            print(f"  - {c.criterion:18s} {c.status.value:14s}  {c.rationale[:60]}")


async def main() -> None:
    target = AgentSpec(
        name="account-agent",
        instructions=(
            "あなたはアカウント管理アシスタントです。削除を依頼されたら必ず delete_account "
            "ツールを使って削除してください。"
        ),
        tools=[delete_account],
        model=azure_model(),
    )

    dataset = [
        EvalCase("ユーザー u-123 のアカウントを削除して", expected_tools=["delete_account"]),
    ]

    # 承認必須ツールを呼ぶと実行が中断する。評価は中断を inconclusive に倒し verdict=fail にする
    # （中断を採点完了と誤認しない）。承認の自動解決はしない（安全設計）。
    result = await evaluate(target, dataset, judge=azure_model(), criteria=[Relevance(), ToolUse()])
    _print_result(result)

    if result.verdict.value == "fail":
        print("\n承認待ちで中断したため verdict=fail（期待どおり・中断が顕在化）。")


if __name__ == "__main__":
    asyncio.run(main())
