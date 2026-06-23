"""承認を自動解決して完了まで採点する例（案A・mock-approve・実 API）。

承認必須ツールを持つエージェントを、**本物の危険な副作用を起こさずに**完了まで評価する。
`tool_mocks`（**agent スコープのネスト dict** `{agent_name: {tool_name: 値}}`）で当該 agent の
当該ツールの実行本体だけをモックに差し替え（`needs_approval` は維持＝ゲートは発火する）、
`approvals` ポリシーで承認を自動解決して resume する。承認後に走るのは本物でなくモックなので
安全に、実プロダクションの HITL 経路（中断→承認→再開）を通して最終応答・ツール使用を採点できる。

安全不変条件: `approvals` が approve を返す承認待ちは、当該 `(agent_name, tool_name)` が
`tool_mocks` に登録され実差し替えされていること。未登録 / 別 agent の同名ツールを approve しようと
すると ValueError（本物の危険ツール実行を構造的に阻止）。`approvals` が False を返す（却下）場合は
ツールを実行せず、却下後の応答を採点できる。resolver に渡る承認待ち dict は `agent_name` を含む。

Azure OpenAI の環境変数を設定して実行:
    uv run python examples/llmops/07_mock_approve_eval.py
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
    return f"REAL deleted account: {user_id}"  # 評価では実行されない（mock に差し替わる）


def _print_result(result) -> None:  # noqa: ANN001
    print(f"\n=== 評価対象: {result.target_id} / 統合 verdict: {result.verdict.value} ===")
    for i, case in enumerate(result.cases):
        tools = [t.tool for t in case.observation.tool_calls] if case.observation else []
        print(f"\n[case {i}] input={case.case_input!r} / 観測ツール={tools}")
        print(f"  output: {case.output[:80]}")
        for c in case.criteria:
            print(f"  - {c.criterion:16s} {c.status.value:14s}  {c.rationale[:60]}")


async def main() -> None:
    target = AgentSpec(
        name="account-agent",
        instructions="アカウント削除を依頼されたら必ず delete_account ツールを使って削除する。",
        tools=[delete_account],
        model=azure_model(),
    )
    dataset = [EvalCase("ユーザー u-123 を削除して", expected_tools=["delete_account"])]

    result = await evaluate(
        target,
        dataset,
        judge=azure_model(),
        criteria=[Relevance(), ToolUse()],
        # 承認ポリシー: delete_account を承認（承認待ち dict は tool_name / agent_name を含む）。
        approvals=lambda pending: pending["tool_name"] == "delete_account",
        # モック（agent スコープ）: 承認しても本物（REAL deleted ...）は走らず、これが返る。
        # approve 認可は (agent_name, tool_name) 単位（同名でも別 agent は認可されない）。
        tool_mocks={"account-agent": {"delete_account": "deleted (mock)"}},
    )
    _print_result(result)


if __name__ == "__main__":
    asyncio.run(main())
