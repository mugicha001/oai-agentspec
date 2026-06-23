"""承認ゲート発火を APO 報酬に使う例（`approval_match` reward・実 API）。

「危険ツールを正しく人間承認のゲートへ回す」プロンプトを APO で学習させる。報酬は
`approval_match()`（既定 field=expected_approvals）で、観測した承認ゲート発火集合
（`RolloutResult.fired_approvals`）が `OptimizeCase.expected_approvals` を recall すれば 1.0。
`tool_match` が承認後に実行されたツールを見るのに対し、`approval_match` は承認待ちに出たか自体を
見るため、危険操作を「実行する前に必ず承認に回す」プロンプト挙動を直接学習できる（llmops
`ApprovalGate` の対応物）。

approvals + tool_mocks を併用して中断 -> 承認自動解決 -> 再開を安全に通すことで、APO の多数回
rollout 中も本物の危険操作は一切実行されず、ゲート発火集合は全ラウンドで観測される。
`approvals` / `tool_mocks` を渡さない場合は初回中断時の pending のみが fired_approvals に積まれ、
rollout は中断のまま採点される（こちらでも `approval_match` で評価可能）。

NOTE: APO は内部で追加 LLM（textual gradient + prompt edit）を使うため、利用者は `apo_client=` で
AsyncOpenAI 互換クライアントを直接渡す。検証データ `val` は必須。

Azure OpenAI の環境変数を設定して実行:
    uv run python examples/lightning/06_approval_match_apo.py

導入: pip install 'oai-agentspec[lightning]'
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from oai_agentspec import AgentSpec, function_tool
from oai_agentspec.runtime.lightning import (
    OptimizeCase,
    approval_match,
    optimize,
    train_val_split,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
from _azure import azure_client, azure_model  # noqa: E402


@function_tool(needs_approval=True)
def delete_account(user_id: str) -> str:
    """指定ユーザーのアカウントを削除する（危険操作・承認必須・example 用ダミー）。"""
    return f"REAL deleted account: {user_id}"  # 最適化中は mock に差し替わり実行されない。


@function_tool(needs_approval=True)
def archive_account(user_id: str) -> str:
    """指定ユーザーのアカウントをアーカイブする（危険操作・承認必須・example 用ダミー）。"""
    return f"REAL archived account: {user_id}"  # 最適化中は mock に差し替わり実行されない。


async def main() -> None:
    target = AgentSpec(
        name="account-agent",
        instructions=(
            "ユーザーの依頼に応じてアカウント操作を実行する。"
            "削除なら delete_account、アーカイブなら archive_account を使う。"
        ),
        tools=[delete_account, archive_account],
        model=azure_model(),
    )

    # 期待承認ゲート（`needs_approval=True` のツール）を ground truth に持つデータセット
    # （OptimizeCase.expected_approvals）。approval_match() は既定 field=expected_approvals で
    # fired_approvals の recall を採点する（承認待ちに出たか自体を見る）。
    data = [
        OptimizeCase(input="ユーザー u-123 を削除して", expected_approvals=["delete_account"]),
        OptimizeCase(input="u-456 をアーカイブしたい", expected_approvals=["archive_account"]),
        OptimizeCase(input="u-789 を消去お願い", expected_approvals=["delete_account"]),
        OptimizeCase(input="u-999 を保管庫へ移して", expected_approvals=["archive_account"]),
    ]
    train, val = train_val_split(data, val_ratio=0.25, seed=0)

    result = await optimize(
        target,
        train=train,
        val=val,
        reward=approval_match(),  # 既定 field=expected_approvals。承認ゲート発火集合の recall。
        # 承認を自動解決（両ツールを承認）。承認待ち dict は tool_name / agent_name を含む。
        approvals=lambda pending: pending["tool_name"] in {"delete_account", "archive_account"},
        # agent スコープのモック: 承認しても本物（REAL ...）は走らず、これが返る。
        # approve 認可は (agent_name, tool_name) 単位（未登録ツールの approve は失敗）。
        tool_mocks={
            "account-agent": {
                "delete_account": "deleted (mock)",
                "archive_account": "archived (mock)",
            }
        },
        apo_client=azure_client(),
        # E2E 動作確認用に最小 APO 設定（1 ラウンド・1 候補）。本番では rounds / beam を増やす。
        rounds=1,
        apo_beam_width=1,
        apo_branch_factor=1,
    )

    print(f"train_score={result.train_score:.3f} | val_score={result.val_score}")
    print("--- before（rollout 時の合成済み instructions）---")
    print(result.seed)
    print("--- after（最適化済みプロンプト）---")
    print(result.prompt)
    print("--- diff（unified diff）---")
    print(result.diff or "(no change)")


if __name__ == "__main__":
    asyncio.run(main())
