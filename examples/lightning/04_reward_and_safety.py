"""報酬ファクトリの使い分けと、危険ツールの副作用を反復させない安全 rollout（実 API）。

APO は同一 rollout を多数回実行するため、承認必須ツール（`function_tool(needs_approval=True)`）を
持つエージェントをそのまま回すと本物の副作用が反復する恐れがある。`optimize` に `tool_mocks=` /
`approvals=` を渡すと、承認を自動解決しつつツール実行だけを安全なモックへ差し替え、HITL 経路
（中断 -> 承認 -> 再開）を安全に通して報酬を計算できる（llmops の評価経路と同じ安全機構を再利用）。

安全不変条件: `approvals` が approve を返す承認待ちは、当該 `(agent_name, tool_name)` が
`tool_mocks` に登録され実差し替えされていること。未登録のツールを approve しようとすると最適化は
失敗（本物の危険ツール実行を構造的に阻止し、`OptimizeError` に倒れる）。

報酬は利用者供給。代表的なファクトリ（既定 `field` は `OptimizeCase` の標準フィールド名）:
    contains()           expected_output が出力に含まれれば 1.0（field=expected_output 既定）
    exact()              expected_output と完全一致なら 1.0
    tool_match()         expected_tools が全て呼ばれていれば 1.0（recall）
    approval_match()     expected_approvals が全て発火していれば 1.0（recall）
    route_match()        expected_route と route_steps が完全一致なら 1.0
    last_agent_match()   expected_last_agent と last_agent が一致なら 1.0
    judge(rubric, model) 利用者 Judge モデルで 0.0..1.0 採点
手書きの `(RolloutResult) -> float` 関数もそのまま渡せる。`approval_match` は
`RolloutResult.fired_approvals`（中断時に承認ゲートが発火したツール名集合）の recall を見るため、
「危険ツールを正しく承認ゲートへ回せたか」を APO で学習させたいときに使う（`tool_match` が承認後の
実行ツールを見るのに対し、`approval_match` は承認待ちに出たか自体を見る）。

APO は内部で追加 LLM（textual gradient + prompt edit）を使うため、利用者は `apo_client=` で
AsyncOpenAI 互換クライアントを直接渡す。検証データ `val` は必須。

Azure OpenAI の環境変数を設定して実行:
    uv run python examples/lightning/04_reward_and_safety.py

導入: pip install 'oai-agentspec[lightning]'
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from oai_agentspec import AgentSpec, function_tool
from oai_agentspec.runtime.lightning import OptimizeCase, optimize, tool_match, train_val_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
from _azure import api_style, azure_client, azure_deployment, azure_model  # noqa: E402


@function_tool(needs_approval=True)
def delete_account(user_id: str) -> str:
    """指定ユーザーのアカウントを削除する（危険操作・承認必須・example 用ダミー）。"""
    return f"REAL deleted account: {user_id}"  # 最適化では実行されない（mock に差し替わる）


async def main() -> None:
    target = AgentSpec(
        name="account-agent",
        instructions="アカウント削除を依頼されたら必ず delete_account ツールを使って削除する。",
        tools=[delete_account],
        model=azure_model(),
    )

    # 期待ツールを ground truth に持つデータセット（OptimizeCase.expected_tools）。
    # tool_match() は既定 field=expected_tools で recall を採点する。
    data = [
        OptimizeCase(input="ユーザー u-123 を削除して", expected_tools=["delete_account"]),
        OptimizeCase(input="アカウント u-999 を消したい", expected_tools=["delete_account"]),
        OptimizeCase(input="u-555 のアカウントをクローズ", expected_tools=["delete_account"]),
        OptimizeCase(input="u-777 を解約してください", expected_tools=["delete_account"]),
    ]
    train, val = train_val_split(data, val_ratio=0.25, seed=0)

    result = await optimize(
        target,
        train=train,
        val=val,
        reward=tool_match(),  # 既定 field=expected_tools。recall で 1.0。
        # 承認を自動解決（delete_account を承認）。承認待ち dict は tool_name / agent_name を含む。
        approvals=lambda pending: pending["tool_name"] == "delete_account",
        # agent スコープのモック: 承認しても本物（REAL deleted ...）は走らず、これが返る。
        # approve 認可は (agent_name, tool_name) 単位（未登録ツールの approve は失敗）。
        tool_mocks={"account-agent": {"delete_account": "deleted (mock)"}},
        apo_client=azure_client(),
        # APO の gradient / apply-edit 用モデルは rollout と同じものへ明示的に揃える
        # （既定 gpt-5.4-mini はプロバイダ / ゲートウェイによっては存在しないため）。
        apo_gradient_model=azure_deployment(),
        apo_apply_edit_model=azure_deployment(),
        # gradient / apply-edit の API はプロバイダ設定（OPENAI_API_STYLE）に揃えて明示する。
        apo_api=api_style(),
        # E2E 動作確認用に最小 APO 設定（1 ラウンド・1 候補）。本番では rounds / beam を増やす。
        rounds=1,
        apo_beam_width=1,
        apo_branch_factor=1,
    )

    print(f"train_score={result.train_score:.3f}")
    print("--- before（rollout 時の合成済み instructions）---")
    print(result.seed)
    print("--- after（最適化済みプロンプト）---")
    print(result.prompt)
    print("--- diff（unified diff）---")
    print(result.diff or "(no change)")


if __name__ == "__main__":
    asyncio.run(main())
