"""HITL（ツール実行の承認）を in-process で使う例（ConversationService 直接利用）。

危険・不可逆なツール（ここでは擬似的なファイル削除）を `needs_approval=True` で宣言すると、
エージェントがそのツールを呼んでも実行前に「承認待ち」になる。`send` は最終応答ではなく
承認待ち（`SendResult(status="pending", pending=[...])`）を返し、利用者が `resolve_approvals`
で call_id 単位に approve / reject する。approve するとツールが実行され会話が再開し、reject
するとツールを実行せず会話が継続する（承認前は実行されない＝安全機構）。

`function_tool` は oai_agentspec から公開されており、利用者は `agents` を直接 import せず
承認必須ツールを宣言できる。承認は WebSocket サーバ + CLI（`oai-agentspec chat`）経由でも
同じ意味論で行える（本例は in-process の最小デモ）。

Azure OpenAI の環境変数（AZURE_OPENAI_* 。examples/_azure.py 参照）を設定して実行:
    uv run python examples/conversation/04_hitl_approval.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from oai_agentspec import AgentRegistry, AgentSpec, function_tool
from oai_agentspec.runtime.conversation import ApprovalDecision, ConversationService

# examples/ ルートを import パスへ（共有ヘルパ _azure を解決するため）。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
from _azure import azure_model  # noqa: E402

# 実際に実行されたファイル削除を記録する（承認前は空のまま＝NFR-7 の確認用）。
DELETED: list[str] = []


@function_tool(needs_approval=True)
def delete_file(path: str) -> str:
    """指定パスのファイルを削除する（承認必須・本例では擬似的に記録するのみ）。

    Args:
        path: 削除対象のファイルパス。

    Returns:
        削除結果のメッセージ。
    """
    DELETED.append(path)
    return f"削除しました: {path}"


def build_registry() -> AgentRegistry:
    registry = AgentRegistry()
    registry.register(
        AgentSpec(
            name="ops",
            instructions=(
                "あなたは運用アシスタントです。ファイル削除を依頼されたら、必ず delete_file "
                "ツールを呼んで実行してください。自分で確認を求めず、ツールを使うこと。"
            ),
            model=azure_model(),
            tools=[delete_file],
        )
    )
    registry.validate()
    return registry


async def _run_turn(chat: ConversationService, *, decision: str) -> None:
    """1 会話: 削除を依頼 -> 承認待ち -> decision（approve/reject）で解決する。"""
    cid = await chat.create_conversation()
    result = await chat.send("ops", "古いログ /var/log/old.log を削除して。", conversation_id=cid)

    if result.status != "pending":
        # モデルがツールを呼ばなかった場合（非承認応答）。
        print(f"[{decision}] 承認待ちは発生しませんでした:", result.output)
        return

    # 承認待ち（call_id 単位）。ここでは全件に同じ decision を適用する。
    print(f"[{decision}] 承認待ち:", [(p.tool_name, p.call_id) for p in result.pending])
    # 型付き入力 ApprovalDecision を使う（dict も後方互換で受け付ける）。
    decisions = [
        ApprovalDecision(call_id=p.call_id, approve=(decision == "approve"))
        for p in result.pending
    ]
    resolved = await chat.resolve_approvals(cid, decisions)
    print(f"[{decision}] 解決後:", resolved.status, "/", resolved.output)


async def main() -> None:
    chat = ConversationService(build_registry())

    # 1) approve: ツールが実行され、削除が記録される。
    await _run_turn(chat, decision="approve")
    print("approve 後の削除記録:", DELETED)

    # 2) reject: ツールは実行されず、削除は記録されない（会話は継続）。
    before = list(DELETED)
    await _run_turn(chat, decision="reject")
    print("reject 後の削除記録（増えない）:", DELETED, "/ 変化なし:", DELETED == before)


if __name__ == "__main__":
    asyncio.run(main())
