"""session_id 連動の永続化と途中再開（resume）の例。

session_id を明示するとファイルに永続化され、プロセスを跨いで（=別の
ConversationService インスタンスから）過去履歴を踏まえて続きから再開できる。
無指定の会話は in-memory（揮発）で再開できない。

本例では一時ディレクトリを保存先に指定し（実利用の既定は `memory/`）、
同一 session_id で 2 つのサービスを順に動かして「続き」が成立することを示す。

Azure OpenAI の環境変数（examples/_azure.py 参照）を設定して実行:
    uv run python examples/conversation/02_session_resume.py
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

from oai_agentspec import AgentRegistry, AgentSpec
from oai_agentspec.runtime.conversation import ConversationService, SessionPolicy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
from _azure import azure_model  # noqa: E402

SESSION_ID = "demo-resume"


def build_registry() -> AgentRegistry:
    registry = AgentRegistry()
    registry.register(
        AgentSpec(
            name="assistant",
            instructions="あなたは簡潔に答える日本語アシスタントです。直前の話題を覚えます。",
            model=azure_model(),
        )
    )
    registry.validate()
    return registry


async def first_run(policy: SessionPolicy) -> None:
    """1 回目: session_id を明示して永続化し、名前を伝える。"""
    chat = ConversationService(build_registry(), session_policy=policy)
    cid = await chat.create_conversation(session_id=SESSION_ID)
    result = await chat.send(
        "assistant", "私の名前はムギです。覚えてください。", conversation_id=cid
    )
    print("[1 回目]", result.output)


async def second_run(policy: SessionPolicy) -> None:
    """2 回目: 別インスタンスで同一 session_id を渡し、続きから再開する。"""
    chat = ConversationService(build_registry(), session_policy=policy)
    # list_sessions で永続化済み session をメタ情報付きで列挙できる（D5）。
    sessions = await chat.list_sessions()
    print("[永続化済み session]", [(s.session_id, s.turn_count, s.preview) for s in sessions])
    # 既存 session_id を渡して復元（過去履歴を踏まえて継続）。
    cid = await chat.create_conversation(session_id=SESSION_ID)
    result = await chat.send("assistant", "私の名前は何でしたか?", conversation_id=cid)
    print("[2 回目]", result.output)


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        policy = SessionPolicy(base_dir=Path(tmp))  # 実利用の既定は memory/
        await first_run(policy)
        await second_run(policy)  # 同一保存先・同一 session_id で「続き」になる


if __name__ == "__main__":
    asyncio.run(main())
