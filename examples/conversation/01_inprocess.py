"""会話 Helper を in-process で使う最小例（ConversationService 直接利用）。

サーバ / CLI を起動せず、`ConversationService` を Python から直接呼んで会話する。
完結応答（`send`）とストリーミング（`stream`）の両方を示す。会話は同一 conversation_id
で継続し、履歴は SDK Session（既定 in-memory）に委ねる。

Azure OpenAI の環境変数（AZURE_OPENAI_* 。examples/_azure.py 参照）を設定して実行:
    uv run python examples/conversation/01_inprocess.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from oai_agentspec import AgentRegistry, AgentSpec
from oai_agentspec.runtime.conversation import (
    ApprovalRequired,
    ConversationService,
    StreamDelta,
    StreamDone,
    StreamError,
)

# examples/ ルートを import パスへ（共有ヘルパ _azure を解決するため）。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
from _azure import azure_model  # noqa: E402


def build_registry() -> AgentRegistry:
    registry = AgentRegistry()
    registry.register(
        AgentSpec(
            name="assistant",
            instructions="あなたは簡潔に答える日本語アシスタントです。",
            model=azure_model(),
        )
    )
    registry.validate()
    return registry


async def main() -> None:
    chat = ConversationService(build_registry())
    print("利用可能エージェント:", chat.agents())

    # 1 つの会話（conversation_id）を作り、複数ターン継続する。
    conversation_id = await chat.create_conversation()

    # ターン1: 完結応答（send）。send は SendResult を返す（承認待ちなしなら status="final"）。
    result = await chat.send("assistant", "日本の首都は?", conversation_id=conversation_id)
    print("\n[send] ターン1:", result.output)

    # ターン2: ストリーミング（stream）。直前ターンの履歴を踏まえる。
    print("[stream] ターン2: ", end="", flush=True)
    # stream は 4 種を yield しうる: StreamDelta（断片）/ StreamDone（完了）/
    # StreamError（エラー）/ ApprovalRequired（承認待ち・StreamEvent Union とは別型）。
    # 消費側は 4 種すべてを分岐するのが安全（取りこぼし防止）。
    async for event in chat.stream(
        "assistant", "その都市の有名な観光地を 1 つ", conversation_id=conversation_id
    ):
        if isinstance(event, StreamDelta):
            print(event.text, end="", flush=True)
        elif isinstance(event, StreamDone):
            print()  # 改行（完了）
        elif isinstance(event, StreamError):
            print(f"\n[error] {event.code}: {event.message}")
        elif isinstance(event, ApprovalRequired):
            print("\n[approval] 承認待ち:", [(a.tool_name, a.call_id) for a in event.approvals])


if __name__ == "__main__":
    asyncio.run(main())
