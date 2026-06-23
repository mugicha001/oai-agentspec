"""外部クライアントで compaction（履歴圧縮）を有効化し、実際に発火させる会話例。

`SessionPolicy(compaction=CompactionConfig(enabled=True, client=..., model=...))` で
compaction を明示的に有効化し、`ConversationService` を in-process で回す。compaction の
有効化は `enabled` フラグで明示制御し、client/model の受け渡しと有効化判定を分離する
（client を渡しただけでは有効化されない）。

圧縮の原理（SDK 仕様）:
  - compaction は OpenAI Responses API 専用。SDK の `OpenAIResponsesCompactionSession` が
    `SQLiteSession` をラップし、毎ターン後に判定フックを呼んで True なら OpenAI の
    `responses.compact` API（サーバ側＝モデルによる要約）を呼び、SQLite の履歴を要約版へ
    物理置換する。
  - 既定の判定は「候補アイテム（ユーザー発話と既存 compaction を除く履歴）が 10 個以上で発火」。
    本 example はデモのため `options` で閾値を下げ（候補 3 個以上）、数ターン回して実際に
    圧縮を発火させる。発火は SDK の compaction ロガー（DEBUG）で観測する。
  - `model` は OpenAI 形式のモデル名（`gpt-*` / `o*` / `ft:gpt-*`）である必要がある（SDK が
    検証し、満たさないと構築時に ValueError）。Azure ではデプロイ名が OpenAI 形式
    （例: `gpt-4.1`）であること。None なら SDK 既定 `gpt-4.1` を使う。

モデル実行用のクライアント注入（`azure_model()`）とは別軸（圧縮用クライアント）である。

Azure OpenAI の環境変数（AZURE_OPENAI_* 。examples/_shared/_azure.py 参照）を設定して実行:
    uv run python examples/conversation/06_compaction.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from oai_agentspec import AgentRegistry, AgentSpec
from oai_agentspec.runtime.conversation import (
    CompactionConfig,
    ConversationService,
    SessionPolicy,
)

# examples/ ルートを import パスへ（共有ヘルパ _azure を解決するため）。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
from _azure import azure_client, azure_deployment, azure_model  # noqa: E402

# 既定の発火閾値は候補 10 個。デモではこれを下げて数ターンで圧縮を発火させる。
DEMO_COMPACTION_THRESHOLD = 3


def _compact_when_candidates_at_least(threshold: int) -> Callable[[dict[str, Any]], bool]:
    """候補アイテムが threshold 個以上で圧縮を発火させる判定フックを返す。"""

    def should_trigger(context: dict[str, Any]) -> bool:
        return len(context["compaction_candidate_items"]) >= threshold

    return should_trigger


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


# 圧縮タイミングの設定レシピ（CompactionConfig.options 経由で SDK へ素通しされる）:
#   - 既定（options 省略）       : 候補アイテムが 10 個以上で発火
#   - 候補数で閾値を変える       : options={"should_trigger_compaction":
#                                    lambda ctx: len(ctx["compaction_candidate_items"]) >= N}
#   - 全履歴アイテム数で判定     : options={"should_trigger_compaction":
#                                    lambda ctx: len(ctx["session_items"]) >= N}
#   判定フックに渡る context のキー:
#     compaction_candidate_items（ユーザー発話と既存 compaction を除く履歴）/
#     session_items（全履歴）/ response_id / compaction_mode
#   ※ タイミングは should_trigger_compaction で制御する。compaction_mode（auto/
#     previous_response_id/input）は「いつ」ではなく「どう履歴を渡すか」の別軸。
# 本 example はデモのため候補 3 個で発火させる（実運用は既定 10 が経済的）。
def build_policy() -> SessionPolicy:
    """compaction を明示的に有効化した SessionPolicy を組む。

    `enabled=True` かつ client 必須（欠けると CompactionConfig 構築時に ValueError）。
    `enabled=False` や `compaction=None` なら client を渡しても圧縮しない。`options` は
    `OpenAIResponsesCompactionSession` へ素通しされる（ここでは発火閾値を下げる判定フックを渡す）。
    """
    return SessionPolicy(
        compaction=CompactionConfig(
            enabled=True,
            client=azure_client(),
            model=azure_deployment(),
            options={
                "should_trigger_compaction": _compact_when_candidates_at_least(
                    DEMO_COMPACTION_THRESHOLD
                ),
            },
        )
    )


async def main() -> None:
    # 圧縮の発火を観測するため SDK の compaction ロガーを DEBUG にする
    # （"compact: start" / "compact: done" が表示される）。他ログは静かに保つ。
    logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")
    logging.getLogger("openai-agents.openai.compaction").setLevel(logging.DEBUG)

    chat = ConversationService(build_registry(), session_policy=build_policy())
    print("利用可能エージェント:", chat.agents())

    # 1 つの会話を複数ターン継続する。候補が閾値を越えた時点で履歴が compaction される。
    conversation_id = await chat.create_conversation()
    questions = [
        "日本の首都は?",
        "その都市の有名な観光地を 1 つ",
        "そこへの最寄り駅は?",
        "近くで食べられる名物は?",
        "1 日観光のモデルコースを一言で",
    ]
    for i, q in enumerate(questions, start=1):
        result = await chat.send("assistant", q, conversation_id=conversation_id)
        print(f"\n[ターン{i}] Q: {q}\n        A: {result.output}")

    print("\n（上に 'compact: start/done' ログが出ていれば履歴が圧縮されている）")


if __name__ == "__main__":
    asyncio.run(main())
