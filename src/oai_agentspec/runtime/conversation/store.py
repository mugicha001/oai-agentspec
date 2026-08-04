"""会話 store（conversation_id -> エントリ・会話毎ロック・agents 非依存）。

`conversation_id` から会話エントリ（SDK `Session` を不透明型で保持 + 会話毎の
`asyncio.Lock`）を引く in-memory dict。sample の `StateStore` 構造を参考にした最小の
状態管理であり、SDK 標準様式に代わる独自エンジンではない（NFR-2）。`Session` は
不透明型（`Any`）で保持し agents を import しない（NFR-1）。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..._validation import validate_bool

if TYPE_CHECKING:
    from .types import PendingApproval


@dataclass
class ConversationEntry:
    """1 会話のエントリ。

    Attributes:
        conversation_id: 会話 ID（store のキー）。
        session_id: 紐づく SDK Session の session_id。
        session: SDK `Session`（不透明保持・agents を import しないため `Any`）。
        agent_name: 直近に会話したエージェント名（任意・履歴/復元の参考・復元時 initial_agent
            解決元・D-Resume）。
        lock: 同一会話の同時実行を排他する会話毎ロック。
        turn_count: これまでの送受信ターン数。
        persist: この会話を db へ永続化するか（session_id 明示で True）。HITL 中断状態の
            永続化要否の判定に使う（揮発会話はメモリのみ・FR-10）。
        pending_state: HITL の中断状態（不透明 SDK `RunState`・None=中断なし・D-State）。
            agents を import しないため `Any` で保持し中身を覗かない。
        pending_approvals: 現在の承認待ち一覧（call_id 単位の plain `PendingApproval`）。
            中断なしなら空リスト。
    """

    conversation_id: str
    session_id: str
    session: Any
    agent_name: str | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    turn_count: int = 0
    persist: bool = False
    pending_state: Any = None
    pending_approvals: list[PendingApproval] = field(default_factory=list)

    def __post_init__(self) -> None:
        """`persist` が bool であることを構築時に検証する（構築後の代入は対象外）。

        Raises:
            ValueError: `persist` が bool でない場合。
        """
        validate_bool(self.persist, "persist")


class ConversationStore:
    """in-memory な会話エントリストア（dict + 辞書ロックで race を防ぐ）。

    辞書操作は `asyncio.Lock` で保護する。会話毎の排他は各エントリの `lock` を
    呼び出し側（会話サービス）が取得して行う。
    """

    def __init__(self) -> None:
        """空の store を生成する。"""
        self._entries: dict[str, ConversationEntry] = {}
        self._dict_lock: asyncio.Lock = asyncio.Lock()

    async def add(self, entry: ConversationEntry) -> bool:
        """会話エントリを登録する（辞書ロック内で重複を原子的に拒否）。

        既存の `conversation_id` は上書きせず False を返す。重複チェックと挿入を
        同一の辞書ロック内で行うことで、同一 `conversation_id` を並行作成しても
        先勝ちの 1 件だけが登録される（TOCTOU レース回避）。

        Args:
            entry: 登録するエントリ。

        Returns:
            新規登録できたら True、`conversation_id` が既存なら False（上書きしない）。
        """
        async with self._dict_lock:
            if entry.conversation_id in self._entries:
                return False
            self._entries[entry.conversation_id] = entry
            return True

    async def get(self, conversation_id: str) -> ConversationEntry | None:
        """会話エントリを取得する（存在しなければ None・辞書ロック付き）。

        Args:
            conversation_id: 取得対象の会話 ID。

        Returns:
            会話エントリ。未登録なら None。
        """
        async with self._dict_lock:
            return self._entries.get(conversation_id)

    async def remove(self, conversation_id: str) -> bool:
        """会話エントリを削除する（削除成功で True・辞書ロック付き）。

        Args:
            conversation_id: 削除対象の会話 ID。

        Returns:
            削除できたら True、未登録なら False。
        """
        async with self._dict_lock:
            return self._entries.pop(conversation_id, None) is not None


__all__ = ["ConversationEntry", "ConversationStore"]
