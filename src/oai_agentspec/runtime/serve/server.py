"""会話サーバ起動入口（uvicorn・serve extra・agents 非依存）。

`AgentRegistry` または構築済み `ConversationService` を受け、`create_app` で FastAPI app を
組み立て uvicorn で起動する `start_server` と、既定バインド先 `DEFAULT_HOST` / `DEFAULT_PORT` を
提供する。SDK（`agents`）は import しない。uvicorn / `create_app` は起動時のみ必要なため関数内で
遅延 import する（`server -> app` の循環回避・serve extra の局在化）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..conversation import ConversationService

if TYPE_CHECKING:
    from ...registry import AgentRegistry
    from ..conversation import SessionPolicy

# 既定バインド先（localhost のみ・外部非公開・認証なし・NFR-1）。
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


def start_server(
    registry_or_service: AgentRegistry | ConversationService,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    session_policy: SessionPolicy | None = None,
    entry_agent: str | None = None,
) -> None:
    """会話サーバを uvicorn で起動する（ブロッキング）。

    `AgentRegistry` を渡すと `ConversationService` を内部で生成し、`ConversationService`
    を渡すとそのまま使う。既定 host は 127.0.0.1（localhost のみ・認証なし・NFR-1）。
    `session_policy` / `entry_agent` は `AgentRegistry` を渡したとき内部生成する
    `ConversationService` に適用する（`ConversationService` を直接渡した場合は無視する）。

    Args:
        registry_or_service: `AgentRegistry` または構築済み `ConversationService`。
        host: バインド先ホスト（既定 127.0.0.1）。
        port: バインド先ポート（既定 8000）。
        session_policy: session 生成方針（永続化先・compaction 設定 `CompactionConfig`）。
            registry 渡し時のみ適用する。
        entry_agent: エントリ（起点）エージェント名。registry 渡し時のみ適用。None で
            registry 登録順の先頭を採用。
    """
    import uvicorn

    from .app import create_app

    if isinstance(registry_or_service, ConversationService):
        service = registry_or_service
    else:
        service = ConversationService(
            registry_or_service,
            session_policy=session_policy,
            entry_agent=entry_agent,
        )
    app = create_app(service)
    uvicorn.run(app, host=host, port=port)
