"""会話 Helper の共有コア（agents 非依存・公開 API）。

registry 登録済みエージェントとローカルで会話する共有コアサービスと、その受け渡しに使う
plain 型を提供する。SDK 結合は `_adapters` に閉じ、本パッケージは agents を import しない
（NFR-1）。サーバ入口は `oai_agentspec.runtime.serve`（serve extra）、CLI クライアントは
`oai_agentspec.runtime.cli`（cli extra）が別途提供する。
"""

from __future__ import annotations

from .service import ConversationService
from .session import CompactionConfig, SessionPolicy
from .types import (
    ApprovalDecision,
    ApprovalRequired,
    ConversationError,
    ConversationErrorCode,
    PendingApproval,
    SendResult,
    SendStatus,
    SessionInfo,
    StreamDelta,
    StreamDone,
    StreamError,
    StreamEvent,
)

__all__ = [
    "ApprovalDecision",
    "ApprovalRequired",
    "CompactionConfig",
    "ConversationError",
    "ConversationErrorCode",
    "ConversationService",
    "PendingApproval",
    "SendResult",
    "SendStatus",
    "SessionInfo",
    "SessionPolicy",
    "StreamDelta",
    "StreamDone",
    "StreamError",
    "StreamEvent",
]
