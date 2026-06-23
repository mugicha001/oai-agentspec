"""会話エラーコード -> HTTP status の写像（serve extra・agents 非依存）。

`ConversationErrorCode` を REST の HTTP status へ写す `_ERROR_STATUS` マップと `_error_status`
を提供する。`create_app` の exception_handler 本体が参照する。SDK（`agents`）は import しない。
"""

from __future__ import annotations

from ..conversation import ConversationErrorCode

# ConversationErrorCode -> HTTP status の対応（不正名/会話=404、重複作成=409、
# モデル未注入=503、実行エラー=500、HITL 承認系=404/409・D-Err）。
_ERROR_STATUS: dict[ConversationErrorCode, int] = {
    ConversationErrorCode.UNKNOWN_AGENT: 404,
    ConversationErrorCode.UNKNOWN_CONVERSATION: 404,
    ConversationErrorCode.CONVERSATION_ALREADY_EXISTS: 409,
    ConversationErrorCode.MODEL_NOT_CONFIGURED: 503,
    ConversationErrorCode.EXECUTION_ERROR: 500,
    ConversationErrorCode.UNKNOWN_APPROVAL: 404,
    ConversationErrorCode.APPROVAL_ALREADY_RESOLVED: 409,
    ConversationErrorCode.NO_PENDING_APPROVAL: 409,
}


def _error_status(code: ConversationErrorCode) -> int:
    """会話エラーコードに対応する HTTP status を返す（既定 500）。

    Args:
        code: 会話エラーコード。

    Returns:
        HTTP status コード。
    """
    return _ERROR_STATUS.get(code, 500)
