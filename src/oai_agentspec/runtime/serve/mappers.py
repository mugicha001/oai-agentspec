"""plain 値 <-> REST スキーマの変換（serve extra・agents 非依存）。

会話サービスの plain 値（`SendResult` / `PendingApproval`）と REST リクエスト（`ApprovalRequest`）
を REST レスポンススキーマ / decisions plain 形へ写す純粋関数群（`_pending_schemas` /
`_send_response` / `_decisions_from_request` / `_approvals_response`）を提供する。SDK（`agents`）は
import しない。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .schemas import (
    ApprovalsResponse,
    PendingApprovalSchema,
    SendResponse,
)

if TYPE_CHECKING:
    from ..conversation import PendingApproval, SendResult
    from .schemas import ApprovalRequest


def _pending_schemas(pending: list[PendingApproval]) -> list[PendingApprovalSchema]:
    """承認待ち一覧（plain `PendingApproval`）を REST スキーマ列へ写す（共通点）。

    Args:
        pending: 承認待ち一覧（call_id 単位の plain `PendingApproval`）。

    Returns:
        `PendingApprovalSchema` のリスト。
    """
    return [PendingApprovalSchema(tool_name=p.tool_name, call_id=p.call_id) for p in pending]


def _send_response(result: SendResult) -> SendResponse:
    """会話サービスの `SendResult` を REST の `SendResponse` へ写す（D-Disc）。

    `status="final"` は最終応答テキストを `output` に、`status="pending"` は承認待ち一覧を
    `pending` に載せ `output=None`。既存の最終応答形（status 既定 "final" + output）を壊さない。

    Args:
        result: 会話サービスの `SendResult`（最終応答 or 承認待ち）。

    Returns:
        REST レスポンス `SendResponse`。
    """
    if result.status == "pending":
        return SendResponse(
            status="pending",
            output=None,
            pending=_pending_schemas(result.pending),
        )
    return SendResponse(status="final", output=result.output or "", pending=None)


def _decisions_from_request(request: ApprovalRequest) -> list[dict[str, Any]]:
    """REST の `ApprovalRequest` を会話サービスの decisions plain 形へ変換する（D-WsMsg）。

    Args:
        request: 承認/却下リクエスト（decisions 配列）。

    Returns:
        `{"call_id", "decision", "rejection_message"}` の plain dict のリスト。
    """
    return [
        {
            "call_id": d.call_id,
            "decision": d.decision,
            "rejection_message": d.rejection_message,
        }
        for d in request.decisions
    ]


def _approvals_response(conversation_id: str, pending: list[PendingApproval]) -> ApprovalsResponse:
    """承認待ち一覧（plain）を REST の `ApprovalsResponse` へ写す（D-RestGet）。

    Args:
        conversation_id: 対象会話 ID。
        pending: 承認待ち一覧（call_id 単位の plain `PendingApproval`）。

    Returns:
        REST レスポンス `ApprovalsResponse`。
    """
    return ApprovalsResponse(
        conversation_id=conversation_id,
        pending=_pending_schemas(pending),
    )
