"""会話 CLI クライアントの WS メッセージ種別定数とパーサ（cli extra・agents 非依存）。

WS メッセージ種別定数（`WS_TYPE_*`）と、サーバ応答 JSON を client 側 plain 型へ写すパーサ
（`_parse_pending` / `_parse_send_result`）・WebSocket URL 導出（`_ws_url`）を提供する。

`WS_TYPE_*` は `serve.protocol` の `WsClientMsg` / `WsServerMsg` と同じ WS メッセージ種別
文字列を別々に定義した意図的な再掲である（agents 非依存・別プロセス境界）。両者の値一致は
serve / cli 両 extra 導入時のトリップワイヤ（`test_ws_protocol_parity`）で担保する（統合しない）。

SDK（`agents`）は import しない（NFR-1）。
"""

from __future__ import annotations

from typing import Any

from ._models import PendingApproval, SendResult

# WS メッセージ種別（serve.protocol の WsClientMsg / WsServerMsg と一致・agents 非依存の
# ため再掲）。両者の値一致は serve/cli 両 extra 導入時のトリップワイヤで担保する。
WS_TYPE_TURN = "turn"
WS_TYPE_TOKEN = "token"
WS_TYPE_DONE = "done"
WS_TYPE_ERROR = "error"
# HITL: 承認待ち通知（s→c）/ 承認・却下（c→s）。serve.protocol の APPROVAL_REQUIRED /
# APPROVAL と値一致（トリップワイヤ対象）。
WS_TYPE_APPROVAL_REQUIRED = "approval_required"
WS_TYPE_APPROVAL = "approval"

__all__ = [
    "WS_TYPE_APPROVAL",
    "WS_TYPE_APPROVAL_REQUIRED",
    "WS_TYPE_DONE",
    "WS_TYPE_ERROR",
    "WS_TYPE_TOKEN",
    "WS_TYPE_TURN",
]


def _parse_pending(raw: Any) -> list[PendingApproval]:
    """サーバ応答の pending（list[dict]）を `PendingApproval` の列へ変換する。

    Args:
        raw: 承認待ち一覧の生データ（list[dict] を期待・それ以外は空扱い）。

    Returns:
        `PendingApproval` のリスト（不正要素はスキップ）。
    """
    if not isinstance(raw, list):
        return []
    result: list[PendingApproval] = []
    for item in raw:
        if isinstance(item, dict) and "call_id" in item:
            result.append(
                PendingApproval(
                    tool_name=str(item.get("tool_name", "")),
                    call_id=str(item["call_id"]),
                )
            )
    return result


def _parse_send_result(data: dict[str, Any]) -> SendResult:
    """REST の SendResponse JSON を `SendResult` へ変換する（status/pending/output 解釈）。

    `status` 欠落の旧サーバ応答は最終応答（`status="final"`）として扱う（後方互換）。

    Args:
        data: SendResponse の JSON dict。

    Returns:
        `SendResult`（最終応答 or 承認待ち）。
    """
    status = str(data.get("status", "final"))
    if status == "pending":
        return SendResult(
            status="pending",
            output=None,
            pending=_parse_pending(data.get("pending", [])),
        )
    return SendResult(status="final", output=str(data.get("output", "")), pending=[])


def _ws_url(base_url: str) -> str:
    """REST の base_url から WebSocket URL（/ws）を導出する。

    Args:
        base_url: REST のベース URL（http(s)://host:port）。

    Returns:
        WebSocket URL（ws(s)://host:port/ws）。
    """
    ws_base = base_url.rstrip("/")
    if ws_base.startswith("https://"):
        ws_base = "wss://" + ws_base[len("https://") :]
    elif ws_base.startswith("http://"):
        ws_base = "ws://" + ws_base[len("http://") :]
    return ws_base + "/ws"
