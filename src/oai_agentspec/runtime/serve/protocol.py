"""WebSocket メッセージ種別と REST スキーマ（serve 専用・agents 非依存）。

sample（`ws_protocol.py` の turn / token / done / error）を参考にした簡潔なメッセージ
種別を定義する。bargein / stop / hitl / init は不採用（簡易スコープ）。SDK（`agents`）は
import しない（NFR-1。会話サービスの plain 型へ橋渡しするだけ）。
"""

from __future__ import annotations

from enum import StrEnum


class WsClientMsg(StrEnum):
    """クライアント → サーバの WebSocket メッセージ種別。"""

    TURN = "turn"
    # HITL: 承認/却下（decisions 配列を運ぶ・c→s・D-WsMsg）。承認待ちターンで token* の後に
    # クライアントが送り、サーバは会話サービスの同一承認処理点へ委譲して再開する。
    APPROVAL = "approval"


class WsServerMsg(StrEnum):
    """サーバ → クライアントの WebSocket メッセージ種別。"""

    TOKEN = "token"
    DONE = "done"
    ERROR = "error"
    # HITL: 承認待ち通知（tool_name/call_id 一覧・s→c・StreamEvent Union 非混入の専用イベント
    # 由来）。承認待ちターンでは done の代わりに送り、承認/却下を待つ（NFR-6）。
    APPROVAL_REQUIRED = "approval_required"


__all__ = ["WsClientMsg", "WsServerMsg"]
