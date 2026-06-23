"""WebSocket ストリーミング会話ハンドラ（serve extra・agents 非依存）。

会話サービスの stream / stream_resolve を WS の turn -> token* -> done / error /
approval_required ループへ調停する `_register_ws_route` / `_drive_ws_turn` /
`_receive_ws_approval` / `_send_ws_error` を提供する。SDK（`agents`）は import しない。
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from ...conversation import (
    ApprovalRequired,
    ConversationErrorCode,
    ConversationService,
    StreamDelta,
    StreamDone,
    StreamError,
)
from ..protocol import WsClientMsg, WsServerMsg

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from ...conversation import StreamEvent


def _register_ws_route(app: FastAPI) -> None:
    """ストリーミング会話の WebSocket ルートを app に登録する。

    クライアントの turn（agent_name / conversation_id / text）を受け、会話サービスの
    stream を `async for` で回し、StreamDelta → token、StreamDone → done、
    StreamError → error の JSON メッセージを逐次送る。1 turn 処理後に接続を閉じる。

    Args:
        app: ルートを登録する FastAPI app。
    """

    @app.websocket("/ws")
    async def conversation_ws(websocket: WebSocket) -> None:
        """ストリーミング会話の WebSocket ハンドラ。"""
        service: ConversationService = websocket.app.state.service
        await websocket.accept()
        try:
            message = await websocket.receive_json()
        except WebSocketDisconnect:
            return
        except (ValueError, json.JSONDecodeError):
            # 不正 JSON。AttributeError で抜けず error メッセージで返して閉じる。
            await _send_ws_error(
                websocket,
                ConversationErrorCode.EXECUTION_ERROR.value,
                "不正な JSON メッセージです",
            )
            await websocket.close()
            return

        if not isinstance(message, dict) or message.get("type") != WsClientMsg.TURN.value:
            # 非 dict / 未対応種別。dict でないと .get で AttributeError になるため先に弾く。
            kind = message.get("type") if isinstance(message, dict) else type(message).__name__
            await _send_ws_error(
                websocket,
                ConversationErrorCode.EXECUTION_ERROR.value,
                f"未対応の WS メッセージ種別です: {kind!r}",
            )
            await websocket.close()
            return

        agent_name = message.get("agent_name", "")
        conversation_id = message.get("conversation_id", "")
        text = message.get("text", "")
        try:
            stream = service.stream(agent_name, text, conversation_id=conversation_id)
            await _drive_ws_turn(websocket, service, conversation_id, stream)
        except WebSocketDisconnect:
            return
        await websocket.close()


async def _drive_ws_turn(
    websocket: WebSocket,
    service: ConversationService,
    conversation_id: str,
    stream: AsyncIterator[StreamEvent | ApprovalRequired],
) -> None:
    """ストリームを回し token/done/error を送る。承認待ちなら承認を受けて再開する（D-WsMsg）。

    `stream`（最初は通常ターン、以降は再開ストリーム）を `async for` で回し、`StreamDelta` →
    token、`StreamDone` → done、`StreamError` → error を送る。`ApprovalRequired` を受けたら
    `approval_required`（done は送らない）を送り、クライアントの `approval`（decisions 配列）を
    受けて `stream_resolve` で再開する。再開後に再度 `ApprovalRequired` なら段階解決として
    繰り返す。承認待ちが無いターンは token→done のみで終わる（既存 3 分岐不変・NFR-6）。

    Args:
        websocket: 送受信する WebSocket。
        service: 会話サービス（再開の委譲先）。
        conversation_id: 対象会話 ID。
        stream: 通常ターン or 再開ストリーム（`StreamEvent | ApprovalRequired` を yield）。
    """
    while True:
        pending_required: ApprovalRequired | None = None
        async for event in stream:
            if isinstance(event, StreamDelta):
                await websocket.send_json({"type": WsServerMsg.TOKEN.value, "text": event.text})
            elif isinstance(event, StreamDone):
                await websocket.send_json(
                    {"type": WsServerMsg.DONE.value, "output": event.final_output}
                )
            elif isinstance(event, StreamError):
                # エラー受信で確定終端する。error 送出後は承認待ち再受付ループへ進ませない。
                await _send_ws_error(websocket, event.code, event.message)
                return
            elif isinstance(event, ApprovalRequired):
                pending_required = event
        if pending_required is None:
            # 承認待ちなしターン（token→done のみで完了）。
            return
        await websocket.send_json(
            {
                "type": WsServerMsg.APPROVAL_REQUIRED.value,
                "pending": [
                    {"tool_name": p.tool_name, "call_id": p.call_id}
                    for p in pending_required.approvals
                ],
            }
        )
        decisions = await _receive_ws_approval(websocket)
        if decisions is None:
            return
        stream = service.stream_resolve(conversation_id, decisions)


async def _receive_ws_approval(websocket: WebSocket) -> list[dict[str, Any]] | None:
    """承認待ち通知後にクライアントの `approval` メッセージを受け decisions plain 形へ変換する。

    `approval` 種別以外 / 不正 JSON は error メッセージを送って None を返す（呼び出し側は終端）。
    不正 JSON の捕捉は初回 receive（turn）と対称にする（`(ValueError, json.JSONDecodeError)`）。
    `WebSocketDisconnect` は捕捉せず呼び出し側へ伝播させる（既存どおり）。

    Args:
        websocket: 受信する WebSocket。

    Returns:
        decisions の plain dict のリスト。不正メッセージ / 不正 JSON 時は None。
    """
    try:
        message = await websocket.receive_json()
    except (ValueError, json.JSONDecodeError):
        # 不正 JSON。初回 receive と同様に error メッセージで返して終端させる。
        await _send_ws_error(
            websocket,
            ConversationErrorCode.EXECUTION_ERROR.value,
            "不正な JSON メッセージです",
        )
        return None
    if not isinstance(message, dict) or message.get("type") != WsClientMsg.APPROVAL.value:
        kind = message.get("type") if isinstance(message, dict) else type(message).__name__
        await _send_ws_error(
            websocket,
            ConversationErrorCode.EXECUTION_ERROR.value,
            f"承認待ちに対し未対応の WS メッセージ種別です: {kind!r}",
        )
        return None
    raw_decisions = message.get("decisions", [])
    decisions: list[dict[str, Any]] = []
    if isinstance(raw_decisions, list):
        for item in raw_decisions:
            if not isinstance(item, dict):
                continue
            decisions.append(
                {
                    "call_id": str(item.get("call_id", "")),
                    # fail-closed: decision 欠落時の既定は "reject"（非実行・NFR-7）。CLI の未知
                    # 入力→reject、apply_approvals の "approve" 明示一致以外→reject と安全側で統一。
                    "decision": str(item.get("decision", "reject")),
                    "rejection_message": item.get("rejection_message"),
                }
            )
    return decisions


async def _send_ws_error(websocket: WebSocket, code: str, message: str) -> None:
    """WS error メッセージ（type / code / message）を送る。

    Args:
        websocket: 送信先 WebSocket。
        code: エラーコード。
        message: エラーメッセージ。
    """
    await websocket.send_json({"type": WsServerMsg.ERROR.value, "code": code, "message": message})
