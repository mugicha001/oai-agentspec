"""会話サーバへの REST + WebSocket クライアント（cli extra・agents 非依存・別プロセス）。

起動中の FastAPI 会話サーバへ HTTP/WS で接続する `ConversationClient` を提供する。SDK
（`agents`）には依存せず、サーバの JSON メッセージのみを扱う（NFR-1）。httpx（REST）/
websockets（WS）はモジュールトップで import するため、本モジュールは cli extra 導入時のみ
import できる（本体 `__init__` からは強制 import しない）。

client 側 plain 型は `_models`、WS 種別定数とパーサは `_protocol` に分離する。従来の import
経路（`from oai_agentspec.runtime.cli.client import StreamToken, _ws_url, WS_TYPE_TURN, ...`）
を保つため本モジュールから再エクスポートする。
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import httpx
import websockets

from ._models import (
    ApprovalRequired,
    ConversationClientError,
    PendingApproval,
    SendResult,
    SessionMeta,
    StreamDone,
    StreamToken,
)
from ._protocol import (
    WS_TYPE_APPROVAL,
    WS_TYPE_APPROVAL_REQUIRED,
    WS_TYPE_DONE,
    WS_TYPE_ERROR,
    WS_TYPE_TOKEN,
    WS_TYPE_TURN,
    _parse_pending,
    _parse_send_result,
    _ws_url,
)

# 既定の接続先（localhost のみ・serve サーバの既定 host/port と一致）。
DEFAULT_BASE_URL = "http://localhost:8000"


class ConversationClient:
    """会話サーバへの REST + WebSocket クライアント（async）。

    `async with` でコンテキスト管理し、内部の `httpx.AsyncClient` を確実に閉じる。
    """

    def __init__(self, base_url: str = DEFAULT_BASE_URL, *, timeout: float = 30.0) -> None:
        """クライアントを生成する。

        Args:
            base_url: REST のベース URL（既定 http://localhost:8000）。
            timeout: REST リクエストのタイムアウト秒。
        """
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=timeout)

    async def __aenter__(self) -> ConversationClient:
        """コンテキストに入る（自身を返す）。"""
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """コンテキストを抜け、内部 httpx クライアントを閉じる。"""
        await self.aclose()

    async def aclose(self) -> None:
        """内部 httpx クライアントを閉じる。"""
        await self._client.aclose()

    # ------------------------------------------------------------------
    # REST
    # ------------------------------------------------------------------
    async def list_agents(self) -> list[str]:
        """登録済みエージェント名の一覧を取得する。

        Returns:
            エージェント名のリスト。

        Raises:
            ConversationClientError: 接続失敗 / サーバエラーの場合。
        """
        data = await self._get("/agents")
        agents = data.get("agents", [])
        return list(agents) if isinstance(agents, list) else []

    async def get_entry(self) -> str | None:
        """エントリ（起点）エージェント名を取得する（CLI のエントリ起点会話用）。

        Returns:
            エントリエージェント名。未決定（サーバ registry 空 / 旧サーバ）なら None。

        Raises:
            ConversationClientError: 接続失敗 / サーバエラーの場合。
        """
        data = await self._get("/agents")
        entry = data.get("entry")
        return str(entry) if isinstance(entry, str) else None

    async def list_sessions(self) -> list[SessionMeta]:
        """永続化済み session のメタ情報一覧を取得する（D5・復元候補・更新時刻降順）。

        Returns:
            `SessionMeta` のリスト（最終更新の新しい順・永続化会話が無ければ空）。

        Raises:
            ConversationClientError: 接続失敗 / サーバエラーの場合。
        """
        data = await self._get("/sessions")
        sessions = data.get("sessions", [])
        if not isinstance(sessions, list):
            return []
        result: list[SessionMeta] = []
        for item in sessions:
            if isinstance(item, dict) and "session_id" in item:
                result.append(
                    SessionMeta(
                        session_id=str(item["session_id"]),
                        updated_at=str(item.get("updated_at", "")),
                        turn_count=int(item.get("turn_count", 0) or 0),
                        preview=str(item.get("preview", "")),
                    )
                )
        return result

    async def get_history(
        self, session_id: str, *, limit: int | None = None
    ) -> list[dict[str, Any]]:
        """指定 session の過去履歴アイテムを取得する（復元時の表示用・D5）。

        Args:
            session_id: 取得対象の session_id。
            limit: 返す最大件数（直近側）。None でサーバ既定（直近 N 件）。

        Returns:
            履歴アイテム（dict）の時系列リスト。

        Raises:
            ConversationClientError: 接続失敗 / サーバエラーの場合。
        """
        path = f"/sessions/{session_id}/history"
        if limit is not None:
            path = f"{path}?limit={limit}"
        data = await self._get(path)
        items = data.get("items", [])
        return [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []

    async def create_conversation(self, *, session_id: str | None = None) -> str:
        """新規会話を作成し conversation_id を返す。

        Args:
            session_id: SDK Session の session_id（任意・名前付き会話）。

        Returns:
            生成された会話 ID。

        Raises:
            ConversationClientError: 接続失敗 / サーバエラーの場合。
        """
        body: dict[str, Any] = {}
        if session_id is not None:
            body["session_id"] = session_id
        data = await self._post("/conversations", body)
        return str(data["conversation_id"])

    async def send(self, agent_name: str | None, text: str, *, conversation_id: str) -> SendResult:
        """非ストリーミングで 1 ターン会話し最終応答 or 承認待ちを取得する（D-Disc）。

        サーバ応答の `status` / `pending` / `output` を解釈し `SendResult` を返す。
        `status="pending"` なら承認待ち一覧を持ち、`status="final"`（既定）なら最終応答テキスト
        を持つ。`status` 欠落の旧サーバ応答は最終応答として扱う（後方互換）。

        Args:
            agent_name: 会話相手のエージェント名。None でエントリエージェント起点
                （リクエストから agent_name を省略しサーバ側で解決させる）。
            text: ユーザー入力テキスト。
            conversation_id: 対象会話 ID。

        Returns:
            最終応答（`status="final"`・`output`）または承認待ち（`status="pending"`・`pending`）。

        Raises:
            ConversationClientError: 接続失敗 / サーバエラーの場合。
        """
        body: dict[str, Any] = {"text": text}
        if agent_name is not None:
            body["agent_name"] = agent_name
        data = await self._post(
            f"/conversations/{conversation_id}/messages",
            body,
        )
        return _parse_send_result(data)

    async def get_approvals(self, conversation_id: str) -> list[PendingApproval]:
        """現在の承認待ち一覧を取得する（冪等・復元直後の再提示用・D-RestGet）。

        Args:
            conversation_id: 対象会話 ID。

        Returns:
            承認待ち一覧（call_id 単位）。中断なしなら空リスト。

        Raises:
            ConversationClientError: 接続失敗 / サーバエラーの場合。
        """
        data = await self._get(f"/conversations/{conversation_id}/approvals")
        return _parse_pending(data.get("pending", []))

    async def resolve_approvals(
        self, conversation_id: str, decisions: list[dict[str, Any]]
    ) -> SendResult:
        """承認/却下を call_id 単位で適用し、再開後の最終応答 or 残承認待ちを取得する（D-WsMsg）。

        Args:
            conversation_id: 対象会話 ID。
            decisions: 適用する承認/却下（`{"call_id", "decision", "rejection_message"}` の列）。

        Returns:
            最終応答（`status="final"`）または残承認待ち（`status="pending"`・段階解決）。

        Raises:
            ConversationClientError: 接続失敗 / サーバエラーの場合。
        """
        data = await self._post(
            f"/conversations/{conversation_id}/approvals",
            {"decisions": decisions},
        )
        return _parse_send_result(data)

    # ------------------------------------------------------------------
    # WebSocket（ストリーミング）
    # ------------------------------------------------------------------
    async def stream(
        self,
        agent_name: str | None,
        text: str,
        *,
        conversation_id: str,
        approval_handler: Callable[[list[PendingApproval]], Awaitable[list[dict[str, Any]]]]
        | None = None,
    ) -> AsyncIterator[StreamToken | StreamDone | ApprovalRequired]:
        """ストリーミングで 1 ターン会話し token を逐次 yield、done で終端する（承認待ち対応）。

        WS に turn を送信し、token を `StreamToken` で逐次 yield、done を `StreamDone` で 1 回
        yield して終わる。承認待ち（`approval_required`）を受けたら `ApprovalRequired` を yield
        し、`approval_handler` があれば呼んで得た decisions を `approval` として同一接続で送り、
        再開後の token/done（または再度 `approval_required`）へ繋ぐ（段階解決）。handler が無い /
        decisions 空なら承認待ちを yield して終端する。サーバの error は
        `ConversationClientError` へ変換して送出する。

        Args:
            agent_name: 会話相手のエージェント名。None でエントリエージェント起点
                （turn から agent_name を省略しサーバ側で解決させる）。
            text: ユーザー入力テキスト。
            conversation_id: 対象会話 ID。
            approval_handler: 承認待ち発生時に decisions を解決する非同期コールバック。
                `PendingApproval` 列を受け decisions（plain dict の列）を返す。None なら
                承認待ちを yield するだけで継続しない（呼び出し側が別途処理）。

        Yields:
            `StreamToken`（断片）/ `StreamDone`（最終出力）/ `ApprovalRequired`（承認待ち）。

        Raises:
            ConversationClientError: 接続失敗 / サーバ error メッセージの場合。
        """
        url = _ws_url(self._base_url)
        turn: dict[str, Any] = {
            "type": WS_TYPE_TURN,
            "conversation_id": conversation_id,
            "text": text,
        }
        if agent_name is not None:
            turn["agent_name"] = agent_name
        try:
            async with websockets.connect(url) as ws:
                await ws.send(json.dumps(turn))
                async for raw in ws:
                    msg = json.loads(raw)
                    kind = msg.get("type")
                    if kind == WS_TYPE_TOKEN:
                        yield StreamToken(text=msg.get("text", ""))
                    elif kind == WS_TYPE_DONE:
                        yield StreamDone(output=msg.get("output", ""))
                        return
                    elif kind == WS_TYPE_ERROR:
                        raise ConversationClientError(
                            msg.get("message", "サーバエラー"), code=msg.get("code")
                        )
                    elif kind == WS_TYPE_APPROVAL_REQUIRED:
                        pending = _parse_pending(msg.get("pending", []))
                        yield ApprovalRequired(pending=pending)
                        if approval_handler is None:
                            return
                        decisions = await approval_handler(pending)
                        if not decisions:
                            return
                        await ws.send(
                            json.dumps({"type": WS_TYPE_APPROVAL, "decisions": decisions})
                        )
        except (OSError, websockets.WebSocketException) as exc:
            raise ConversationClientError(
                f"サーバへの WebSocket 接続に失敗しました（サーバ起動を確認してください）: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # 内部ヘルパ
    # ------------------------------------------------------------------
    async def _get(self, path: str) -> dict[str, Any]:
        """GET リクエストを送り JSON を返す（エラーを構造化）。"""
        try:
            resp = await self._client.get(path)
        except httpx.HTTPError as exc:
            raise self._connection_error(exc) from exc
        return self._handle(resp)

    async def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        """POST リクエストを送り JSON を返す（エラーを構造化）。"""
        try:
            resp = await self._client.post(path, json=body)
        except httpx.HTTPError as exc:
            raise self._connection_error(exc) from exc
        return self._handle(resp)

    @staticmethod
    def _connection_error(exc: Exception) -> ConversationClientError:
        """接続失敗を分かりやすいメッセージへ変換する。"""
        return ConversationClientError(
            f"サーバへの接続に失敗しました（サーバ起動を確認してください）: {exc}"
        )

    @staticmethod
    def _handle(resp: httpx.Response) -> dict[str, Any]:
        """レスポンスを検査し、成功なら JSON dict、失敗なら構造化エラーを送出する。"""
        if resp.is_success:
            data = resp.json()
            return data if isinstance(data, dict) else {}
        try:
            body = resp.json()
        except (ValueError, json.JSONDecodeError):
            body = {}
        message = body.get("message") if isinstance(body, dict) else None
        code = body.get("code") if isinstance(body, dict) else None
        raise ConversationClientError(
            message or f"サーバがエラーを返しました（status={resp.status_code}）",
            code=code,
        )


__all__ = [
    "DEFAULT_BASE_URL",
    "ApprovalRequired",
    "ConversationClient",
    "ConversationClientError",
    "PendingApproval",
    "SendResult",
    "SessionMeta",
    "StreamDone",
    "StreamToken",
]
