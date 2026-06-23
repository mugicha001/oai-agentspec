"""REST / WebSocket の JSON スキーマ（Pydantic・serve 専用・agents 非依存）。

会話サービスの plain 型と HTTP/WS 境界の橋渡しに使う。SDK（`agents`）は import しない
（NFR-1）。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class AgentsResponse(BaseModel):
    """エージェント一覧レスポンス。

    Attributes:
        agents: 登録済みエージェント名（昇順）。
        entry: エントリ（起点）エージェント名。未決定（registry 空）なら None。
    """

    agents: list[str] = Field(default_factory=list)
    entry: str | None = None


class SessionMeta(BaseModel):
    """過去 session のメタ情報（一覧/復元 UI 用・D5）。

    Attributes:
        session_id: 過去会話の session_id（復元キー）。
        updated_at: 最終更新時刻の文字列（空文字なら不明）。
        turn_count: assistant 応答数（おおよその往復数）。
        preview: 先頭 user 発話のテキストプレビュー。
    """

    session_id: str
    updated_at: str = ""
    turn_count: int = 0
    preview: str = ""


class SessionsResponse(BaseModel):
    """永続化済み session 一覧レスポンス（D5・更新時刻降順）。

    Attributes:
        sessions: 過去会話のメタ情報（最終更新の新しい順）。
    """

    sessions: list[SessionMeta] = Field(default_factory=list)


class HistoryResponse(BaseModel):
    """過去 session の履歴アイテムレスポンス（復元時の表示用・D5）。

    Attributes:
        session_id: 対象 session_id。
        items: 履歴アイテム（SDK の入力アイテム dict）の時系列リスト（直近側に限定されうる）。
    """

    session_id: str
    items: list[dict[str, Any]] = Field(default_factory=list)


class CreateConversationRequest(BaseModel):
    """会話作成リクエスト。

    Attributes:
        conversation_id: 会話 ID。None でサーバ自動採番。
        session_id: SDK Session の session_id。明示でファイル永続化、None で in-memory。
    """

    conversation_id: str | None = None
    session_id: str | None = None


class CreateConversationResponse(BaseModel):
    """会話作成レスポンス。

    Attributes:
        conversation_id: 作成された会話 ID。
    """

    conversation_id: str


class SendRequest(BaseModel):
    """非ストリーミング会話リクエスト。

    Attributes:
        agent_name: 会話相手のエージェント名。None / 省略でエントリエージェント起点。
        text: ユーザー入力テキスト。
    """

    agent_name: str | None = None
    text: str


class PendingApprovalSchema(BaseModel):
    """承認待ちのツール呼び出し 1 件（HITL・call_id 単位・D-Disc）。

    Attributes:
        tool_name: 承認待ちツールの名前。
        call_id: ツール呼び出しを一意に識別する ID（承認/却下の引き当てキー）。
    """

    tool_name: str
    call_id: str


class SendResponse(BaseModel):
    """非ストリーミング会話レスポンス（最終応答 or 承認待ちの判別付き・D-Disc）。

    既存の `output` を破壊せず判別フィールドを追加する。`status="final"`（既定）なら
    `output` に最終応答テキストを従来どおり載せ `pending` は None。`status="pending"` なら
    承認待ちで `output` は None、`pending` に承認待ち一覧（call_id 単位）を載せる。承認待ちを
    扱わない既存消費者は `status` 既定 "final" + `output` で従来どおり動作する（NFR-6）。

    Attributes:
        output: 最終応答テキスト（`status="final"` 時）。承認待ち時は None。
        status: `"final"`（最終応答）または `"pending"`（承認待ち）。既定 "final"。
        pending: 承認待ち一覧（call_id 単位）。`status="final"` 時は None。
    """

    output: str | None = None
    status: str = "final"
    pending: list[PendingApprovalSchema] | None = None


class ApprovalsResponse(BaseModel):
    """承認待ち一覧レスポンス（冪等取得・復元直後の再提示用・D-RestGet）。

    Attributes:
        conversation_id: 対象会話 ID。
        pending: 現在の承認待ち一覧（call_id 単位）。中断なしなら空リスト。
    """

    conversation_id: str
    pending: list[PendingApprovalSchema] = Field(default_factory=list)


class ApprovalDecisionSchema(BaseModel):
    """承認/却下の 1 決定（call_id 単位・D-WsMsg）。

    Attributes:
        call_id: 対象のツール呼び出し ID。
        decision: `"approve"`（承認）または `"reject"`（却下）。`Literal` で REST 入力を型検証し、
            未知値は 422 で弾く（fail-closed の一環・NFR-7）。
        rejection_message: 却下理由（`decision="reject"` 時の任意・会話継続入力へ反映）。
    """

    call_id: str
    decision: Literal["approve", "reject"]
    rejection_message: str | None = None


class ApprovalRequest(BaseModel):
    """承認/却下リクエスト（decisions 配列・部分指定可・段階解決・D-WsMsg）。

    Attributes:
        decisions: 適用する承認/却下の配列。未指定 call_id は未解決のまま残る（FR-7）。
    """

    decisions: list[ApprovalDecisionSchema] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    """構造化エラーレスポンスのボディ。

    Attributes:
        code: 機械可読のエラーコード（`ConversationErrorCode` の値）。
        message: 人間可読のエラーメッセージ。
    """

    code: str
    message: str


__all__ = [
    "AgentsResponse",
    "ApprovalDecisionSchema",
    "ApprovalRequest",
    "ApprovalsResponse",
    "CreateConversationRequest",
    "CreateConversationResponse",
    "ErrorResponse",
    "HistoryResponse",
    "PendingApprovalSchema",
    "SendRequest",
    "SendResponse",
    "SessionMeta",
    "SessionsResponse",
]
