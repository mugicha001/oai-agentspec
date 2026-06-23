"""会話ストリームイベント / 会話エラーの plain 型定義（agents 非依存）。

本モジュールは openai-agents（`agents`）を一切 import しない。`_adapters` が SDK の
ストリーミングイベントを本モジュールの plain dataclass へ変換し、会話サービス・サーバ・
CLI はこの plain 型のみを扱う（NFR-1）。`protocols.py` / `spec.py` には置かない
（責務混在の回避）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


@dataclass(frozen=True)
class StreamDelta:
    """ストリーミング中のテキスト断片（逐次 token）。

    Attributes:
        text: 直近に生成されたテキスト断片。
    """

    text: str


@dataclass(frozen=True)
class StreamDone:
    """ストリーミング完了イベント（最終出力を載せる終端）。

    Attributes:
        final_output: 会話ターンの最終出力（通常はテキスト全文）。
    """

    final_output: str


@dataclass(frozen=True)
class StreamError:
    """ストリーミング中に発生したエラー（構造化済み・SDK 例外を生で漏らさない）。

    Attributes:
        code: エラー種別を表す機械可読コード（`ConversationErrorCode` の値）。
        message: 人間可読のエラーメッセージ。
    """

    code: str
    message: str


# ストリーミング会話で yield されうるイベントの Union。
StreamEvent = StreamDelta | StreamDone | StreamError


@dataclass(frozen=True)
class PendingApproval:
    """承認待ちのツール呼び出し 1 件（HITL・plain 一覧アイテム）。

    承認の最小粒度（call_id 単位）。WS の承認待ち通知 / REST の承認待ち応答・取得に共通で
    使う plain 型で、SDK の `ToolApprovalItem` には依存しない（NFR-1）。

    Attributes:
        tool_name: 承認待ちツールの名前。
        call_id: ツール呼び出しを一意に識別する ID（承認/却下の引き当てキー）。
    """

    tool_name: str
    call_id: str


@dataclass(frozen=True)
class ApprovalDecision:
    """承認/却下の判断 1 件（HITL・call_id 単位の型付き入力）。

    `ConversationService.resolve_approvals` / `stream_resolve` へ渡す承認/却下の型付き入力。
    生 dict（`{"call_id", "decision", "rejection_message"}`）も後方互換で受け付けるが、本型を
    使うとキー名・値（approve/reject）の取り違えを防げる。`approve=True` で承認、False で却下。

    Attributes:
        call_id: 対象ツール呼び出しの ID（`PendingApproval.call_id` に対応）。
        approve: True で承認（ツール実行）、False で却下（非実行）。
        rejection_message: 却下時にモデルへ返す任意のメッセージ。承認時は無視される。
    """

    call_id: str
    approve: bool
    rejection_message: str | None = None


class SendStatus(StrEnum):
    """非ストリーミング 1 ターン結果の状態（`SendResult.status`）。

    `StrEnum` のため文字列比較（`status == "final"`）も従来どおり成立する（後方互換）。

    Attributes:
        FINAL: 最終応答が得られた（`SendResult.output` にテキスト）。
        PENDING: 承認待ちが発生した（`SendResult.pending` に一覧）。
    """

    FINAL = "final"
    PENDING = "pending"


@dataclass(frozen=True)
class SendResult:
    """非ストリーミング 1 ターンの結果（最終応答 or 承認待ちの判別付き・D-Disc）。

    既存の文字列戻りを破壊せず承認待ちを表現するための plain 判別型。`status="final"` なら
    `output` に最終応答テキスト・`pending` は空。`status="pending"` なら `output` は None・
    `pending` に承認待ち一覧（call_id 単位）。REST 層がこの判別を `SendResponse` の status/
    pending フィールドへ写す。

    Attributes:
        status: `SendStatus.FINAL`（最終応答）または `SendStatus.PENDING`（承認待ち）。
            `StrEnum` のため `"final"` / `"pending"` の文字列比較とも互換。
        output: 最終応答テキスト。承認待ち時は None。
        pending: 承認待ち一覧（call_id 単位の `PendingApproval`）。最終応答時は空リスト。
    """

    status: SendStatus
    output: str | None = None
    pending: list[PendingApproval] = field(default_factory=list)


@dataclass(frozen=True)
class ApprovalRequired:
    """承認待ち発生を表す専用イベント（**`StreamEvent` Union には混ぜない**・NFR-6）。

    ストリーミング会話で承認必須ツールの呼び出しが発生したターンに、`StreamDelta` の後に
    `StreamDone` の代わりとして 1 件流す専用イベント。既存 3 メンバ（StreamDelta / StreamDone /
    StreamError）網羅前提の消費者を壊さないよう、`StreamEvent` とは別型で表現する（D-Compat）。

    Attributes:
        approvals: 承認待ち一覧（call_id 単位の `PendingApproval`）。
    """

    approvals: list[PendingApproval]


@dataclass(frozen=True)
class SessionInfo:
    """永続化された過去 session のメタ情報（一覧/復元 UI 用・D5）。

    SDK Session db から導出した読み取り専用サマリー。CLI のセッション選択画面で
    更新時刻・ターン数・先頭発話プレビューを表示するために使う。

    Attributes:
        session_id: 過去会話の session_id（復元キー）。
        updated_at: 最終更新時刻の文字列（SDK の TIMESTAMP・空文字なら不明）。
        turn_count: assistant 応答数（おおよその会話往復数）。
        preview: 先頭 user 発話のテキストプレビュー（改行除去・空なら未取得）。
    """

    session_id: str
    updated_at: str
    turn_count: int
    preview: str


class ConversationErrorCode(StrEnum):
    """会話エラーの機械可読コード。

    REST のステータス対応（`serve.app._ERROR_STATUS`）や WS error メッセージの分岐に使う。
    """

    UNKNOWN_AGENT = "unknown_agent"
    UNKNOWN_CONVERSATION = "unknown_conversation"
    CONVERSATION_ALREADY_EXISTS = "conversation_already_exists"
    MODEL_NOT_CONFIGURED = "model_not_configured"
    EXECUTION_ERROR = "execution_error"
    # HITL（承認）系。WS / REST 共通コード（D-Err）。
    UNKNOWN_APPROVAL = "unknown_approval"
    APPROVAL_ALREADY_RESOLVED = "approval_already_resolved"
    NO_PENDING_APPROVAL = "no_pending_approval"


class ConversationError(Exception):
    """会話サービスが送出する構造化エラー（SDK 例外を生で漏らさないための変換先）。

    Attributes:
        code: エラー種別（`ConversationErrorCode`）。
        message: 人間可読のエラーメッセージ。
    """

    def __init__(self, code: ConversationErrorCode, message: str) -> None:
        """会話エラーを生成する。

        Args:
            code: エラー種別コード。
            message: 人間可読メッセージ。
        """
        super().__init__(message)
        self.code = code
        self.message = message

    def to_stream_error(self) -> StreamError:
        """本エラーを `StreamError`（plain）へ変換する。

        Returns:
            code / message を引き継いだ `StreamError`。
        """
        return StreamError(code=self.code.value, message=self.message)


__all__ = [
    "ApprovalDecision",
    "ApprovalRequired",
    "ConversationError",
    "ConversationErrorCode",
    "PendingApproval",
    "SendResult",
    "SendStatus",
    "SessionInfo",
    "StreamDelta",
    "StreamDone",
    "StreamError",
    "StreamEvent",
]
