"""会話 CLI クライアントの plain 型（cli extra・agents 非依存）。

REST / WebSocket クライアントが受け渡す client 側 plain 型（`StreamToken` / `StreamDone` /
`PendingApproval` / `ApprovalRequired` / `SendResult` / `SessionMeta`）と例外
`ConversationClientError` を提供する。`conversation.types` とは別プロセス境界の独立定義
（意図的に統合しない）。SDK（`agents`）は import しない（NFR-1）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "ApprovalRequired",
    "ConversationClientError",
    "PendingApproval",
    "SendResult",
    "SessionMeta",
    "StreamDone",
    "StreamToken",
]


class ConversationClientError(Exception):
    """CLI クライアントが返す接続 / サーバ応答エラー。

    Attributes:
        message: 人間可読のエラーメッセージ。
        code: サーバ由来の構造化コード（あれば）。None で接続/不明エラー。
    """

    def __init__(self, message: str, *, code: str | None = None) -> None:
        """クライアントエラーを生成する。

        Args:
            message: 人間可読のエラーメッセージ。
            code: サーバ由来の構造化エラーコード（任意）。
        """
        super().__init__(message)
        self.message = message
        self.code = code


@dataclass(frozen=True)
class StreamToken:
    """ストリーミング中のテキスト断片。"""

    text: str


@dataclass(frozen=True)
class StreamDone:
    """ストリーミング完了（最終出力）。"""

    output: str


@dataclass(frozen=True)
class PendingApproval:
    """承認待ちのツール呼び出し 1 件（client 側 plain・call_id 単位）。

    Attributes:
        tool_name: 承認待ちツールの名前。
        call_id: ツール呼び出しを一意に識別する ID（承認/却下の引き当てキー）。
    """

    tool_name: str
    call_id: str


@dataclass(frozen=True)
class ApprovalRequired:
    """承認待ち発生（client 側 plain・ストリーム/REST 共通の承認待ち提示用）。

    Attributes:
        pending: 承認待ち一覧（call_id 単位の `PendingApproval`）。
    """

    pending: list[PendingApproval] = field(default_factory=list)


@dataclass(frozen=True)
class SendResult:
    """非ストリーミング 1 ターンの結果（最終応答 or 承認待ちの判別付き・client 側 plain）。

    Attributes:
        status: `"final"`（最終応答）または `"pending"`（承認待ち）。
        output: 最終応答テキスト（`status="final"` 時）。承認待ち時は None。
        pending: 承認待ち一覧（`status="pending"` 時）。最終応答時は空リスト。
    """

    status: str
    output: str | None = None
    pending: list[PendingApproval] = field(default_factory=list)


@dataclass(frozen=True)
class SessionMeta:
    """過去 session のメタ情報（一覧/復元 UI 用・サーバ /sessions レスポンス由来）。

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
