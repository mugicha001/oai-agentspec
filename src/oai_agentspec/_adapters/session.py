"""SDK Session 生成・破棄アダプタ（SDK 結合を `_adapters` に閉じる・NFR-1）。

`make_session`（SQLite / compaction ラッパー生成）/ `close_session`（破棄）を提供する。SDK 結合
（`agents` の `Session` 系）は本モジュール内に閉じ、外へは不透明な `Session` のみを渡す。会話履歴
の読み取り（列挙 / メタ / 履歴）は `_session_store`、HITL 承認待ちの永続 CRUD は `_pending_store`
に分離する。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agents import (
    OpenAIResponsesCompactionSession,
    Session,
    SQLiteSession,
)

if TYPE_CHECKING:
    from openai import AsyncOpenAI


def make_session(
    session_id: str,
    *,
    db_path: str | None = None,
    enable_compaction: bool = False,
    client: AsyncOpenAI | None = None,
    model: str | None = None,
    compaction_options: dict[str, Any] | None = None,
) -> Session:
    """会話履歴用の SDK `Session` を生成する（SQLite・任意で compaction ラップ）。

    `db_path` が None なら in-memory（`":memory:"`・揮発）、パスならファイル永続化
    （再起動後 resume 可）。`enable_compaction=True` のときのみ `client`（必須）/ `model`
    （任意）で `OpenAIResponsesCompactionSession` でラップして返す（OpenAI Responses 専用）。
    `enable_compaction=False`（既定）なら client/model の指定有無にかかわらず plain な
    `SQLiteSession` を返す（暗黙有効化を行わない）。`model` が None ならキーを渡さず SDK 既定
    モデルを使う。`compaction_options` は `OpenAIResponsesCompactionSession` へ素通しする。

    Args:
        session_id: SDK `Session` の session_id。
        db_path: SQLite db ファイルパス。None で in-memory（`":memory:"`）。
        enable_compaction: compaction を有効化するか（既定 False）。False なら plain SQLite。
        client: OpenAI Responses 互換クライアント（`AsyncOpenAI` 系）。有効化時に必須。
        model: 圧縮に使うモデル名。None なら SDK 既定モデルを使う（キーを渡さない）。
        compaction_options: `OpenAIResponsesCompactionSession` へ素通しする追加オプション。

    Returns:
        SDK `Session`（`SQLiteSession` または `OpenAIResponsesCompactionSession`）。

    Raises:
        ValueError: `enable_compaction=True` かつ `client` が欠けている場合。
    """
    base = SQLiteSession(session_id, db_path or ":memory:")
    if not enable_compaction:
        return base
    if client is None:
        raise ValueError("compaction を有効化する場合は client（AsyncOpenAI 系）が必須です")
    opts = dict(compaction_options or {})
    if model is not None:
        opts["model"] = model
    return OpenAIResponsesCompactionSession(session_id, base, client=client, **opts)


async def close_session(session: Any) -> None:
    """破棄する SDK Session を閉じる（`close` を持つ場合のみ・同期/非同期両対応）。

    file-backed `SQLiteSession` は sqlite 接続を抱えるため、登録されず破棄される Session
    （会話 ID 重複時など）はこの関数で閉じてリソースリークを防ぐ。`close` を持たない Session
    （compaction ラッパー等）は何もしない（ベストエフォート）。

    Args:
        session: 閉じる SDK Session（不透明値）。
    """
    close = getattr(session, "close", None)
    if close is None:
        return
    result = close()
    if hasattr(result, "__await__"):  # 非同期 close にも対応
        await result
