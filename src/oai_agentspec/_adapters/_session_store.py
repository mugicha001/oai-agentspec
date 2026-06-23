"""SQLite 会話履歴の読み取りアダプタ（SDK 内部スキーマ結合を `_adapters` に閉じる・NFR-1）。

`list_session_ids`（永続化 session の列挙）/ `list_session_meta`（メタ情報列挙）/
`get_session_items`（履歴アイテム取得）と、それらが共有する sqlite ローカルヘルパ
（`_connect` / `_fetch_or_empty`）・コンテンツ整形ヘルパ（`_content_text` / `_decode_items`）を
提供する。SDK の `SQLiteSession` が書く既定スキーマ（テーブル名 / 列名）への結合は本モジュールの
定数群と各関数に局在化する（D5・NFR-7 トリップワイヤで監視）。
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = [
    "get_session_items",
    "list_session_ids",
    "list_session_meta",
]


# SQLiteSession の既定スキーマ前提（sessions / messages テーブルと列名）。SDK の内部スキーマ
# への結合はこの定数群と本モジュールの列挙/メタ/履歴関数に閉じる（D5・NFR-7 トリップワイヤで
# 監視）。messages テーブルは 1 行 = 1 履歴アイテム（message_data に JSON 1 件）。
_SQLITE_SESSIONS_TABLE = "agent_sessions"
_SQLITE_SESSION_ID_COLUMN = "session_id"
_SQLITE_SESSIONS_UPDATED_COLUMN = "updated_at"
_SQLITE_MESSAGES_TABLE = "agent_messages"
_SQLITE_MESSAGES_ID_COLUMN = "id"
_SQLITE_MESSAGES_DATA_COLUMN = "message_data"


# 以下は本モジュール内に閉じた sqlite ボイラープレートのローカルヘルパ（NFR-1/NFR-7 境界を
# 越えて src/utils 等へ出さない）。SDK 内部スキーマ結合は各列挙/CRUD 関数と定数群に局在化する。
@contextmanager
def _connect(db_path: str) -> Iterator[sqlite3.Connection]:
    """SQLite 接続を開き、終了時に確実に閉じる contextmanager（本モジュール限定）。

    `connect → try → finally close` の反復を 1 箇所に括る。接続生成・クローズのみを担い、
    `OperationalError` の握り潰しは呼び出し側 / `_fetch_or_empty` の責務とする（挙動不変）。

    Args:
        db_path: SQLite db のファイルパス。

    Yields:
        開いた `sqlite3.Connection`（with ブロックを抜けると close）。
    """
    connection = sqlite3.connect(db_path)
    try:
        yield connection
    finally:
        connection.close()


def _fetch_or_empty(
    db_path: str, query: str, params: tuple[Any, ...] = ()
) -> list[tuple[Any, ...]]:
    """読み取り SELECT を実行し行を返す（db 不在 / テーブル未作成は空リスト・本モジュール限定）。

    `os.path.exists 早期 return` と `except OperationalError → []` のボイラープレートを括る。
    単一 SELECT で全行を取る読み取り関数（list_session_ids / get_session_items /
    list_pending_approvals）が共通利用する。複数クエリ / 単一行 fetchone / 書き込みは各関数が
    `_connect` を直接使う（挙動不変）。

    Args:
        db_path: SQLite db のファイルパス。
        query: 実行する SELECT 文。
        params: バインドパラメータ（省略可）。

    Returns:
        取得行（タプル）の列。db ファイルが無い / テーブルが無い場合は空リスト。
    """
    if not os.path.exists(db_path):
        return []
    with _connect(db_path) as connection:
        try:
            return connection.execute(query, params).fetchall()
        except sqlite3.OperationalError:
            # テーブル未作成（まだ永続化されていない）等。
            return []


def list_session_ids(db_path: str) -> list[str]:
    """ファイル永続化された過去 session の session_id を列挙する（D5）。

    SDK の `SQLiteSession` が書く SQLite db を標準ライブラリ `sqlite3` で開き、既定の
    sessions テーブル（`agent_sessions`）の `session_id` 列を素の SELECT で列挙する。自前の
    別インデックスは持たず SDK と同一 db を読むだけ（NFR-2）。SDK 内部スキーマ（テーブル名 /
    列名）への結合はこの関数に閉じ、NFR-7 トリップワイヤで前提崩れを監視する。

    Args:
        db_path: `SQLiteSession` が永続化した SQLite db のファイルパス。

    Returns:
        session_id の昇順リスト。db ファイルが無い / sessions テーブルが無い場合は空リスト。
    """
    query = (
        f"SELECT {_SQLITE_SESSION_ID_COLUMN} FROM {_SQLITE_SESSIONS_TABLE} "
        f"ORDER BY {_SQLITE_SESSION_ID_COLUMN}"
    )
    return [str(row[0]) for row in _fetch_or_empty(db_path, query)]


def _content_text(content: Any) -> str:
    """履歴アイテムの content フィールドをテキスト文字列へ変換する。

    SDK の content は `str` または `[{"text": "..."}, ...]` 形式の `list[dict]` を取り得る
    （`output_text` 等）。両者を吸収して連結した文字列を返す。

    Args:
        content: 履歴アイテムの content（str / list / その他）。

    Returns:
        テキスト化した文字列。None / 未知型は空文字または str() 変換結果。
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(content)


def _decode_items(rows: list[tuple[Any, ...]]) -> list[dict[str, Any]]:
    """message_data 行（JSON 文字列）の集合を dict 履歴アイテムの列へデコードする。

    JSON デコード失敗・dict 以外の要素はベストエフォートでスキップする（破損行で全体を
    落とさない）。

    Args:
        rows: `(message_data,)` タプルの列。

    Returns:
        デコードできた履歴アイテム（dict）の列。
    """
    items: list[dict[str, Any]] = []
    for (raw,) in rows:
        try:
            item = json.loads(raw)
        except (TypeError, ValueError):
            continue
        if isinstance(item, dict):
            items.append(item)
    return items


def list_session_meta(db_path: str) -> list[dict[str, Any]]:
    """永続化された過去 session のメタ情報（更新時刻 / ターン数 / プレビュー）を列挙する。

    SDK の `agent_sessions`（session_id / updated_at）と `agent_messages`（履歴アイテム）を
    素 SELECT し、各 session について最終更新時刻・assistant 応答数（turn_count）・先頭 user
    発話（preview）を導出する。`updated_at` の降順（最近の会話が先頭）で返す。SDK 内部スキーマ
    への結合は本モジュールに閉じ、NFR-7 トリップワイヤで前提崩れを監視する（D5）。

    Args:
        db_path: `SQLiteSession` が永続化した SQLite db のファイルパス。

    Returns:
        `{"session_id": str, "updated_at": str, "turn_count": int, "preview": str}` の列
        （updated_at 降順）。db / テーブルが無い場合は空リスト。
    """
    if not os.path.exists(db_path):
        return []
    sessions_query = (
        f"SELECT {_SQLITE_SESSION_ID_COLUMN}, {_SQLITE_SESSIONS_UPDATED_COLUMN} "
        f"FROM {_SQLITE_SESSIONS_TABLE} "
        f"ORDER BY {_SQLITE_SESSIONS_UPDATED_COLUMN} DESC, {_SQLITE_SESSION_ID_COLUMN}"
    )
    messages_query = (
        f"SELECT {_SQLITE_MESSAGES_DATA_COLUMN} FROM {_SQLITE_MESSAGES_TABLE} "
        f"WHERE {_SQLITE_SESSION_ID_COLUMN} = ? ORDER BY {_SQLITE_MESSAGES_ID_COLUMN} ASC"
    )
    with _connect(db_path) as connection:
        try:
            session_rows = connection.execute(sessions_query).fetchall()
        except sqlite3.OperationalError:
            return []
        result: list[dict[str, Any]] = []
        for session_id, updated_at in session_rows:
            try:
                message_rows = connection.execute(messages_query, (session_id,)).fetchall()
            except sqlite3.OperationalError:
                message_rows = []
            items = _decode_items(message_rows)
            turn_count = sum(1 for item in items if item.get("role") == "assistant")
            preview = ""
            for item in items:
                if item.get("role") == "user":
                    preview = _content_text(item.get("content", "")).strip().replace("\n", " ")
                    break
            result.append(
                {
                    "session_id": str(session_id),
                    "updated_at": "" if updated_at is None else str(updated_at),
                    "turn_count": turn_count,
                    "preview": preview,
                }
            )
        return result


def get_session_items(
    db_path: str, session_id: str, *, limit: int | None = None
) -> list[dict[str, Any]]:
    """指定 session の履歴アイテムを時系列で返す（任意で直近 limit 件に限定）。

    `agent_messages` を `id` 昇順（=時系列）に読む。`limit` 指定時は直近（末尾）`limit` 件
    だけを時系列順で返す（SDK `get_items(limit)` と同じ「最近 N 件」セマンティクス）。会話
    復元時の履歴プレビュー表示に使う。

    Args:
        db_path: `SQLiteSession` が永続化した SQLite db のファイルパス。
        session_id: 取得対象の session_id。
        limit: 返す最大件数（直近側）。None で全件。

    Returns:
        履歴アイテム（dict）の時系列リスト。db / テーブルが無い場合は空リスト。
    """
    if limit is not None:
        # 直近 limit 件を取るため id 降順 LIMIT で引き、時系列へ反転する。
        query = (
            f"SELECT {_SQLITE_MESSAGES_DATA_COLUMN} FROM {_SQLITE_MESSAGES_TABLE} "
            f"WHERE {_SQLITE_SESSION_ID_COLUMN} = ? "
            f"ORDER BY {_SQLITE_MESSAGES_ID_COLUMN} DESC LIMIT ?"
        )
        params: tuple[Any, ...] = (session_id, limit)
    else:
        query = (
            f"SELECT {_SQLITE_MESSAGES_DATA_COLUMN} FROM {_SQLITE_MESSAGES_TABLE} "
            f"WHERE {_SQLITE_SESSION_ID_COLUMN} = ? ORDER BY {_SQLITE_MESSAGES_ID_COLUMN} ASC"
        )
        params = (session_id,)
    rows = _fetch_or_empty(db_path, query, params)
    if limit is not None:
        rows = list(reversed(rows))
    return _decode_items(rows)
