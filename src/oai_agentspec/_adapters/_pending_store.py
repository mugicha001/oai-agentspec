"""HITL 承認待ち（中断状態）の永続テーブル CRUD アダプタ（SDK 結合を `_adapters` に閉じる・NFR-1）。

`save_pending_approval` / `load_pending_approval` / `delete_pending_approval` /
`list_pending_approvals` と、それらが使う専用テーブルの定数群・スキーマ自己修復ヘルパ
（`_ensure_pending_table`）を提供する。承認待ちの中断状態を会話履歴 db に同居させる専用テーブル
（方針B・D-Table）を扱い、SDK の会話履歴テーブル（`agent_sessions` / `agent_messages`）には一切
触れない。sqlite ローカルヘルパ（`_connect` / `_fetch_or_empty`）は `_session_store` を共有する。
"""

from __future__ import annotations

import os
import sqlite3
from datetime import UTC, datetime
from typing import Any

from ._session_store import _connect, _fetch_or_empty

__all__ = [
    "delete_pending_approval",
    "list_pending_approvals",
    "load_pending_approval",
    "save_pending_approval",
]


# HITL（承認待ち）の中断状態を会話履歴 db に同居させる専用テーブル（方針B・D-Table）。SDK の
# agent_sessions / agent_messages と衝突しない接頭辞（oai_agentspec_）を持つ。
# run_state 列に RunState の
# JSON（NFR-7 トリップワイヤ監視対象）、pending 列に承認待ち一覧（call_id / tool_name）の JSON。
# 主キーは session_id（FR-10: CLI のセッション復元は同 session_id に新 conversation_id を振るため、
# session_id をキーにしないと再起動跨ぎで承認待ちが見えない）。agent_name 列に RunState を生んだ
# 解決済みエージェント名を保存し、復元時の initial_agent 解決（D-Resume）に使う。
_PENDING_TABLE = "oai_agentspec_pending_approvals"
_PENDING_SESSION_ID_COLUMN = "session_id"
_PENDING_AGENT_NAME_COLUMN = "agent_name"
_PENDING_RUN_STATE_COLUMN = "run_state"
_PENDING_PENDING_COLUMN = "pending"
_PENDING_UPDATED_COLUMN = "updated_at"

_PENDING_CREATE_TABLE = (
    f"CREATE TABLE IF NOT EXISTS {_PENDING_TABLE} ("
    f"{_PENDING_SESSION_ID_COLUMN} TEXT PRIMARY KEY, "
    f"{_PENDING_AGENT_NAME_COLUMN} TEXT, "
    f"{_PENDING_RUN_STATE_COLUMN} TEXT, "
    f"{_PENDING_PENDING_COLUMN} TEXT, "
    f"{_PENDING_UPDATED_COLUMN} TEXT"
    ")"
)

# 期待する列集合（スキーマ進化を検知して旧テーブルを作り直す自己修復用）。
_PENDING_EXPECTED_COLUMNS = frozenset(
    {
        _PENDING_SESSION_ID_COLUMN,
        _PENDING_AGENT_NAME_COLUMN,
        _PENDING_RUN_STATE_COLUMN,
        _PENDING_PENDING_COLUMN,
        _PENDING_UPDATED_COLUMN,
    }
)


def _ensure_pending_table(connection: sqlite3.Connection) -> None:
    """承認待ちテーブルを期待スキーマで用意する（旧スキーマは作り直して自己修復する）。

    `oai_agentspec_pending_approvals` は実行中の承認待ち（一時状態）を保持する自前テーブル。本機能の
    スキーマ進化（主キー / 列）で、既存 db に残る旧スキーマのテーブルだと INSERT が「列が無い」で
    失敗しうる。一時状態のため、列集合が期待と異なれば DROP して作り直し自己修復する。SDK の
    会話履歴テーブル（`agent_sessions` / `agent_messages`）には一切触れない。

    Args:
        connection: 対象 db への接続。
    """
    rows = connection.execute(f"PRAGMA table_info({_PENDING_TABLE})").fetchall()
    if rows:
        columns = {str(row[1]) for row in rows}
        if columns != _PENDING_EXPECTED_COLUMNS:
            connection.execute(f"DROP TABLE {_PENDING_TABLE}")
    connection.execute(_PENDING_CREATE_TABLE)


def save_pending_approval(
    db_path: str,
    session_id: str,
    agent_name: str,
    run_state_json: str,
    pending_json: str,
) -> None:
    """承認待ち（中断状態）を専用テーブルへ upsert する（テーブル無ければ CREATE・FR-10）。

    `session_id` を主キーに、`agent_name`（RunState を生んだ解決済みエージェント名）/ `run_state`
    （RunState の JSON）/ `pending`（承認待ち一覧の JSON）/ 更新時刻を保存する。同一 session_id は
    上書き（段階解決・再中断で更新される）。session_id をキーにするのは、CLI のセッション復元が
    同 session_id に新 conversation_id を振るため（再起動跨ぎ復元・FR-10）。保存は冪等で再試行に
    耐える（NFR-4(a)）。

    Args:
        db_path: 会話履歴 db のファイルパス（`SQLiteSession` と同居）。
        session_id: 紐づく SDK Session の session_id（主キー）。
        agent_name: RunState を生んだ解決済みエージェント名（復元時 initial_agent 解決用）。
        run_state_json: `RunState.to_string()` の JSON 文字列。
        pending_json: 承認待ち一覧（call_id / tool_name）の JSON 文字列。
    """
    with _connect(db_path) as connection:
        _ensure_pending_table(connection)
        connection.execute(
            f"INSERT INTO {_PENDING_TABLE} ("
            f"{_PENDING_SESSION_ID_COLUMN}, {_PENDING_AGENT_NAME_COLUMN}, "
            f"{_PENDING_RUN_STATE_COLUMN}, {_PENDING_PENDING_COLUMN}, "
            f"{_PENDING_UPDATED_COLUMN}) VALUES (?, ?, ?, ?, ?) "
            f"ON CONFLICT({_PENDING_SESSION_ID_COLUMN}) DO UPDATE SET "
            f"{_PENDING_AGENT_NAME_COLUMN}=excluded.{_PENDING_AGENT_NAME_COLUMN}, "
            f"{_PENDING_RUN_STATE_COLUMN}=excluded.{_PENDING_RUN_STATE_COLUMN}, "
            f"{_PENDING_PENDING_COLUMN}=excluded.{_PENDING_PENDING_COLUMN}, "
            f"{_PENDING_UPDATED_COLUMN}=excluded.{_PENDING_UPDATED_COLUMN}",
            (
                session_id,
                agent_name,
                run_state_json,
                pending_json,
                datetime.now(UTC).isoformat(),
            ),
        )
        connection.commit()


def load_pending_approval(db_path: str, session_id: str) -> dict[str, Any] | None:
    """承認待ち（中断状態）を専用テーブルから読み出す（無ければ None・復元用・FR-10）。

    Args:
        db_path: 会話履歴 db のファイルパス。
        session_id: 対象 session_id（主キー）。

    Returns:
        `{"session_id", "agent_name", "run_state", "pending", "updated_at"}` の plain dict。
        db / テーブル / 行が無ければ None。
    """
    if not os.path.exists(db_path):
        return None
    query = (
        f"SELECT {_PENDING_SESSION_ID_COLUMN}, {_PENDING_AGENT_NAME_COLUMN}, "
        f"{_PENDING_RUN_STATE_COLUMN}, {_PENDING_PENDING_COLUMN}, "
        f"{_PENDING_UPDATED_COLUMN} FROM {_PENDING_TABLE} "
        f"WHERE {_PENDING_SESSION_ID_COLUMN} = ?"
    )
    with _connect(db_path) as connection:
        try:
            row = connection.execute(query, (session_id,)).fetchone()
        except sqlite3.OperationalError:
            # 承認待ちテーブル未作成（まだ中断が発生していない）等。
            return None
        if row is None:
            return None
        return {
            "session_id": str(row[0]),
            "agent_name": "" if row[1] is None else str(row[1]),
            "run_state": "" if row[2] is None else str(row[2]),
            "pending": "" if row[3] is None else str(row[3]),
            "updated_at": "" if row[4] is None else str(row[4]),
        }


def delete_pending_approval(db_path: str, session_id: str) -> None:
    """承認待ち（中断状態）を専用テーブルから削除する（session_id 条件付き・FR-10）。

    再開完了時に呼ぶ。`session_id` を条件に含めた DELETE で別 session の中断状態を取り違えて
    消す事故を防ぐ（NFR-4(b)）。削除は冪等（対象が無くても例外にしない）。

    Args:
        db_path: 会話履歴 db のファイルパス。
        session_id: 削除対象の session_id（主キー）。
    """
    if not os.path.exists(db_path):
        return
    with _connect(db_path) as connection:
        try:
            connection.execute(
                f"DELETE FROM {_PENDING_TABLE} WHERE {_PENDING_SESSION_ID_COLUMN} = ?",
                (session_id,),
            )
            connection.commit()
        except sqlite3.OperationalError:
            # テーブル未作成なら削除対象も無い（冪等）。
            return


def list_pending_approvals(db_path: str) -> list[dict[str, Any]]:
    """専用テーブルの全承認待ち（中断状態）を列挙する（運用・デバッグ用）。

    Args:
        db_path: 会話履歴 db のファイルパス。

    Returns:
        `{"session_id", "agent_name", "pending", "updated_at"}` の列（updated_at 降順）。
        db / テーブルが無ければ空リスト（`run_state` JSON は重いため含めない）。
    """
    query = (
        f"SELECT {_PENDING_SESSION_ID_COLUMN}, {_PENDING_AGENT_NAME_COLUMN}, "
        f"{_PENDING_PENDING_COLUMN}, {_PENDING_UPDATED_COLUMN} FROM {_PENDING_TABLE} "
        f"ORDER BY {_PENDING_UPDATED_COLUMN} DESC, {_PENDING_SESSION_ID_COLUMN}"
    )
    return [
        {
            "session_id": str(row[0]),
            "agent_name": "" if row[1] is None else str(row[1]),
            "pending": "" if row[2] is None else str(row[2]),
            "updated_at": "" if row[3] is None else str(row[3]),
        }
        for row in _fetch_or_empty(db_path, query)
    ]
