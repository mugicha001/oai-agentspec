"""L2: _adapters.make_session が in-memory / file / compaction で正しい Session を返す。

実 OpenAI は叩かず、型 / ラップ有無のみを確認する。NFR-1（SDK Session 生成は _adapters
に閉じる）の生成口を検証する。
"""

from __future__ import annotations

from typing import Any

import pytest
from agents import OpenAIResponsesCompactionSession, SQLiteSession

from oai_agentspec._adapters import make_session

pytestmark = pytest.mark.integration


class _FakeClient:
    """実 API を叩かないダミー AsyncOpenAI 互換クライアント。"""


def test_make_session_in_memory_returns_sqlite() -> None:
    """db_path 省略で in-memory な SQLiteSession を返す。"""
    session = make_session("sess-mem")
    assert isinstance(session, SQLiteSession)


def test_make_session_file_returns_sqlite(tmp_path: Any) -> None:
    """db_path 指定でファイル永続化の SQLiteSession を返す（db ファイルが作られる）。"""
    db = tmp_path / "conv.db"
    session = make_session("sess-file", db_path=str(db))
    assert isinstance(session, SQLiteSession)
    assert db.exists()


def test_make_session_compaction_disabled_returns_sqlite() -> None:
    """enable_compaction=False は client を渡しても plain SQLiteSession を返す。"""
    session = make_session(
        "sess-off", enable_compaction=False, client=_FakeClient(), model="gpt-4.1"
    )
    assert isinstance(session, SQLiteSession)


def test_make_session_compaction_without_client_raises() -> None:
    """enable_compaction=True で client 欠落のとき ValueError を送出する。"""
    with pytest.raises(ValueError, match="client"):
        make_session("sess-c", enable_compaction=True, model="gpt-4.1")


def test_make_session_compaction_wraps_with_client() -> None:
    """enable_compaction=True + client で OpenAIResponsesCompactionSession でラップする。"""
    session = make_session(
        "sess-wrap",
        enable_compaction=True,
        client=_FakeClient(),
        model="gpt-4.1",
    )
    assert isinstance(session, OpenAIResponsesCompactionSession)


def test_make_session_compaction_options_passthrough() -> None:
    """compaction_options が OpenAIResponsesCompactionSession へ素通しされる（ラップ成立）。"""
    session = make_session(
        "sess-opts",
        enable_compaction=True,
        client=_FakeClient(),
        compaction_options={"compaction_mode": "auto"},
    )
    assert isinstance(session, OpenAIResponsesCompactionSession)
