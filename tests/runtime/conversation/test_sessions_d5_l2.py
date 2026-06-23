"""L2: D5（session 一覧 / 復元）と NFR-7 トリップワイヤを検証する。

`_adapters.list_session_ids` の素 SELECT 列挙、`ConversationService.list_sessions` / 復元
（過去履歴を踏まえた継続）を FakeModel + SQLiteSession(tmpfile) でオフライン検証する。
NFR-7 トリップワイヤは SDK の SQLiteSession スキーマ前提（テーブル名 / 列名）を突く。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest
from agents import SQLiteSession

from oai_agentspec import AgentRegistry, AgentSpec
from oai_agentspec._adapters import get_session_items, list_session_ids, list_session_meta
from oai_agentspec.runtime.conversation import ConversationService, SessionPolicy

from _helpers.fake_model import FakeModel

pytestmark = pytest.mark.integration


# ----------------------------------------------------------------------
# (1) _adapters.list_session_ids（素 SELECT 列挙）
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_session_ids_enumerates_persisted_sessions(tmp_path: Path) -> None:
    """ファイル永続化された session_id が昇順で列挙される。"""
    db = str(tmp_path / "conv.db")
    await SQLiteSession("sess-b", db).add_items([{"role": "user", "content": "hi"}])
    await SQLiteSession("sess-a", db).add_items([{"role": "user", "content": "yo"}])
    assert list_session_ids(db) == ["sess-a", "sess-b"]


def test_list_session_ids_missing_db_returns_empty(tmp_path: Path) -> None:
    """db ファイルが無ければ空リストを返す（例外にしない）。"""
    assert list_session_ids(str(tmp_path / "absent.db")) == []


@pytest.mark.asyncio
async def test_list_session_meta_derives_turns_and_preview(tmp_path: Path) -> None:
    """list_session_meta が更新時刻・ターン数・先頭発話プレビューを導出する。"""
    db = str(tmp_path / "conv.db")
    await SQLiteSession("sess-1", db).add_items(
        [
            {"role": "user", "content": "最初の質問"},
            {"role": "assistant", "content": [{"type": "output_text", "text": "回答1"}]},
            {"role": "user", "content": "次の質問"},
            {"role": "assistant", "content": [{"type": "output_text", "text": "回答2"}]},
        ]
    )
    meta = list_session_meta(db)
    assert len(meta) == 1
    assert meta[0]["session_id"] == "sess-1"
    assert meta[0]["turn_count"] == 2
    assert meta[0]["preview"] == "最初の質問"
    assert meta[0]["updated_at"]  # 何らかの更新時刻文字列


@pytest.mark.asyncio
async def test_get_session_items_limit_returns_recent_chronological(tmp_path: Path) -> None:
    """get_session_items(limit) が直近 N 件を時系列で返す。"""
    db = str(tmp_path / "conv.db")
    await SQLiteSession("sess-1", db).add_items(
        [{"role": "user", "content": f"msg-{i}"} for i in range(5)]
    )
    items = get_session_items(db, "sess-1", limit=2)
    assert [it["content"] for it in items] == ["msg-3", "msg-4"]
    # 全件取得も時系列。
    allitems = get_session_items(db, "sess-1")
    assert [it["content"] for it in allitems] == [f"msg-{i}" for i in range(5)]


def test_session_meta_missing_db_returns_empty(tmp_path: Path) -> None:
    """db ファイルが無ければ list_session_meta / get_session_items は空。"""
    absent = str(tmp_path / "absent.db")
    assert list_session_meta(absent) == []
    assert get_session_items(absent, "x", limit=10) == []


def test_list_session_ids_db_without_sessions_table_returns_empty(tmp_path: Path) -> None:
    """sessions テーブルが無い db は空リストを返す（OperationalError を握る）。"""
    db = str(tmp_path / "other.db")
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE unrelated (x INTEGER)")
    con.commit()
    con.close()
    assert list_session_ids(db) == []


# ----------------------------------------------------------------------
# (2) ConversationService.list_sessions / 復元
# ----------------------------------------------------------------------


def _service(tmp_path: Path, model: Any) -> ConversationService:
    """tmp ディレクトリを永続化先にした FakeModel エージェント付きサービスを作る。"""
    reg = AgentRegistry()
    reg.register(AgentSpec(name="bot", instructions="b", model=model))
    policy = SessionPolicy(base_dir=tmp_path, db_name="conversations.db")
    return ConversationService(reg, session_policy=policy)


@pytest.mark.asyncio
async def test_list_sessions_lists_only_persisted(tmp_path: Path) -> None:
    """file 永続化会話のみ列挙され、in-memory（無指定）会話は対象外。"""
    svc = _service(tmp_path, FakeModel().queue_text("ok").queue_text("ok"))
    # 永続化会話（session_id 明示）。
    cid_file = await svc.create_conversation(session_id="named-1")
    await svc.send("bot", "hello", conversation_id=cid_file)
    # in-memory 会話（session_id 無指定・揮発）。
    cid_mem = await svc.create_conversation()
    await svc.send("bot", "hi", conversation_id=cid_mem)

    listed = await svc.list_sessions()
    listed_ids = [info.session_id for info in listed]
    assert "named-1" in listed_ids
    # in-memory 会話の session_id（= conversation_id）は列挙されない。
    assert cid_mem not in listed_ids
    # メタ情報（ターン数 / プレビュー）が導出されている。
    named = next(info for info in listed if info.session_id == "named-1")
    assert named.turn_count == 1
    assert named.preview == "hello"


@pytest.mark.asyncio
async def test_list_sessions_empty_when_no_persistence(tmp_path: Path) -> None:
    """永続化会話が無ければ空リスト。"""
    svc = _service(tmp_path, FakeModel().queue_text("ok"))
    cid = await svc.create_conversation()  # in-memory のみ
    await svc.send("bot", "x", conversation_id=cid)
    assert await svc.list_sessions() == []


@pytest.mark.asyncio
async def test_restore_continues_past_history(tmp_path: Path) -> None:
    """既存 session_id で会話を作り直すと過去履歴を踏まえて継続する（復元）。"""
    # 1 つ目のサービス: 永続化 session に 1 ターン積む。
    svc1 = _service(tmp_path, FakeModel().queue_text("first"))
    cid1 = await svc1.create_conversation(session_id="resume-me")
    await svc1.send("bot", "turn-1", conversation_id=cid1)

    # 2 つ目のサービス（別プロセス相当）: 同じ session_id を一覧から復元して継続。
    model2 = FakeModel().queue_text("second")
    svc2 = _service(tmp_path, model2)
    assert "resume-me" in [info.session_id for info in await svc2.list_sessions()]
    cid2 = await svc2.create_conversation(session_id="resume-me")
    await svc2.send("bot", "turn-2", conversation_id=cid2)

    # 復元後 send の get_response input に前回履歴（turn-1）が含まれる（継続の証跡）。
    restored_input = model2.calls[0].input
    assert isinstance(restored_input, list)
    assert len(restored_input) > 1


# ----------------------------------------------------------------------
# (5) NFR-7 トリップワイヤ
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sqlite_session_schema_has_agent_sessions_table(tmp_path: Path) -> None:
    """session 一覧/復元（D5）が SDK SQLiteSession の既定スキーマに依存することを担保する。

    `list_session_ids` は SDK が書く db の sessions テーブル名 `agent_sessions` と
    `session_id` 列を素 SELECT する前提に結合している。将来 SDK が既定テーブル名 / 列名を
    変えると過去 session の一覧/復元が静かに壊れるため、前提（テーブル名 `agent_sessions`・
    `session_id` 列の存在）の変化を早期検知するトリップワイヤ（NFR-7・バージョン耐性）。
    """
    db = str(tmp_path / "schema.db")
    await SQLiteSession("probe", db).add_items([{"role": "user", "content": "x"}])

    con = sqlite3.connect(db)
    try:
        tables = {
            row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert "agent_sessions" in tables
        columns = {row[1] for row in con.execute("PRAGMA table_info(agent_sessions)")}
        assert "session_id" in columns
    finally:
        con.close()
