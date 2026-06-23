"""ユニット: _adapters の HITL 関数（apply_approvals の検証先行 2 パス・session 専用テーブル）。

`apply_approvals` は不透明 `RunState` 相当（`get_interruptions()` と `_context.is_tool_approved`
を持つ fake）に対し、検証先行で承認/却下を適用する純ロジックとして検証する。session の専用
テーブル（oai_agentspec_pending_approvals）は save/load/delete/list のラウンドトリップを
sqlite tmp で
検証する。実 SDK・実 LLM は呼ばない。

`apply_approvals` は item 引き当てに `state.get_interruptions()` を使う（SDK `RunState` は
`get_interruptions()` を公開し `interruptions` 属性を持たない）。fake state も SDK に合わせ
`get_interruptions()` を実装する。
"""

from __future__ import annotations

from typing import Any

import pytest

from oai_agentspec._adapters import (
    apply_approvals,
    delete_pending_approval,
    list_pending_approvals,
    load_pending_approval,
    save_pending_approval,
)

pytestmark = pytest.mark.unit


class _FakeItem:
    """ToolApprovalItem 相当の fake（call_id / tool_name を持つ）。"""

    def __init__(self, call_id: str, tool_name: str) -> None:
        self.call_id = call_id
        self.tool_name = tool_name


class _FakeState:
    """RunState 相当の fake（get_interruptions / approve / reject / _context を持つ）。

    `_context.is_tool_approved(tool_name, call_id)` は approve/reject 済みなら True/False、
    未解決なら None を返す（SDK の解決状態問い合わせを模す）。
    """

    def __init__(self, items: list[_FakeItem]) -> None:
        self._items = items
        self.approved: list[str] = []
        self.rejected: list[tuple[str, str | None]] = []
        self._resolved: dict[str, bool] = {}
        self._context = self._Context(self._resolved)

    class _Context:
        def __init__(self, resolved: dict[str, bool]) -> None:
            self._resolved = resolved

        def is_tool_approved(self, tool_name: str, call_id: str) -> bool | None:
            return self._resolved.get(call_id)

    def get_interruptions(self) -> list[_FakeItem]:
        return self._items

    def approve(self, item: _FakeItem) -> None:
        self.approved.append(item.call_id)
        self._resolved[item.call_id] = True

    def reject(self, item: _FakeItem, *, rejection_message: str | None = None) -> None:
        self.rejected.append((item.call_id, rejection_message))
        self._resolved[item.call_id] = False


# ----------------------------------------------------------------------
# apply_approvals: 検証先行 2 パス / fail-closed
# ----------------------------------------------------------------------
def test_apply_approve_marks_applied() -> None:
    """approve 指定で state.approve が呼ばれ applied に call_id が入る。"""
    state = _FakeState([_FakeItem("c1", "danger")])
    result = apply_approvals(state, [{"call_id": "c1", "decision": "approve"}])
    assert result.applied == ["c1"]
    assert result.unknown == []
    assert state.approved == ["c1"]


def test_apply_reject_passes_rejection_message() -> None:
    """reject 指定で rejection_message が state.reject へ渡る。"""
    state = _FakeState([_FakeItem("c1", "danger")])
    result = apply_approvals(
        state,
        [{"call_id": "c1", "decision": "reject", "rejection_message": "no"}],
    )
    assert result.applied == ["c1"]
    assert state.rejected == [("c1", "no")]


def test_apply_fail_closed_unknown_decision_rejects() -> None:
    """decision 未知値は reject へ倒す（fail-closed・approve しない・NFR-7）。"""
    state = _FakeState([_FakeItem("c1", "danger")])
    apply_approvals(state, [{"call_id": "c1", "decision": "weird"}])
    assert state.approved == []
    assert [cid for cid, _ in state.rejected] == ["c1"]


def test_apply_unknown_call_id_is_reported_and_no_mutation() -> None:
    """未知 call_id は unknown に集まり state へ一切適用されない（検証パスで止まる）。"""
    state = _FakeState([_FakeItem("c1", "danger")])
    result = apply_approvals(state, [{"call_id": "ghost", "decision": "approve"}])
    assert result.unknown == ["ghost"]
    assert state.approved == []
    assert state.rejected == []


def test_apply_mixed_batch_with_unknown_applies_nothing() -> None:
    """混在バッチ（正常 + 未知）は正常分も適用せず state 不変（部分適用回避・FR-4）。"""
    state = _FakeState([_FakeItem("c1", "danger")])
    result = apply_approvals(
        state,
        [
            {"call_id": "c1", "decision": "approve"},
            {"call_id": "ghost", "decision": "approve"},
        ],
    )
    assert result.unknown == ["ghost"]
    assert result.applied == []
    # 検証先行 2 パス: 1 件でも無効なら state を一切変更しない。
    assert state.approved == []


def test_apply_already_resolved_is_reported() -> None:
    """解決済み call_id への再操作は already_resolved に集まり再適用しない。"""
    state = _FakeState([_FakeItem("c1", "danger")])
    # 1 度 approve 済みにする。
    apply_approvals(state, [{"call_id": "c1", "decision": "approve"}])
    result = apply_approvals(state, [{"call_id": "c1", "decision": "reject"}])
    assert result.already_resolved == ["c1"]
    # 再操作は反映されない（reject は記録されない）。
    assert state.rejected == []


# ----------------------------------------------------------------------
# session 専用テーブル（oai_agentspec_pending_approvals）CRUD ラウンドトリップ
# ----------------------------------------------------------------------
def test_save_then_load_roundtrip(tmp_path: Any) -> None:
    """save → load で run_state / pending JSON / agent_name が往復する（session_id キー）。"""
    db = str(tmp_path / "conversations.db")
    save_pending_approval(db, "sess-1", "bot", '{"state": 1}', '[{"call_id": "c1"}]')

    record = load_pending_approval(db, "sess-1")
    assert record is not None
    assert record["session_id"] == "sess-1"
    assert record["agent_name"] == "bot"
    assert record["run_state"] == '{"state": 1}'
    assert record["pending"] == '[{"call_id": "c1"}]'


def test_save_upsert_overwrites_same_session(tmp_path: Any) -> None:
    """同一 session_id の save は上書き（段階解決/再中断で更新される）。"""
    db = str(tmp_path / "conversations.db")
    save_pending_approval(db, "sess-1", "bot", '{"v": 1}', "[]")
    save_pending_approval(db, "sess-1", "bot", '{"v": 2}', "[]")

    record = load_pending_approval(db, "sess-1")
    assert record is not None
    assert record["run_state"] == '{"v": 2}'
    # 1 行のみ（重複行が増えない）。
    assert len(list_pending_approvals(db)) == 1


def test_delete_is_session_scoped(tmp_path: Any) -> None:
    """delete は session_id 条件付きで、別 session の中断状態を消さない（NFR-4(b)）。"""
    db = str(tmp_path / "conversations.db")
    save_pending_approval(db, "s1", "bot", '{"a": 1}', "[]")
    save_pending_approval(db, "s2", "bot", '{"b": 1}', "[]")

    delete_pending_approval(db, "s1")

    assert load_pending_approval(db, "s1") is None
    # s2 は残る。
    assert load_pending_approval(db, "s2") is not None


def test_save_recreates_old_schema_table(tmp_path: Any) -> None:
    """旧スキーマの承認待ちテーブルが残っていても save が作り直して成功する（自己修復）。

    HITL テーブルのスキーマ進化（conversation_id PK・agent_name 無し -> session_id PK・
    agent_name 有り）で残る旧テーブルに対し、新 save が DROP+CREATE して INSERT できること、
    会話履歴テーブル（agent_sessions / agent_messages）は維持されることを検証する。
    """
    import sqlite3

    db = str(tmp_path / "conversations.db")
    con = sqlite3.connect(db)
    # 旧スキーマの承認待ちテーブルと、触ってはいけない会話履歴テーブルを用意する。
    con.execute(
        "CREATE TABLE oai_agentspec_pending_approvals "
        "(conversation_id TEXT PRIMARY KEY, session_id TEXT, run_state TEXT, "
        "pending TEXT, updated_at TEXT)"
    )
    con.execute("CREATE TABLE agent_sessions (session_id TEXT PRIMARY KEY)")
    con.execute("INSERT INTO agent_sessions VALUES ('keep-me')")
    con.commit()
    con.close()

    # 旧スキーマでは agent_name 列が無いが、save は作り直して成功する。
    save_pending_approval(db, "sess-1", "ops", '{"v": 1}', "[]")
    record = load_pending_approval(db, "sess-1")
    assert record is not None
    assert record["agent_name"] == "ops"

    # 会話履歴テーブルは維持される（自前テーブルのみ作り直す）。
    con = sqlite3.connect(db)
    kept = [r[0] for r in con.execute("SELECT session_id FROM agent_sessions")]
    con.close()
    assert kept == ["keep-me"]


def test_load_missing_db_returns_none(tmp_path: Any) -> None:
    """db ファイルが無いとき load は None（復元対象なし）。"""
    db = str(tmp_path / "absent.db")
    assert load_pending_approval(db, "sess-x") is None


def test_load_missing_row_returns_none(tmp_path: Any) -> None:
    """テーブルはあるが該当行が無いとき load は None。"""
    db = str(tmp_path / "conversations.db")
    save_pending_approval(db, "s1", "bot", "{}", "[]")
    assert load_pending_approval(db, "sess-other") is None


def test_delete_missing_db_is_noop(tmp_path: Any) -> None:
    """db が無くても delete は例外にならない（冪等）。"""
    db = str(tmp_path / "absent.db")
    delete_pending_approval(db, "sess-x")  # 例外が出ないこと


def test_list_pending_approvals_empty_when_no_db(tmp_path: Any) -> None:
    """db / テーブルが無ければ list は空リスト。"""
    db = str(tmp_path / "absent.db")
    assert list_pending_approvals(db) == []


def test_list_pending_approvals_omits_run_state(tmp_path: Any) -> None:
    """list は run_state（重い JSON）を含めず session_id / agent_name / pending 等を返す。"""
    db = str(tmp_path / "conversations.db")
    save_pending_approval(db, "s1", "bot", '{"big": 1}', '[{"call_id": "c1"}]')
    rows = list_pending_approvals(db)
    assert len(rows) == 1
    assert "run_state" not in rows[0]
    assert rows[0]["session_id"] == "s1"
    assert rows[0]["agent_name"] == "bot"
    assert rows[0]["pending"] == '[{"call_id": "c1"}]'
