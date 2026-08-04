"""L1: 会話 store の内部型（`ConversationEntry`）を検証する（unit・外部依存なし）。

`ConversationEntry` の既定値保持と、bool フィールド `persist`（永続化要否）の構築時型検証を
pin する。SDK `Session` はダミーの不透明値で代用し、実 API は叩かない。
"""

from __future__ import annotations

import re

import pytest

from oai_agentspec.runtime.conversation.store import ConversationEntry

pytestmark = pytest.mark.unit


class _FakeSession:
    """実 API を叩かないダミー SDK Session（不透明値として保持されるだけ）。"""


def _entry_kwargs(**overrides: object) -> dict[str, object]:
    """ConversationEntry の最小構築 kwargs（bool 検証テストの共通引数）。"""
    base: dict[str, object] = {
        "conversation_id": "c1",
        "session_id": "s1",
        "session": _FakeSession(),
    }
    base.update(overrides)
    return base


def test_conversation_entry_defaults() -> None:
    """既定は agent_name None・turn_count 0・persist False・承認待ち空。"""
    entry = ConversationEntry(**_entry_kwargs())  # type: ignore[arg-type]
    assert entry.agent_name is None
    assert entry.turn_count == 0
    assert entry.persist is False
    assert entry.pending_state is None
    assert entry.pending_approvals == []


# ----------------------------------------------------------------------
# persist の構築時 bool 型検証
# ----------------------------------------------------------------------


def test_conversation_entry_persist_none_raises() -> None:
    """persist=None は bool でないため構築時 ValueError（メッセージ全文を pin）。

    永続化要否フラグが黙って falsy になると HITL 中断状態が保存されず、復元不能になる。
    """
    with pytest.raises(ValueError, match=re.escape("persist must be a bool, got 'NoneType'")):
        ConversationEntry(**_entry_kwargs(persist=None))  # type: ignore[arg-type]


def test_conversation_entry_persist_str_raises() -> None:
    """persist="no" は truthy な文字列だが ValueError で弾く（意図と逆の永続化を防ぐ）。"""
    with pytest.raises(ValueError, match=re.escape("persist must be a bool, got 'str'")):
        ConversationEntry(**_entry_kwargs(persist="no"))  # type: ignore[arg-type]


def test_conversation_entry_persist_int_zero_raises() -> None:
    """persist=0（int）は bool でないため ValueError（int の 0 / 1 も受理しない）。"""
    with pytest.raises(ValueError, match="persist"):
        ConversationEntry(**_entry_kwargs(persist=0))  # type: ignore[arg-type]


def test_conversation_entry_persist_bool_constructs() -> None:
    """persist へ True / False を渡した構築は成功する（正常系の維持）。"""
    assert ConversationEntry(**_entry_kwargs(persist=True)).persist is True  # type: ignore[arg-type]
    assert ConversationEntry(**_entry_kwargs(persist=False)).persist is False  # type: ignore[arg-type]
