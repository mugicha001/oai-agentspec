"""L2: _adapters.builders の標準ルート build_agent extra 検証の特性化テスト。

Issue #19（純リファクタ）の安全網として、標準ルート `build_agent` の extra 検証
（専用フィールド同名キー衝突 / agents.Agent 未知キー）の ValueError メッセージ原文を
`_adapters` レベルで直接ピン留めする。Realtime 側は `test_realtime_l2.py` の extra reject
テストで担保済みだが、標準ルートには `_adapters` レベルでメッセージ原文を固定するテストが
不在（既存 `tests/runtime/guardrails/test_factories_l2.py` は `match="input_guardrails"` の
部分一致のみ）。

本モジュールは現状で GREEN になる特性化テスト（characterization test）であり、将来の
リファクタでメッセージ文字列が変わったら失敗するよう原文を完全一致でピン留めする。
"""

from __future__ import annotations

import pytest
from agents import Agent

from oai_agentspec._adapters import build_agent
from oai_agentspec.spec import AgentSpec

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# build_agent: extra に専用フィールド同名キー → ValueError（衝突メッセージ原文）
# ---------------------------------------------------------------------------
def test_build_rejects_dedicated_field_collision_message() -> None:
    """extra に専用フィールド同名キー（name）を積むと衝突メッセージ原文で弾く。"""
    spec = AgentSpec(name="bot", instructions="i", extra={"name": "dup"})
    with pytest.raises(ValueError) as excinfo:
        build_agent(spec)
    message = str(excinfo.value)
    # メッセージ原文を完全一致でピン留め（agent 名 + 「専用フィールドと同名」+ キー一覧）。
    assert message == ("agent 'bot': extra に専用フィールドと同名のキーが含まれます: ['name']")


def test_build_collision_message_lists_keys_sorted() -> None:
    """複数の専用フィールド同名キーはソート済みリストで列挙される。"""
    spec = AgentSpec(name="bot", instructions="i", extra={"name": "dup", "model": object()})
    with pytest.raises(ValueError) as excinfo:
        build_agent(spec)
    # sorted() でキーが昇順（model, name）に整列することを含めてピン留めする。
    assert str(excinfo.value) == (
        "agent 'bot': extra に専用フィールドと同名のキーが含まれます: ['model', 'name']"
    )


def test_build_collision_takes_precedence_over_unknown() -> None:
    """衝突キーと未知キーが同時にある場合は衝突メッセージが優先される（検査順の固定）。"""
    spec = AgentSpec(name="bot", instructions="i", extra={"name": "dup", "bogus": 1})
    with pytest.raises(ValueError) as excinfo:
        build_agent(spec)
    # 衝突検査が未知検査より先に実行され、未知キー（bogus）はメッセージに現れない
    # （完全一致 assert が bogus の非出現も含めて固定する）。
    assert str(excinfo.value) == (
        "agent 'bot': extra に専用フィールドと同名のキーが含まれます: ['name']"
    )


# ---------------------------------------------------------------------------
# build_agent: extra に未知キー → ValueError（未知メッセージ原文）
# ---------------------------------------------------------------------------
def test_build_rejects_unknown_key_message() -> None:
    """extra に agents.Agent が受け付けない未知キーを積むと未知メッセージ原文で弾く。"""
    spec = AgentSpec(name="bot", instructions="i", extra={"nonexistent_kw": 1})
    with pytest.raises(ValueError) as excinfo:
        build_agent(spec)
    message = str(excinfo.value)
    # メッセージ原文を完全一致でピン留め（agent 名 + 「agents.Agent が受け付けない」+ キー一覧）。
    assert message == (
        "agent 'bot': extra に agents.Agent が受け付けないキーが含まれます: ['nonexistent_kw']"
    )


def test_build_unknown_message_lists_keys_sorted() -> None:
    """複数の未知キーはソート済みリストで列挙される。"""
    spec = AgentSpec(name="bot", instructions="i", extra={"zzz": 1, "aaa": 2})
    with pytest.raises(ValueError) as excinfo:
        build_agent(spec)
    # sorted() でキーが昇順（aaa, zzz）に整列することを含めてピン留めする。
    assert str(excinfo.value) == (
        "agent 'bot': extra に agents.Agent が受け付けないキーが含まれます: ['aaa', 'zzz']"
    )


# ---------------------------------------------------------------------------
# build_agent: 正常系（有効な素通し extra が反映される）
# ---------------------------------------------------------------------------
def test_build_passes_valid_extra_through() -> None:
    """agents.Agent が受け付ける有効な extra キーは構築された Agent へ素通しされる。"""
    spec = AgentSpec(name="bot", instructions="i", extra={"handoff_description": "desk"})
    agent = build_agent(spec)
    assert isinstance(agent, Agent)
    assert agent.handoff_description == "desk"
