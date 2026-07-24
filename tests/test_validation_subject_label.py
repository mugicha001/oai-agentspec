"""`validate_extra_kwargs` の `subject_label` 引数拡張の検証（Issue #27 Task 2・RED 先行）。

`_adapters/tools.py` の `build_function_tool` が extra 検証を `validate_extra_kwargs`
に流用する（設計判断 6・案 A）ため、既存の `agent 'xxx': ...` メッセージ prefix を
差し替え可能にする必要がある。既定値 `subject_label="agent"` により既存 `build_agent` /
`build_realtime_agent` からの呼び出しはメッセージ・挙動ともに完全不変。

実装未完のため（`subject_label` 引数が未追加）、`subject_label="tool"` を渡す呼び出しは
`TypeError` になる（RED 状態が正しい）。
"""

from __future__ import annotations

import pytest

from oai_agentspec._validation import validate_extra_kwargs


# ---------------------------------------------------------------------------
# 既定値 "agent": 既存メッセージ完全不変（後方互換の担保）
# ---------------------------------------------------------------------------
def test_正常系_subject_label_既定値agentで既存メッセージ不変() -> None:
    """`subject_label` を省略した既存呼び出しの衝突メッセージ原文が変わっていない。"""
    with pytest.raises(ValueError) as excinfo:
        validate_extra_kwargs(
            "bot",
            {"name": 1},
            dedicated=frozenset({"name"}),
            field_names=frozenset({"name"}),
            agent_label="agents.Agent",
        )
    assert str(excinfo.value) == (
        "agent 'bot': extra に専用フィールドと同名のキーが含まれます: ['name']"
    )


# ---------------------------------------------------------------------------
# subject_label="tool": prefix 語が差し替わる
# ---------------------------------------------------------------------------
def test_正常系_subject_label_tool指定で衝突文言のprefixが差し替わる() -> None:
    """`subject_label="tool"` 指定で衝突メッセージ prefix が `tool 'xxx'` に差し替わる。"""
    with pytest.raises(ValueError) as excinfo:
        validate_extra_kwargs(
            "bot",
            {"name": 1},
            dedicated=frozenset({"name"}),
            field_names=frozenset({"name"}),
            agent_label="agents.Agent",
            subject_label="tool",
        )
    assert str(excinfo.value) == (
        "tool 'bot': extra に専用フィールドと同名のキーが含まれます: ['name']"
    )


def test_正常系_subject_label_未知キーメッセージも同様に差し替わる() -> None:
    """未知キーメッセージ側も同じ prefix 差し替え（`tool 'xxx'`）が効く。"""
    with pytest.raises(ValueError) as excinfo:
        validate_extra_kwargs(
            "bot",
            {"bogus": 1},
            dedicated=frozenset({"name"}),
            field_names=frozenset({"name"}),
            agent_label="agents.function_tool",
            subject_label="tool",
        )
    assert str(excinfo.value) == (
        "tool 'bot': extra に agents.function_tool が受け付けないキーが含まれます: ['bogus']"
    )
