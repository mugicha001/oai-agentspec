"""L1: `_adapters.run_context.unwrap_run_context` の展開契約の pin。

approvals / runner / serialization / intent の 6 呼び出し箇所が依存する共有ヘルパの
単一契約（`RunContextWrapper` は `.context` を開く・それ以外は素通し）を直接固定する。
"""

from __future__ import annotations

import pytest
from agents import RunContextWrapper

from oai_agentspec._adapters.run_context import unwrap_run_context

pytestmark = pytest.mark.unit


def test_run_context_wrapper_is_unwrapped() -> None:
    """`RunContextWrapper` を渡すと `.context` の生オブジェクトが返る（同一性まで pin）。"""
    raw = {"tenant": "acme"}
    wrapped = RunContextWrapper(context=raw)
    assert unwrap_run_context(wrapped) is raw


def test_plain_object_passes_through() -> None:
    """wrapper でない値はそのまま返る（同一性まで pin）。"""
    raw = object()
    assert unwrap_run_context(raw) is raw


def test_none_passes_through() -> None:
    """None は None のまま返る。"""
    assert unwrap_run_context(None) is None
