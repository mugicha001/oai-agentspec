"""L1: 評価の観測 plain 型（`ObservedRun`）の純検証（外部 SDK 非依存）。

`ObservedRun` の既定値保持と、bool フィールド `interrupted`（中断の有無）の構築時型検証を
pin する。すべて純データ操作で外部依存なし（`@pytest.mark.unit`）。
"""

from __future__ import annotations

import re

import pytest

from oai_agentspec.runtime.llmops import ObservedRoute, ObservedRun

pytestmark = pytest.mark.unit


def test_observed_run_defaults() -> None:
    """既定は tool_calls / pending_approvals 空・interrupted False。"""
    run = ObservedRun(route=ObservedRoute(steps=[], last_agent="bot"))
    assert run.tool_calls == []
    assert run.pending_approvals == []
    assert run.interrupted is False


# ----------------------------------------------------------------------
# interrupted の構築時 bool 型検証
# ----------------------------------------------------------------------


def test_observed_run_interrupted_none_raises() -> None:
    """interrupted=None は bool でないため構築時 ValueError（メッセージ全文を pin）。

    中断フラグが黙って falsy になると「未解決の承認待ちを完了採点扱いする」誤診断になるため、
    構築時に fail-fast する。
    """
    with pytest.raises(ValueError, match=re.escape("interrupted must be a bool, got 'NoneType'")):
        ObservedRun(
            route=ObservedRoute(steps=[], last_agent="bot"),
            tool_calls=[],
            interrupted=None,  # type: ignore[arg-type]
        )


def test_observed_run_interrupted_str_raises() -> None:
    """interrupted="no" は truthy な文字列だが ValueError で弾く（silent 受理しない）。"""
    with pytest.raises(ValueError, match=re.escape("interrupted must be a bool, got 'str'")):
        ObservedRun(
            route=ObservedRoute(steps=[], last_agent="bot"),
            tool_calls=[],
            interrupted="no",  # type: ignore[arg-type]
        )


def test_observed_run_interrupted_int_zero_raises() -> None:
    """interrupted=0（int）は bool でないため ValueError（int の 0 / 1 も受理しない）。"""
    with pytest.raises(ValueError, match="interrupted"):
        ObservedRun(
            route=ObservedRoute(steps=[], last_agent="bot"),
            tool_calls=[],
            interrupted=0,  # type: ignore[arg-type]
        )


def test_observed_run_interrupted_bool_constructs() -> None:
    """interrupted へ True / False を渡した構築は成功する（正常系の維持）。"""
    route = ObservedRoute(steps=[], last_agent="bot")
    assert ObservedRun(route=route, interrupted=True).interrupted is True
    assert ObservedRun(route=route, interrupted=False).interrupted is False
