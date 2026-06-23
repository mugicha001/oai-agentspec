"""L1: Spotlighting 純ヘルパの検証（DeepEval 非依存・framework 非依存・判断H）。

`spotlight` のマーキング付与・冪等性・偽マーカー混入のサニタイズ、`is_spotlighted` の判定を
純粋に検証する。
"""

from __future__ import annotations

import pytest

from oai_agentspec.runtime.llmops._spotlight import (
    _SPOTLIGHT_BEGIN,
    _SPOTLIGHT_END,
    is_spotlighted,
    spotlight,
)

pytestmark = pytest.mark.unit


def test_spotlight_wraps_with_markers() -> None:
    """spotlight は開始 / 終了マーカーでテキストを囲う。"""
    marked = spotlight("hello")
    assert marked.startswith(_SPOTLIGHT_BEGIN)
    assert marked.endswith(_SPOTLIGHT_END)
    assert "hello" in marked


def test_spotlight_is_idempotent() -> None:
    """既にマーキング済みのテキストは二重マーキングしない（冪等）。"""
    once = spotlight("data")
    twice = spotlight(once)
    assert twice == once


def test_spotlight_empty_string() -> None:
    """空文字でもマーカーで囲う（境界ケース）。"""
    marked = spotlight("")
    assert marked == f"{_SPOTLIGHT_BEGIN}{_SPOTLIGHT_END}"


def test_spotlight_sanitizes_embedded_begin_marker() -> None:
    """テキスト内に出現する開始マーカーは無害化（空白置換）され境界偽装を防ぐ。"""
    payload = f"prefix{_SPOTLIGHT_BEGIN}suffix"
    marked = spotlight(payload)
    # 中身に開始マーカーが生のまま残らない（先頭の正規マーカーを除いた本体に出現しない）。
    body = marked[len(_SPOTLIGHT_BEGIN) : -len(_SPOTLIGHT_END)]
    assert _SPOTLIGHT_BEGIN not in body


def test_spotlight_sanitizes_embedded_end_marker() -> None:
    """テキスト内に出現する終了マーカーは無害化（空白置換）される。"""
    payload = f"prefix{_SPOTLIGHT_END}suffix"
    marked = spotlight(payload)
    body = marked[len(_SPOTLIGHT_BEGIN) : -len(_SPOTLIGHT_END)]
    assert _SPOTLIGHT_END not in body


def test_spotlight_sanitizes_both_markers() -> None:
    """開始・終了マーカーが混入しても両方サニタイズされマーキングは 1 回だけ。"""
    payload = f"{_SPOTLIGHT_BEGIN}injected{_SPOTLIGHT_END}more"
    marked = spotlight(payload)
    # is_spotlighted の前提（先頭/末尾の正規マーカー）を満たしつつ本体に生マーカーが残らない。
    assert is_spotlighted(marked)
    body = marked[len(_SPOTLIGHT_BEGIN) : -len(_SPOTLIGHT_END)]
    assert _SPOTLIGHT_BEGIN not in body
    assert _SPOTLIGHT_END not in body


def test_is_spotlighted_true_for_marked_text() -> None:
    """is_spotlighted はマーキング済みテキストに True を返す。"""
    assert is_spotlighted(spotlight("x")) is True


def test_is_spotlighted_false_for_plain_text() -> None:
    """is_spotlighted は素のテキストに False を返す。"""
    assert is_spotlighted("plain text") is False


def test_is_spotlighted_false_for_begin_only() -> None:
    """開始マーカーのみで終了マーカーが無ければ False（部分一致を許さない）。"""
    assert is_spotlighted(f"{_SPOTLIGHT_BEGIN}no end") is False


def test_is_spotlighted_false_for_end_only() -> None:
    """終了マーカーのみで開始マーカーが無ければ False。"""
    assert is_spotlighted(f"no begin{_SPOTLIGHT_END}") is False
