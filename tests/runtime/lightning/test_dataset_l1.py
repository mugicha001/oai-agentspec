"""L1: APO データセット型 `OptimizeCase` + データ分割ヘルパ `train_val_split` を検証する。

`OptimizeCase` は `input` のみ必須・他は安全な既定値（None / 空 list / 空 dict）・frozen で属性
再代入不可・既定 list / dict はインスタンスごと独立。`train_val_split` は決定的シャッフル（seed
固定で同結果・別 seed で別結果）・`shuffle=False` で入力順保持・val 件数 = round(n * ratio)・
入力不変（新リスト）・val_ratio 範囲外 ValueError・n_val=0 の空 val を網羅する。純データ操作で
外部依存なし（`@pytest.mark.unit`）。
"""

from __future__ import annotations

import pytest

from oai_agentspec.runtime.lightning import OptimizeCase, train_val_split

pytestmark = pytest.mark.unit


# ----------------------------------------------------------------------
# OptimizeCase（typed なケース型・llmops EvalCase 相当）
# ----------------------------------------------------------------------


def test_optimize_case_minimum_only_input() -> None:
    """OptimizeCase は input のみ必須で、他は安全な既定値（None / 空 list / 空 dict）を持つ。"""
    case = OptimizeCase(input="ユーザー依頼")
    assert case.input == "ユーザー依頼"
    assert case.id is None
    assert case.expected_output is None
    assert case.expected_tools == []
    assert case.expected_route == []
    assert case.expected_last_agent is None
    assert case.expected_approvals == []
    assert case.metadata == {}


def test_optimize_case_full_fields() -> None:
    """OptimizeCase は全期待フィールドと metadata を保持する（reward が field 既定で参照する）。"""
    case = OptimizeCase(
        input="返金して",
        id="case-1",
        expected_output="返金",
        expected_tools=["lookup_invoice"],
        expected_route=["triage", "billing"],
        expected_last_agent="billing",
        expected_approvals=["refund"],
        metadata={"tag": "billing"},
    )
    assert case.id == "case-1"
    assert case.expected_output == "返金"
    assert case.expected_tools == ["lookup_invoice"]
    assert case.expected_route == ["triage", "billing"]
    assert case.expected_last_agent == "billing"
    assert case.expected_approvals == ["refund"]
    assert case.metadata == {"tag": "billing"}


def test_optimize_case_is_frozen() -> None:
    """OptimizeCase は frozen dataclass で属性再代入できない（reward が破壊的変更を受けない）。"""
    case = OptimizeCase(input="x")
    with pytest.raises((AttributeError, TypeError)):
        case.input = "y"  # type: ignore[misc]


def test_optimize_case_independent_default_lists() -> None:
    """既定 list / dict は default_factory でインスタンスごとに独立（同一参照を共有しない）。"""
    a = OptimizeCase(input="a")
    b = OptimizeCase(input="b")
    assert a.expected_tools is not b.expected_tools
    assert a.metadata is not b.metadata


# ----------------------------------------------------------------------
# train_val_split（決定的データ分割）
# ----------------------------------------------------------------------


def test_split_is_deterministic_for_same_seed() -> None:
    """同 seed なら何度分割しても同じ (train, val) を返す（決定的）。"""
    data = list(range(10))
    a = train_val_split(data, val_ratio=0.2, seed=7)
    b = train_val_split(data, val_ratio=0.2, seed=7)
    assert a == b


def test_split_differs_for_different_seed() -> None:
    """別 seed なら（多くの場合）異なる分割になる（シャッフルが seed 依存）。"""
    data = list(range(20))
    a = train_val_split(data, val_ratio=0.3, seed=1)
    b = train_val_split(data, val_ratio=0.3, seed=2)
    assert a != b


def test_split_no_shuffle_preserves_input_order() -> None:
    """shuffle=False は入力順を保ったまま末尾を train・先頭を val に切る。"""
    data = list(range(10))
    train, val = train_val_split(data, val_ratio=0.2, shuffle=False)
    # n_val = round(10 * 0.2) = 2 → val は先頭 2 件、train はそれ以降。
    assert val == [0, 1]
    assert train == [2, 3, 4, 5, 6, 7, 8, 9]


def test_split_val_count_is_rounded() -> None:
    """val 件数は round(len * val_ratio) で算出する。"""
    data = list(range(10))
    _train, val = train_val_split(data, val_ratio=0.25, shuffle=False)
    # round(10 * 0.25) = 2。
    assert len(val) == 2


def test_split_does_not_mutate_input() -> None:
    """入力シーケンスを改変せず新リストを返す（純データ操作）。"""
    data = [3, 1, 2, 5, 4]
    snapshot = list(data)
    train, val = train_val_split(data, val_ratio=0.4, seed=0)
    assert data == snapshot
    # 新リストであり元 list と identity が異なる。
    assert train is not data
    assert val is not data


def test_split_zero_val_returns_empty_val() -> None:
    """n_val=0（val_ratio=0.0）のとき val は空・train は全件を返す。"""
    data = list(range(5))
    train, val = train_val_split(data, val_ratio=0.0)
    assert val == []
    assert sorted(train) == [0, 1, 2, 3, 4]


def test_split_full_val_ratio() -> None:
    """val_ratio=1.0 は全件 val・train 空（境界・shuffle=False で順序保持）。"""
    data = list(range(4))
    train, val = train_val_split(data, val_ratio=1.0, shuffle=False)
    assert train == []
    assert val == [0, 1, 2, 3]


def test_split_invalid_ratio_below_zero_raises() -> None:
    """val_ratio < 0.0 は ValueError。"""
    with pytest.raises(ValueError, match="val_ratio"):
        train_val_split([1, 2, 3], val_ratio=-0.1)


def test_split_invalid_ratio_above_one_raises() -> None:
    """val_ratio > 1.0 は ValueError。"""
    with pytest.raises(ValueError, match="val_ratio"):
        train_val_split([1, 2, 3], val_ratio=1.5)


def test_split_keeps_all_items() -> None:
    """train + val で全要素を過不足なく含む（要素の欠落 / 重複なし）。"""
    data = list(range(15))
    train, val = train_val_split(data, val_ratio=0.3, seed=3)
    assert sorted(train + val) == data
