"""L1: `${var}` プレースホルダの抽出 / 置換ヘルパ（外部 SDK 非依存）を検証する。

`PLACEHOLDER_RE`（braced のみマッチ）・`extract_placeholders`（識別子集合）・
`substitute_braced`（braced のみ置換・bare `$var` 不変・非 str 値 warn）を網羅する。
すべて純 regex / str 操作で外部依存なし（`@pytest.mark.unit`）。
"""

from __future__ import annotations

import logging

import pytest

from oai_agentspec.runtime.lightning._placeholders import (
    PLACEHOLDER_RE,
    extract_placeholders,
    substitute_braced,
)

pytestmark = pytest.mark.unit


# ----------------------------------------------------------------------
# PLACEHOLDER_RE（braced のみマッチ・bare $var 不変）
# ----------------------------------------------------------------------


def test_placeholder_re_matches_braced_only() -> None:
    """`${var}` のみがマッチし、bare `$var` / `${1abc}` 等の不正名は match しない。"""
    assert PLACEHOLDER_RE.findall("${name} ${role} hi") == ["name", "role"]
    # bare `$name` / `$5` / `$PATH` は触らない（safe_substitute との差分）。
    assert PLACEHOLDER_RE.findall("$name $5 $PATH") == []
    # 識別子は Python 識別子相当。先頭数字は不正名で match しない。
    assert PLACEHOLDER_RE.findall("${1abc} ${_ok} ${a1}") == ["_ok", "a1"]


# ----------------------------------------------------------------------
# extract_placeholders
# ----------------------------------------------------------------------


def test_extract_placeholders_returns_set_of_identifiers() -> None:
    """重複は集約され set として返る（順序非依存・空テキストは空集合）。"""
    assert extract_placeholders("${a} ${b} ${a}") == {"a", "b"}
    assert extract_placeholders("") == set()
    assert extract_placeholders("no placeholders here") == set()


# ----------------------------------------------------------------------
# substitute_braced
# ----------------------------------------------------------------------


def test_substitute_braced_replaces_only_braced() -> None:
    """braced `${name}` のみ置換し、bare `$var` は触らない。"""
    text = "Hello ${name}, balance: $5 (env: $PATH)"
    result = substitute_braced(text, {"name": "AgentSpec", "5": "FIVE", "PATH": "X"})
    # `${name}` だけが置換され、bare `$5` / `$PATH` は不変。
    assert result == "Hello AgentSpec, balance: $5 (env: $PATH)"


def test_substitute_braced_unknown_keys_kept() -> None:
    """`vars_dict` に未指定の placeholder は `${name}` のまま保持される（safe_substitute 互換）。"""
    assert substitute_braced("${a} ${b}", {"a": "X"}) == "X ${b}"


def test_substitute_braced_no_op_for_empty_inputs() -> None:
    """`vars_dict` が None / 空、`text` が空のときは text をそのまま返す。"""
    assert substitute_braced("hi", None) == "hi"
    assert substitute_braced("hi", {}) == "hi"
    assert substitute_braced("", {"a": "X"}) == ""


def test_substitute_braced_warns_on_non_str_value(caplog: pytest.LogCaptureFixture) -> None:
    """非 str 値は str(value) で変換するが、利用者へ warning ログで通知する。"""
    caplog.set_level(logging.WARNING, logger="oai_agentspec.runtime.lightning._placeholders")
    result = substitute_braced("count=${n}", {"n": 42})
    assert result == "count=42"
    assert any("vars['n']=42" in rec.getMessage() for rec in caplog.records)


# ----------------------------------------------------------------------
# split_marked（RED: Issue #40 T2・境界マーカーによる複数 tune 候補分割）
#
# `split_marked` / `compose_segments` は本テスト作成時点で未実装のため、モジュール
# トップレベルで import すると collection 自体が ImportError で落ち、既存テストまで
# 巻き込んで NG になる。既存テストを緑のまま保つため、各テスト関数内で遅延 import する。
# ----------------------------------------------------------------------


def test_split_marked_single_tune_without_marker_returns_candidate_as_is() -> None:
    """`n_tune=1` はマーカー不要で、候補テキストをそのまま単一要素のリストで返す。"""
    from oai_agentspec.runtime.lightning._placeholders import split_marked

    assert split_marked("hello world", 1) == ["hello world"]


def test_split_marked_single_tune_with_marker_returns_none() -> None:
    """`n_tune=1` なのにマーカーが混入している候補は None（不正候補・reward 0.0 経路）。"""
    from oai_agentspec.runtime.lightning._placeholders import split_marked

    assert split_marked("hello ${oas_boundary_1} world", 1) is None


def test_split_marked_two_tune_with_single_marker_splits_correctly() -> None:
    """`n_tune=2` でマーカーが 1 個ちょうど出現していれば構成順の 2 要素に分割する。"""
    from oai_agentspec.runtime.lightning._placeholders import split_marked

    assert split_marked("part1${oas_boundary_1}part2", 2) == ["part1", "part2"]


def test_split_marked_two_tune_missing_marker_returns_none() -> None:
    """`n_tune=2` でマーカーが欠落していれば None を返す。"""
    from oai_agentspec.runtime.lightning._placeholders import split_marked

    assert split_marked("part1 part2", 2) is None


def test_split_marked_two_tune_duplicated_marker_returns_none() -> None:
    """`n_tune=2` でマーカーが 2 回以上出現（重複）していれば None を返す。"""
    from oai_agentspec.runtime.lightning._placeholders import split_marked

    candidate = "part1${oas_boundary_1}part2${oas_boundary_1}part3"
    assert split_marked(candidate, 2) is None


def test_split_marked_three_tune_with_two_markers_splits_correctly() -> None:
    """`n_tune=3` でマーカーが 2 個ちょうど正常出現していれば構成順の 3 要素に分割する。"""
    from oai_agentspec.runtime.lightning._placeholders import split_marked

    candidate = "a${oas_boundary_1}b${oas_boundary_2}c"
    assert split_marked(candidate, 3) == ["a", "b", "c"]


def test_split_marked_three_tune_with_missing_second_marker_returns_none() -> None:
    """`n_tune=3` で `${oas_boundary_2}` が欠落していれば None を返す。"""
    from oai_agentspec.runtime.lightning._placeholders import split_marked

    candidate = "a${oas_boundary_1}b"
    assert split_marked(candidate, 3) is None


def test_split_marked_zero_tune_returns_none() -> None:
    """`n_tune=0` は呼び出し側の誤用に対する防御的 None を返す。"""
    from oai_agentspec.runtime.lightning._placeholders import split_marked

    assert split_marked("anything", 0) is None


# ----------------------------------------------------------------------
# compose_segments（RED: Issue #40 T2・segments 構成順 + tune_texts の再インターリーブ合成）
# ----------------------------------------------------------------------


def test_compose_segments_all_fixed_concatenates_with_vars_substituted() -> None:
    """全セグメントが `tune=False` なら、vars 注入済みテキストを `\\n\\n` で連結する。"""
    from oai_agentspec.runtime.lightning._placeholders import compose_segments
    from oai_agentspec.runtime.lightning.types import SlotSegment

    segments = (
        SlotSegment(ref="base:main", text="hello ${name}", tune=False),
        SlotSegment(ref="part:style", text="style ${x}", tune=False),
    )
    result = compose_segments(segments, [], {"name": "AgentSpec", "x": "Y"})
    assert result == "hello AgentSpec\n\nstyle Y"


def test_compose_segments_interleaves_tune_and_fixed_in_construction_order() -> None:
    """`tune=True` / `tune=False` 混在時、`tune_texts` は構成順の位置へ正しく再挿入される。"""
    from oai_agentspec.runtime.lightning._placeholders import compose_segments
    from oai_agentspec.runtime.lightning.types import SlotSegment

    segments = (
        SlotSegment(ref="base:main", text="fixed1", tune=False),
        SlotSegment(ref="agent:triage", text="seed_tune", tune=True),
        SlotSegment(ref="part:style", text="fixed2", tune=False),
    )
    result = compose_segments(segments, ["TUNED_TEXT"], {})
    assert result == "fixed1\n\nTUNED_TEXT\n\nfixed2"


def test_compose_segments_fixed_segment_substitutes_known_var() -> None:
    """fixed セグメントの `${var}` は `vars_dict` に対応キーがあれば値注入される。"""
    from oai_agentspec.runtime.lightning._placeholders import compose_segments
    from oai_agentspec.runtime.lightning.types import SlotSegment

    segments = (SlotSegment(ref="base:main", text="Hello ${name}", tune=False),)
    result = compose_segments(segments, [], {"name": "World"})
    assert result == "Hello World"


def test_compose_segments_fixed_segment_keeps_unknown_var() -> None:
    """fixed セグメントの `${var}` に対応する vars_dict キーが無ければ `${var}` を保持する。"""
    from oai_agentspec.runtime.lightning._placeholders import compose_segments
    from oai_agentspec.runtime.lightning.types import SlotSegment

    segments = (SlotSegment(ref="base:main", text="Hello ${name}", tune=False),)
    result = compose_segments(segments, [], {})
    assert result == "Hello ${name}"


def test_compose_segments_tune_segment_keeps_var_even_if_vars_dict_has_key() -> None:
    """tune セグメントは vars_dict に対応キーがあっても `${var}` を温存する。"""
    from oai_agentspec.runtime.lightning._placeholders import compose_segments
    from oai_agentspec.runtime.lightning.types import SlotSegment

    segments = (SlotSegment(ref="agent:triage", text="seed", tune=True),)
    result = compose_segments(segments, ["Keep ${var} raw"], {"var": "MUST_NOT_APPEAR"})
    assert result == "Keep ${var} raw"


def test_compose_segments_tune_texts_length_mismatch_raises_value_error() -> None:
    """`tune_texts` の長さと `tune=True` の要素数が不一致なら `ValueError`（実装者ミス検出）。"""
    from oai_agentspec.runtime.lightning._placeholders import compose_segments
    from oai_agentspec.runtime.lightning.types import SlotSegment

    segments = (SlotSegment(ref="agent:triage", text="seed", tune=True),)
    with pytest.raises(ValueError):
        compose_segments(segments, [], {})
