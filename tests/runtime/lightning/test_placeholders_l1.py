"""L1: `${var}` プレースホルダの抽出 / 置換 / 合成ヘルパ（外部 SDK 非依存）を検証する。

`PLACEHOLDER_RE`（braced のみマッチ）・`extract_placeholders`（識別子集合）・
`substitute_braced`（braced のみ置換・bare `$var` 不変・非 str 値 warn）・
`compose_with_vars`（fixed 側 vars 再注入・空 fixed の素通し・空文字 fixed の脱落防止）を網羅する。
すべて純 regex / str 操作で外部依存なし（`@pytest.mark.unit`）。
"""

from __future__ import annotations

import logging

import pytest

from oai_agentspec.runtime.lightning._placeholders import (
    PLACEHOLDER_RE,
    compose_with_vars,
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
# compose_with_vars
# ----------------------------------------------------------------------


def test_compose_with_vars_empty_fixed_returns_tune() -> None:
    """`fixed` が空文字なら `tune` をそのまま返す。"""
    assert compose_with_vars("", "tune body", {"a": "X"}) == "tune body"


def test_compose_with_vars_concatenates_fixed_and_tune() -> None:
    """`fixed` 非空なら `fixed_substituted + "\\n\\n" + tune` を返す。"""
    result = compose_with_vars("base ${role}", "tune body", {"role": "engineer"})
    assert result == "base engineer\n\ntune body"


def test_compose_with_vars_keeps_separator_when_fixed_substitutes_to_empty() -> None:
    """`fixed` が `${var}` のみで vars 値が空文字でも、`fixed` 非空なら "\\n\\n" 区切りを保つ。

    `if fixed_substituted else tune` と書くと空文字結果を素通してしまい、rollout 実体
    （`_default_build` の合成）と差が出る。空判定は **substitution 前** の `fixed` で行う。
    """
    result = compose_with_vars("${role}", "tune body", {"role": ""})
    assert result == "\n\ntune body"


def test_compose_with_vars_does_not_substitute_tune_side() -> None:
    """tune 側は `${var}` 温存契約のため substitute しない。

    rollout 直前の `_reinject_vars` で別途注入される（`compose_with_vars` の責務は fixed 側のみ）。
    """
    result = compose_with_vars("base", "${tune_var}", {"tune_var": "MUST_NOT_APPLY"})
    assert result == "base\n\n${tune_var}"
