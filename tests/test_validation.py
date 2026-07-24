"""共有バリデータ `validate_realtime_handoff_options` の直接ユニットテスト（RED 先行）。

`RealtimeAgentRegistry._validate_spec` にインラインで実装されている handoff 系検証
（handoff_options のキー整合・input_type→on_handoff 必須・on_handoff の引数個数）を
`oai_agentspec._validation` の共有関数へ抽出する設計（Issue #15 タスク1）に対する検証。

抽出関数 `validate_realtime_handoff_options(agent_name, handoffs, handoff_options)` は
未実装のため、本モジュールの import は ImportError（collection error = RED）になる想定。
検証意図は「既存 registry のインライン検証と同一挙動・同一エラーメッセージであること」を
一次情報として担保することにある（抽出後も挙動不変であるべき仕様の固定）。
"""

from __future__ import annotations

import pytest

from oai_agentspec._validation import (
    validate_instructions_callable,
    validate_realtime_handoff_options,
)
from oai_agentspec.realtime.spec import RealtimeHandoffConfig


# ------------------------------------------------------------------
# 正常系: キー整合・on_handoff arity が SDK 契約どおりなら例外を出さない
# ------------------------------------------------------------------
def test_正常系_handoff_options_なしは通過() -> None:
    """handoff_options が空なら handoffs の有無にかかわらず検証を通過する。"""
    validate_realtime_handoff_options("a", ["b"], {})
    validate_realtime_handoff_options("a", [], {})


def test_正常系_on_handoff_なしのエッジ設定は通過() -> None:
    """on_handoff / input_type を持たない per-edge 設定は検証を通過する。"""
    validate_realtime_handoff_options(
        "a", ["b"], {"b": RealtimeHandoffConfig(tool_name_override="go")}
    )


def test_正常系_input_type_なし_on_handoff_は1引数() -> None:
    """input_type 未指定時、on_handoff は (context) の 1 引数なら通過する。"""
    validate_realtime_handoff_options(
        "a", ["b"], {"b": RealtimeHandoffConfig(on_handoff=lambda c: None)}
    )


def test_正常系_input_type_あり_on_handoff_は2引数() -> None:
    """input_type 指定時、on_handoff は (context, input) の 2 引数なら通過する。"""
    validate_realtime_handoff_options(
        "a",
        ["b"],
        {"b": RealtimeHandoffConfig(input_type=object, on_handoff=lambda c, i: None)},
    )


# ------------------------------------------------------------------
# 異常系: キー不整合
# ------------------------------------------------------------------
def test_異常系_handoffs_に無いキーは_ValueError() -> None:
    """handoffs に存在しない handoff_options キーは agent 名・キー名入り ValueError。

    タイポによる per-edge 設定の silent drop を防ぐ既存挙動を抽出関数でも維持する。
    """
    with pytest.raises(ValueError, match=r"a.*suport.*handoffs"):
        validate_realtime_handoff_options("a", ["support"], {"suport": RealtimeHandoffConfig()})


# ------------------------------------------------------------------
# 異常系: input_type 指定に on_handoff が伴わない
# ------------------------------------------------------------------
def test_異常系_input_type_だけで_on_handoff_なしは_ValueError() -> None:
    """input_type 指定時に on_handoff を欠く設定は agent 名・エッジ名入り ValueError。"""
    with pytest.raises(ValueError, match=r"'a' -> 'b'.*on_handoff"):
        validate_realtime_handoff_options(
            "a", ["b"], {"b": RealtimeHandoffConfig(input_type=object)}
        )


# ------------------------------------------------------------------
# 異常系: on_handoff の引数個数不一致
# ------------------------------------------------------------------
def test_異常系_1引数期待に2引数の_on_handoff_は_ValueError() -> None:
    """input_type なし（1 引数期待）に 2 引数 on_handoff を渡すと ValueError。"""
    with pytest.raises(ValueError, match=r"'a' -> 'b'.*1 引数.*2 引数"):
        validate_realtime_handoff_options(
            "a", ["b"], {"b": RealtimeHandoffConfig(on_handoff=lambda c, i: None)}
        )


def test_異常系_2引数期待に1引数の_on_handoff_は_ValueError() -> None:
    """input_type あり（2 引数期待）に 1 引数 on_handoff を渡すと ValueError。"""
    with pytest.raises(ValueError, match=r"'c' -> 'd'.*2 引数.*1 引数"):
        validate_realtime_handoff_options(
            "c",
            ["d"],
            {"d": RealtimeHandoffConfig(input_type=object, on_handoff=lambda c: None)},
        )


# ------------------------------------------------------------------
# 境界: シグネチャ取得不能な callable はスキップ
# ------------------------------------------------------------------
def test_境界_シグネチャ取得不能な_on_handoff_はスキップ() -> None:
    """inspect.signature が取れない callable（builtin 等）は arity 検査をスキップし通過する。"""
    validate_realtime_handoff_options("a", ["b"], {"b": RealtimeHandoffConfig(on_handoff=zip)})


# ------------------------------------------------------------------
# validate_instructions_callable: フィールドラベル引数（Issue #21 T2・RED 先行）
# ------------------------------------------------------------------
def test_既定ラベルのメッセージは_instructions_のまま不変() -> None:
    """フィールドラベル未指定時のエラーメッセージ原文は従来どおり instructions を含む。"""
    with pytest.raises(ValueError) as excinfo:
        validate_instructions_callable("a", lambda x: x)
    assert str(excinfo.value) == (
        "agent 'a': instructions callable は (context, agent) の 2 引数で呼び出せる必要があります"
    )


def test_フィールドラベル指定でメッセージが_base_instructions_になる() -> None:
    """field_label='base_instructions' 指定時はメッセージに base_instructions が入る。"""
    with pytest.raises(ValueError) as excinfo:
        validate_instructions_callable("a", lambda x: x, field_label="base_instructions")
    message = str(excinfo.value)
    assert "'a'" in message
    assert "base_instructions" in message


def test_フィールドラベル指定でも2引数_callable_は通過() -> None:
    """field_label を指定しても (context, agent) の 2 引数 callable は検証を通過する。"""
    validate_instructions_callable("a", lambda c, a: "x", field_label="base_instructions")
