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

import re

import pytest

from oai_agentspec import _validation
from oai_agentspec._validation import (
    validate_bool,
    validate_instructions_callable,
    validate_optional_bool,
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


# ------------------------------------------------------------------
# validate_bool / validate_optional_bool: 宣言的 bool フィールド検証（Issue #58 T1・RED 先行）
# ------------------------------------------------------------------
def test_正常系_validate_bool_は真偽値を受理() -> None:
    """validate_bool は True / False をそのまま受理し例外を出さない。"""
    validate_bool(True, "enabled")
    validate_bool(False, "enabled")


def test_正常系_validate_optional_bool_は真偽値とNoneを受理() -> None:
    """validate_optional_bool は True / False / None を受理し例外を出さない。"""
    validate_optional_bool(True, "strict_mode")
    validate_optional_bool(False, "strict_mode")
    validate_optional_bool(None, "strict_mode")


def test_異常系_validate_bool_に_None_はメッセージ全文つき_ValueError() -> None:
    """None は拒否し、label 名と型名 'NoneType' を含むメッセージ全文を固定する。"""
    with pytest.raises(ValueError, match=re.escape("enabled must be a bool, got 'NoneType'")):
        validate_bool(None, "enabled")


def test_異常系_validate_bool_に文字列は_ValueError() -> None:
    """文字列（truthy な 'no' 等）は型名 'str' 入りの ValueError で拒否する。"""
    with pytest.raises(ValueError, match=re.escape("enabled must be a bool, got 'str'")):
        validate_bool("no", "enabled")


def test_異常系_validate_bool_に_int_の_0_と_1_は_ValueError() -> None:
    """bool は int の subclass だが、int の 0 / 1 は strict 判定で拒否する（Issue の核心）。"""
    with pytest.raises(ValueError, match=re.escape("enabled must be a bool, got 'int'")):
        validate_bool(0, "enabled")
    with pytest.raises(ValueError, match=re.escape("enabled must be a bool, got 'int'")):
        validate_bool(1, "enabled")


def test_異常系_validate_bool_に_float_は_ValueError() -> None:
    """float の 1.0 は型名 'float' 入りの ValueError で拒否する。"""
    with pytest.raises(ValueError, match=re.escape("enabled must be a bool, got 'float'")):
        validate_bool(1.0, "enabled")


def test_異常系_validate_optional_bool_に文字列はメッセージ全文つき_ValueError() -> None:
    """文字列は拒否し、label 名と型名 'str' を含むメッセージ全文（or None 形）を固定する。"""
    with pytest.raises(
        ValueError, match=re.escape("strict_mode must be a bool or None, got 'str'")
    ):
        validate_optional_bool("strict", "strict_mode")


def test_異常系_validate_optional_bool_に_int_は_ValueError() -> None:
    """int の 0 は strict 判定で拒否する（None 許容でも bool 以外の型は通さない）。"""
    with pytest.raises(
        ValueError, match=re.escape("strict_mode must be a bool or None, got 'int'")
    ):
        validate_optional_bool(0, "strict_mode")


# ------------------------------------------------------------------
# validate_stop_at_tool_names_shape: tool_use_behavior dict 形の形状検査（Issue #115 T1）
# ------------------------------------------------------------------
# 新関数は名前で import せず `_validation.<関数名>` で属性参照する（未実装時に本モジュール
# 全体の収集を壊さず、新規テストだけを AttributeError で落とすため）。
class FunctionTool:
    """name 属性を持つ非 str 要素のダミー（SDK の FunctionTool を誤って渡したケースの代用）。

    agents 非依存の L1 テストのため SDK 型は使わない。repr に固有文字列
    （SECRET-DESCRIPTION）を持たせ、メッセージに repr が混入しないことを確かめる。
    """

    def __init__(self) -> None:
        self.name = "refund"
        self.description = "SECRET-DESCRIPTION"

    def __repr__(self) -> str:
        return f"FunctionTool(name={self.name!r}, description={self.description!r})"


def test_異常系_stop_at_tool_names_に文字列値は全文つき_ValueError() -> None:
    """値が list / tuple でない（文字列）場合、型名と repr を含む全文の ValueError。"""
    with pytest.raises(ValueError) as excinfo:
        _validation.validate_stop_at_tool_names_shape("router", {"stop_at_tool_names": "refund"})
    assert str(excinfo.value) == (
        "agent 'router': tool_use_behavior の stop_at_tool_names は str の list / tuple "
        "である必要がありますが 'str' が渡されました: 'refund'"
    )


def test_異常系_stop_at_tool_names_キー欠落は指定キー一覧つき_ValueError() -> None:
    """dict に stop_at_tool_names キーが無い場合、指定されたキー一覧を含む全文の ValueError。"""
    with pytest.raises(ValueError) as excinfo:
        _validation.validate_stop_at_tool_names_shape("router", {"stop_at_tool_name": ["refund"]})
    assert str(excinfo.value) == (
        "agent 'router': tool_use_behavior の dict には stop_at_tool_names キーが必要です"
        "（指定されたキー: ['stop_at_tool_name']）"
    )


def test_異常系_stop_at_tool_names_キー欠落のキー一覧はソート済み() -> None:
    """キー欠落時のキー一覧は sorted 順で列挙される。"""
    with pytest.raises(ValueError) as excinfo:
        _validation.validate_stop_at_tool_names_shape("router", {"zzz": 1, "aaa": 2})
    assert str(excinfo.value) == (
        "agent 'router': tool_use_behavior の dict には stop_at_tool_names キーが必要です"
        "（指定されたキー: ['aaa', 'zzz']）"
    )


def test_異常系_stop_at_tool_names_に_name_属性つき非str要素は_ValueError() -> None:
    """name 属性が str の非 str 要素は位置・型名・name を含み、repr 由来の内容は含まない。"""
    with pytest.raises(ValueError) as excinfo:
        _validation.validate_stop_at_tool_names_shape(
            "router", {"stop_at_tool_names": [FunctionTool()]}
        )
    message = str(excinfo.value)
    assert message == (
        "agent 'router': tool_use_behavior の stop_at_tool_names[0] は str である必要が"
        "ありますが 'FunctionTool' が渡されました（name='refund'）"
    )
    assert "SECRET-DESCRIPTION" not in message


def test_異常系_stop_at_tool_names_の非str要素は位置を示す() -> None:
    """str 要素に続く非 str 要素は、その位置（[1]）をメッセージに含む。"""
    with pytest.raises(ValueError) as excinfo:
        _validation.validate_stop_at_tool_names_shape("router", {"stop_at_tool_names": ["ok", 123]})
    assert str(excinfo.value) == (
        "agent 'router': tool_use_behavior の stop_at_tool_names[1] は str である必要が"
        "ありますが 'int' が渡されました"
    )


def test_異常系_stop_at_tool_names_に_name_属性なし要素は_name_を含まない() -> None:
    """name 属性を持たない要素（123）は型名 'int' を含み、name= を含まない全文の ValueError。"""
    with pytest.raises(ValueError) as excinfo:
        _validation.validate_stop_at_tool_names_shape("router", {"stop_at_tool_names": [123]})
    message = str(excinfo.value)
    assert message == (
        "agent 'router': tool_use_behavior の stop_at_tool_names[0] は str である必要が"
        "ありますが 'int' が渡されました"
    )
    assert "name=" not in message


def test_異常系_stop_at_tool_names_の_name_属性が非strなら_name_を含まない() -> None:
    """name 属性が str でない要素は name= を付けない。"""

    class _NonStrName:
        name = 42

    with pytest.raises(ValueError) as excinfo:
        _validation.validate_stop_at_tool_names_shape(
            "router", {"stop_at_tool_names": [_NonStrName()]}
        )
    assert str(excinfo.value) == (
        "agent 'router': tool_use_behavior の stop_at_tool_names[0] は str である必要が"
        "ありますが '_NonStrName' が渡されました"
    )


def test_正常系_stop_at_tool_names_の_list_tuple_空list_は通過() -> None:
    """list / tuple / 空 list の str 要素は検証を通過する（戻り値 None）。"""
    assert (
        _validation.validate_stop_at_tool_names_shape(
            "router", {"stop_at_tool_names": ["refund", "get_order"]}
        )
        is None
    )
    assert (
        _validation.validate_stop_at_tool_names_shape("router", {"stop_at_tool_names": ("refund",)})
        is None
    )
    assert (
        _validation.validate_stop_at_tool_names_shape("router", {"stop_at_tool_names": []}) is None
    )


def test_正常系_tool_use_behavior_の非dict形は素通し() -> None:
    """文字列形 / 関数形 / None は検査対象外で例外を出さない。"""

    def behavior(context: object, results: object) -> object:
        return results

    assert _validation.validate_stop_at_tool_names_shape("router", "stop_on_first_tool") is None
    assert _validation.validate_stop_at_tool_names_shape("router", behavior) is None
    assert _validation.validate_stop_at_tool_names_shape("router", None) is None


def test_異常系_stop_at_tool_names_に非list非str値は型名と_name_のみで_repr_を含まない() -> None:
    """list で包まずに渡した非 str 値は型名と name だけを載せ、repr 由来の内容を含まない。"""
    with pytest.raises(ValueError) as excinfo:
        _validation.validate_stop_at_tool_names_shape(
            "router", {"stop_at_tool_names": FunctionTool()}
        )
    message = str(excinfo.value)
    assert message == (
        "agent 'router': tool_use_behavior の stop_at_tool_names は str の list / tuple "
        "である必要がありますが 'FunctionTool' が渡されました（name='refund'）"
    )
    assert "SECRET-DESCRIPTION" not in message


def test_異常系_stop_at_tool_names_に_name_属性なし非list値は型名のみ() -> None:
    """name 属性の無い非 list 値（dict）は型名だけで終わり、name= も値の中身も含まない。"""
    with pytest.raises(ValueError) as excinfo:
        _validation.validate_stop_at_tool_names_shape(
            "router", {"stop_at_tool_names": {"secret_key": "SECRET-VALUE"}}
        )
    message = str(excinfo.value)
    assert message == (
        "agent 'router': tool_use_behavior の stop_at_tool_names は str の list / tuple "
        "である必要がありますが 'dict' が渡されました"
    )
    assert "name=" not in message
    assert "secret_key" not in message
    assert "SECRET-VALUE" not in message


def test_異常系_stop_at_tool_names_キー欠落でキー型混在でも_ValueError() -> None:
    """キーの型が混在する dict でも TypeError にならず、repr 順のキー一覧つき ValueError。"""
    with pytest.raises(ValueError) as excinfo:
        _validation.validate_stop_at_tool_names_shape("router", {1: "x", "b": 2})
    assert str(excinfo.value) == (
        "agent 'router': tool_use_behavior の dict には stop_at_tool_names キーが必要です"
        "（指定されたキー: ['b', 1]）"
    )
