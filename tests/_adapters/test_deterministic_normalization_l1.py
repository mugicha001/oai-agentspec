"""L1: `_adapters.deterministic` の入力正規化ヘルパの pin（`_input_items` / `_item_fields`）。

`ModelRequest` の導出フィールド（`user_text` / `turn` / `tool_outputs`）はすべて
`_input_items` が返す item 列から導出され、各 item の role / type は `_item_fields` が
取り出す。両ヘルパの防御分岐（戻り値の list 検査・SDK 正規化の `TypeError`・dict でない
item の属性参照）は、`ModelRequest` のどのフィールドからも観測できないか他の分岐と帰結が
重なるため、公開経路（`get_response` / `Runner`）のテストでは固定できない。ここでは私有
ヘルパを直接呼んで分岐そのものを pin する（`ModelRequest` 経由の契約は L2 側で pin 済み）。
"""

from __future__ import annotations

from typing import Any

import pytest
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)

from oai_agentspec._adapters import deterministic

pytestmark = pytest.mark.unit


class _TypeErrorOnIter:
    """走査すると `TypeError` を送出する入力（SDK 正規化が例外を投げる経路の再現）。"""

    def __iter__(self) -> Any:
        """走査開始時点で `TypeError` を送出する。"""
        raise TypeError("iteration failed")


class _RoleAndTypeAttributes:
    """role / type を属性で持つ非 iterable な item（dict でない item の再現）。"""

    role = "assistant"
    type = "message"


def _sdk_message() -> ResponseOutputMessage:
    """SDK のアシスタントメッセージ item を作る。"""
    return ResponseOutputMessage(
        id="msg_1",
        content=[ResponseOutputText(text="hi", type="output_text", annotations=[])],
        role="assistant",
        status="completed",
        type="message",
    )


def _sdk_function_call() -> ResponseFunctionToolCall:
    """SDK の function ToolCall item を作る。"""
    return ResponseFunctionToolCall(
        id="fc_1",
        call_id="call_1",
        name="add_one",
        arguments="{}",
        type="function_call",
    )


# ---------------------------------------------------------------------------
# _input_items（正規化の唯一の起点）
# ---------------------------------------------------------------------------
def test_input_items_は_dict_単体では戻り値の_list_検査で空列になる() -> None:
    """input-list でない単体 dict は空列へ倒す（戻り値の list 検査の pin）。

    `ItemHelpers.input_to_new_input_list` は単体 dict をそのまま返すため、素朴に `list()`
    するとキー文字列の列（`["role", "content"]`）という非 item 列ができる。この検査は
    `_count_turns` の非 item ガードと帰結が重なり `ModelRequest` のどのフィールドからも
    観測できないため、直接呼び出しでのみ固定できる。
    """
    assert deterministic._input_items({"role": "user", "content": "x"}) == []


def test_input_items_は_list_へ正規化できない値でも空列になる() -> None:
    """`None` や非 iterable はそのまま返るため、戻り値の list 検査で空列へ倒す。"""
    assert deterministic._input_items(None) == []
    assert deterministic._input_items(42) == []


def test_input_items_は_SDK_正規化の_TypeError_を空列へ倒す() -> None:
    """SDK 正規化が `TypeError` を投げる入力でも例外を持ち出さず空列にする。

    `ItemHelpers.input_to_new_input_list` は iterable を要素ごとに走査するため、走査自体が
    `TypeError` になる入力では例外が上がる。これを `get_response` の入口で持ち出すと
    ルール関数へ到達する前に run が落ちるため、導出フィールドを空値へ倒す側に寄せる。
    """
    assert deterministic._input_items(_TypeErrorOnIter()) == []


def test_input_items_は文字列を_user_メッセージ_1_件へ展開する() -> None:
    """正常系（文字列入力）は SDK の展開結果をそのまま list で返す。"""
    assert deterministic._input_items("hello") == [{"content": "hello", "role": "user"}]


async def test_get_response_は_input_が_None_でも導出フィールドが空値になる() -> None:
    """`input=None` でも例外にならず、導出フィールドが空値の `ModelRequest` が組める。"""
    captured: list[Any] = []

    def rule(request: Any) -> Any:
        captured.append(request)
        return deterministic.text_response("ok")

    model = deterministic.DeterministicResponseModel(rule)

    await model.get_response(system_instructions=None, input=None)

    request = captured[0]
    assert request.user_text == ""
    assert request.turn == 0
    assert request.tool_outputs == ()


# ---------------------------------------------------------------------------
# _item_fields（role / type の取り出し）
# ---------------------------------------------------------------------------
def test_item_fields_は_dict_item_からキーで_role_と_type_を取る() -> None:
    """dict item は `get` で role / type を取り、欠けているキーは None になる。"""
    assert deterministic._item_fields({"role": "user", "content": "x"}) == ("user", None)
    assert deterministic._item_fields({"type": "function_call_output"}) == (
        None,
        "function_call_output",
    )


def test_item_fields_は_dict_でない_item_から属性で_role_と_type_を取る() -> None:
    """dict でない item は属性参照へ倒す（`_is_turn_boundary` / `_collect_tool_outputs` の起点）。

    SDK は input item をオブジェクトのまま渡すことがあり、属性参照の分岐が落ちると role も
    type も取れない要素として扱われ、ターン数と tool 実行結果の導出が静かに狂う。
    """
    assert deterministic._item_fields(_RoleAndTypeAttributes()) == ("assistant", "message")
    # role / type のどちらも持たないオブジェクトは (None, None)（`_count_turns` の非計上条件）。
    assert deterministic._item_fields(object()) == (None, None)


def test_属性で_role_を持つ_item_はモデル応答として数えられる() -> None:
    """属性参照で取れた role が `_count_turns` のターン境界判定まで届く。"""
    items = [{"role": "user", "content": "x"}, _RoleAndTypeAttributes()]

    assert deterministic._count_turns(items) == 1


def test_input_list_の_SDK_オブジェクトは_list_化され_role_と_type_が取れなくなる() -> None:
    """SDK item オブジェクトを input-list へ混ぜると `[[key, value], ...]` の list になる。

    SDK の正規化（`agents.util._json._to_dump_compatible`）は pydantic モデルを `Iterable`
    と見なして `[[key, value], ...]` の list へ変換するため、正規化後の要素は dict でも属性
    持ちオブジェクトでもなくなり `_item_fields` は `(None, None)` を返す。結果として
    `_count_turns` は計上せず `_collect_tool_outputs` も拾わない（実測された挙動の pin）。
    """
    items = deterministic._input_items(
        [{"role": "user", "content": "x"}, _sdk_message(), _sdk_function_call()]
    )

    assert [type(item) for item in items] == [dict, list, list]
    assert deterministic._item_fields(items[1]) == (None, None)
    assert deterministic._item_fields(items[2]) == (None, None)
    assert deterministic._count_turns(items) == 0
    assert deterministic._collect_tool_outputs(items) == ()
