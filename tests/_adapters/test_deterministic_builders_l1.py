"""L1: 公開応答ビルダ 5 種の出力構造と既定 id 値の pin（Issue #70 テーマB・FR-5 / FR-6）。

`oai_agentspec._adapters.deterministic` が提供する応答ビルダ（`text_response` /
`text_response_with_usage` / `tool_call_response` / `multi_tool_call_response` /
`mixed_response`）が、設計方針 §3-2 の JSON サンプルどおりの `ModelResponse` を組み立てる
ことを固定する。ビルダは `agents` / `openai` の SDK item 型を直接組むため、SDK の必須
フィールド追加・既定値変更をここで検知する。

本モジュールは 2 系統のテストを持つ。

1. **新規公開ビルダの契約**: 実装（`_adapters/deterministic.py`）が存在しない間は
   `ModuleNotFoundError` で失敗する（RED）。新規 API は `_builders()` 経由の遅延 import で
   参照し、実装不在が下記 2 の回帰 pin を巻き込んで collection error にならないようにする。
2. **既存ワークフロー経路の回帰 pin**（`_adapters/responses.py` / `_adapters/models.py`）:
   実装前から緑で、共有 item ヘルパ化・streaming ヘルパの引数化（タスク B2）を経ても
   ワークフロー用の id 値（`msg_workflow` / `wf_call` / `id is None` / `resp_workflow` /
   `oai-agentspec-workflow`）が変わらないことを守る。id 値が変わっても run は壊れず
   テストも通ってしまうため、リテラル pin が唯一の検知手段になる。

既定 id 値は定数のリネームで pin が空振りしないよう**文字列リテラル**で照合する。
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import pytest

from oai_agentspec._adapters.models import DeterministicToolCallModel, WorkflowModel
from oai_agentspec._adapters.responses import text_response as workflow_text_response
from oai_agentspec._adapters.responses import tool_call_response as workflow_tool_call_response

pytestmark = pytest.mark.unit


# 公開ビルダの既定 id 値（設計方針 決定 6(e)）。値そのものが契約なのでリテラルで持つ。
_EXPECTED_DEFAULT_IDS = (
    "msg_deterministic",
    "fc_deterministic",
    "call_deterministic",
    "resp_deterministic",
    "oai-agentspec-deterministic",
)

# 公開面の既定 id 値に現れてはならない語（FR-6。`workflow` / `wf` は内部ワークフロー用 id）。
_FORBIDDEN_ID_WORDS = ("fake", "mock", "dummy", "workflow", "wf")


def _builders() -> ModuleType:
    """公開応答ビルダのモジュールを取得する（未実装の間は ImportError で RED になる）。

    Returns:
        `oai_agentspec._adapters.deterministic` モジュール。
    """
    return importlib.import_module("oai_agentspec._adapters.deterministic")


def _module_string_constants(module: ModuleType) -> dict[str, str]:
    """モジュール直下の文字列定数（SCREAMING_SNAKE_CASE）を集める。

    Args:
        module: 走査対象モジュール。

    Returns:
        `{定数名: 値}`。dunder 属性・非文字列は除外する。
    """
    return {
        name: value
        for name, value in vars(module).items()
        if name.isupper() and isinstance(value, str)
    }


@dataclass(frozen=True)
class _StubWorkflowResult:
    """`WorkflowModel` が受け取る内部インタプリタ結果の代役。"""

    final_output: str


async def _interpret_echo(model_input: Any) -> _StubWorkflowResult:
    """入力をそのまま最終出力に載せて返す内部インタプリタ代役。

    Args:
        model_input: `WorkflowModel` が正規化した START 入力。

    Returns:
        `final_output` に echo 文字列を持つ結果。
    """
    return _StubWorkflowResult(final_output=f"echo: {model_input}")


# ---------------------------------------------------------------------------
# text_response / text_response_with_usage
# ---------------------------------------------------------------------------
def test_text_response_は単一のアシスタントテキストメッセージを返す() -> None:
    """`text_response` は message item 1 件・usage 全 0・response_id None を返す。"""
    response = _builders().text_response("こんにちは")

    assert len(response.output) == 1
    message = response.output[0]
    assert message.id == "msg_deterministic"
    assert message.type == "message"
    assert message.role == "assistant"
    assert message.status == "completed"
    assert len(message.content) == 1
    part = message.content[0]
    assert part.type == "output_text"
    assert part.text == "こんにちは"
    assert part.annotations == []
    assert response.usage.requests == 0
    assert response.usage.input_tokens == 0
    assert response.usage.output_tokens == 0
    assert response.usage.total_tokens == 0
    assert response.response_id is None


def test_text_response_with_usage_は指定した_usage_を反映する() -> None:
    """`text_response_with_usage` は total_tokens / requests を usage へ載せる。"""
    response = _builders().text_response_with_usage("ok", total_tokens=120, requests=3)

    message = response.output[0]
    assert message.id == "msg_deterministic"
    assert message.content[0].text == "ok"
    assert response.usage.total_tokens == 120
    assert response.usage.requests == 3
    assert response.response_id is None


def test_text_response_with_usage_の_requests_既定値は_1() -> None:
    """`requests` 省略時は 1（usage 欠損検知の回避）で、他のトークン項目は 0 のまま。"""
    response = _builders().text_response_with_usage("ok", total_tokens=7)

    assert response.usage.requests == 1
    assert response.usage.total_tokens == 7
    assert response.usage.input_tokens == 0
    assert response.usage.output_tokens == 0


# ---------------------------------------------------------------------------
# tool_call_response / multi_tool_call_response
# ---------------------------------------------------------------------------
def test_tool_call_response_は既定の_call_id_と引数で単一_ToolCall_を返す() -> None:
    """`arguments` / `call_id` 省略時は `"{}"` / `call_deterministic` になる。"""
    response = _builders().tool_call_response("transfer_to_booking")

    assert len(response.output) == 1
    call = response.output[0]
    assert call.id == "fc_deterministic"
    assert call.type == "function_call"
    assert call.call_id == "call_deterministic"
    assert call.name == "transfer_to_booking"
    assert call.arguments == "{}"
    assert response.usage.requests == 0
    assert response.usage.total_tokens == 0
    assert response.response_id is None


def test_tool_call_response_は_call_id_をキーワードで指定できる() -> None:
    """`call_id` 指定時は call_id のみが変わり、item id は既定値のまま固定される。"""
    response = _builders().tool_call_response("book", '{"n": 1}', call_id="call_x")

    call = response.output[0]
    assert call.call_id == "call_x"
    assert call.id == "fc_deterministic"
    assert call.name == "book"
    assert call.arguments == '{"n": 1}'


def test_multi_tool_call_response_は宣言順に複数_ToolCall_を並べる() -> None:
    """複数 ToolCall を 1 応答へ載せ、item id は `fc_<call_id>` で個別化される。"""
    response = _builders().multi_tool_call_response(
        [("book", '{"n":1}', "call_a"), ("pay", "{}", "call_b")]
    )

    assert len(response.output) == 2
    first, second = response.output
    assert (first.id, first.call_id, first.name, first.arguments) == (
        "fc_call_a",
        "call_a",
        "book",
        '{"n":1}',
    )
    assert (second.id, second.call_id, second.name, second.arguments) == (
        "fc_call_b",
        "call_b",
        "pay",
        "{}",
    )
    assert [item.type for item in response.output] == ["function_call", "function_call"]
    assert response.usage.requests == 0
    assert response.usage.total_tokens == 0
    assert response.response_id is None


# ---------------------------------------------------------------------------
# mixed_response
# ---------------------------------------------------------------------------
def test_mixed_response_はテキスト_1_件と_ToolCall_を順に載せる() -> None:
    """テキストメッセージが先頭、続けて宣言順の ToolCall が並ぶ（1 応答で混在）。"""
    response = _builders().mixed_response(
        "少々お待ちください",
        [("book", '{"n":1}', "call_a"), ("pay", "{}", "call_b")],
    )

    assert [item.type for item in response.output] == [
        "message",
        "function_call",
        "function_call",
    ]
    message = response.output[0]
    assert message.id == "msg_deterministic"
    assert message.role == "assistant"
    assert message.status == "completed"
    assert message.content[0].text == "少々お待ちください"
    assert [item.id for item in response.output[1:]] == ["fc_call_a", "fc_call_b"]
    assert [item.call_id for item in response.output[1:]] == ["call_a", "call_b"]
    assert [item.name for item in response.output[1:]] == ["book", "pay"]
    assert [item.arguments for item in response.output[1:]] == ['{"n":1}', "{}"]
    assert response.response_id is None


def test_mixed_response_の_usage_既定は_0_で指定すると反映される() -> None:
    """`total_tokens` / `requests` の既定は 0、指定すると usage へ反映される。"""
    default = _builders().mixed_response("待って", [("book", "{}", "call_a")])
    assert default.usage.requests == 0
    assert default.usage.total_tokens == 0

    with_usage = _builders().mixed_response(
        "待って", [("book", "{}", "call_a")], total_tokens=42, requests=1
    )
    assert with_usage.usage.requests == 1
    assert with_usage.usage.total_tokens == 42


def test_mixed_response_は空の_calls_でもテキストのみを返す() -> None:
    """`calls` が空列（Sequence）ならテキストメッセージ 1 件だけの応答になる。"""
    response = _builders().mixed_response("だけ", ())

    assert len(response.output) == 1
    assert response.output[0].type == "message"
    assert response.output[0].content[0].text == "だけ"


# ---------------------------------------------------------------------------
# 既定 id 値の pin（ADR 0019 の Confirmation）
# ---------------------------------------------------------------------------
def test_既定_id_値が禁止語を含まない固定値である() -> None:
    """既定 id 定数 5 件がリテラルどおりで、禁止語（Fake/Mock/Dummy/workflow/wf）を含まない。

    id 値の変更は run を壊さず通るため、リテラル pin が唯一の検知手段になる（FR-6）。
    定数のリネームで pin が空振りしないよう、名前ではなく**値**の集合で照合する。
    """
    module = _builders()
    constants = _module_string_constants(module)

    assert set(_EXPECTED_DEFAULT_IDS) <= set(constants.values())
    for name, value in constants.items():
        lowered = value.lower()
        for word in _FORBIDDEN_ID_WORDS:
            assert word not in lowered, f"{name}={value!r} が禁止語 {word!r} を含む"

    # 定数宣言とビルダ出力の乖離（定数だけ直してビルダは旧値のまま）を防ぐ。
    assert module.text_response("x").output[0].id == "msg_deterministic"
    call = module.tool_call_response("t").output[0]
    assert (call.id, call.call_id) == ("fc_deterministic", "call_deterministic")


# ---------------------------------------------------------------------------
# 既存ワークフロー経路の回帰 pin（実装前から緑）
# ---------------------------------------------------------------------------
def test_既存ワークフロー用ビルダの_id_値が不変である() -> None:
    """`_adapters/responses.py` の 2 ビルダの id 値を現状のまま固定する。

    タスク B2 で共有 item ヘルパへ委譲しても、ワークフロー経路の応答 id
    （`msg_workflow` / `wf_call` / `ResponseFunctionToolCall.id is None`）は変わらない。
    """
    response = workflow_text_response("hi")
    message = response.output[0]
    assert message.id == "msg_workflow"
    assert message.type == "message"
    assert message.role == "assistant"
    assert message.status == "completed"
    assert message.content[0].text == "hi"
    assert message.content[0].annotations == []
    assert response.usage.requests == 0
    assert response.response_id is None

    call_response = workflow_tool_call_response("wf_tool", '{"input": "x"}')
    call = call_response.output[0]
    assert call.id is None
    # SDK は `RunItem.to_input_item()` で `model_dump(exclude_unset=True)` を使う
    # （`agents/items.py`）。`id=None` を明示代入すると input-list と Session 保存内容へ
    # `"id": null` が現れて既存挙動が変わるため、id が未設定のままであることを pin する
    # （`call.id is None` は明示代入でも通るため区別できない）。
    assert "id" not in call.model_dump(exclude_unset=True)
    assert call.type == "function_call"
    assert call.call_id == "wf_call"
    assert call.name == "wf_tool"
    assert call.arguments == '{"input": "x"}'
    assert call_response.usage.requests == 0
    assert call_response.response_id is None


async def test_既存_WorkflowModel_の_streaming_識別子が不変である() -> None:
    """`WorkflowModel.stream_response` のイベント識別子を現状のまま固定する。

    タスク B2 で `_text_delta_events` / `_completed_event` を引数化しても、ワークフロー
    経路の呼び出しは従来値（`msg_workflow` / `resp_workflow` / `oai-agentspec-workflow`）
    を渡し続ける（挙動不変）。
    """
    model = WorkflowModel(_interpret_echo)

    events = [event async for event in model.stream_response(None, "hello streaming text")]

    deltas, completed = events[:-1], events[-1]
    assert deltas, "テキスト応答では差分イベントが 1 件以上流れる"
    assert {event.item_id for event in deltas} == {"msg_workflow"}
    assert "".join(event.delta for event in deltas) == "echo: hello streaming text"
    assert completed.type == "response.completed"
    assert completed.response.id == "resp_workflow"
    assert completed.response.model == "oai-agentspec-workflow"
    assert completed.response.output[0].id == "msg_workflow"


async def test_既存_DeterministicToolCallModel_の_streaming_識別子が不変である() -> None:
    """ToolCall のみの応答では差分が流れず、終端イベントの識別子も現状のまま固定される。"""
    model = DeterministicToolCallModel("wf_tool")

    events = [event async for event in model.stream_response(None, "hi")]

    assert len(events) == 1
    completed = events[0]
    assert completed.type == "response.completed"
    assert completed.response.id == "resp_workflow"
    assert completed.response.model == "oai-agentspec-workflow"
    assert completed.response.output[0].call_id == "wf_call"
