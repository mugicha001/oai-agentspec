"""ModelResponse / ストリームイベント構築ヘルパと入力正規化（SDK 結合を閉じる・NFR-1）。

`text_response` / `tool_call_response`（`ModelResponse` 構築）・`_completed_event` /
`_text_delta_events` / `_text_of`（ストリームイベント / テキスト抽出）・`latest_user_text`
（入力正規化）・`make_required_tool_choice_settings` / `make_facade_extra`（ファサード設定）を
提供する。SDK 結合（`agents` / `openai`）は本モジュール内に閉じる。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agents import (
    ItemHelpers,
    ModelSettings,
    Usage,
)
from agents.items import ModelResponse
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseTextDeltaEvent,
)

from ..constants import WORKFLOW_TOOL_CHOICE_REQUIRED, WORKFLOW_TOOL_USE_BEHAVIOR

if TYPE_CHECKING:
    from collections.abc import Iterator


def text_response(text: str) -> ModelResponse:
    """単一のアシスタントテキストメッセージを返す ModelResponse を作る。

    `tests/_helpers/responses.py` の `text_response` と同型。WorkflowModel の
    `output_extractor` 既定の出力組み立てに使う（OQ-4）。

    Args:
        text: メッセージ本文。

    Returns:
        単一テキストメッセージ・tool/handoff なしの ModelResponse。
    """
    message = ResponseOutputMessage(
        id="msg_workflow",
        type="message",
        role="assistant",
        status="completed",
        content=[ResponseOutputText(type="output_text", text=text, annotations=[])],
    )
    return ModelResponse(output=[message], usage=Usage(), response_id=None)


def tool_call_response(tool_name: str, arguments: str) -> ModelResponse:
    """単一の function ToolCall を返す ModelResponse を作る（経路D 入口モデル等で利用）。

    `text_response` と同型の `ModelResponse` 構築ヘルパで、SDK 結合を `_adapters` に集約する
    （NFR-1）。`call_id` は固定だが、利用側（決定論モデル）は `stop_on_first_tool` 前提で
    1 run につき ToolCall を 1 回しか発行しないため tool_call と tool_result の対応は衝突しない。

    Args:
        tool_name: 呼び出す tool 名。
        arguments: tool 引数の JSON 文字列。

    Returns:
        単一の function ToolCall を持つ ModelResponse。
    """
    call = ResponseFunctionToolCall(
        type="function_call",
        call_id="wf_call",
        name=tool_name,
        arguments=arguments,
    )
    return ModelResponse(output=[call], usage=Usage(), response_id=None)


# stream_response で逐次テキストを区切る擬似トークン長（後述・post-execution streaming）。
_WORKFLOW_STREAM_CHUNK = 8


def _completed_event(output_items: list[Any], sequence_number: int) -> ResponseCompletedEvent:
    """エンジン実行後の最終出力を載せた `ResponseCompletedEvent` を作る（stream_response 終端）。

    Runner のストリーミングループは `ResponseCompletedEvent.response.output` を最終出力として
    取り出す。`Response` の必須フィールドのみ埋める（SDK 結合は `_adapters` に集約・NFR-1）。

    Args:
        output_items: 最終 ModelResponse の output（テキストメッセージ or ToolCall）。
        sequence_number: イベント連番。

    Returns:
        終端の ResponseCompletedEvent。
    """
    response = Response(
        id="resp_workflow",
        created_at=0.0,
        model="oai-agentspec-workflow",
        object="response",
        output=output_items,
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
    )
    return ResponseCompletedEvent(
        response=response, sequence_number=sequence_number, type="response.completed"
    )


def _text_delta_events(text: str) -> Iterator[ResponseTextDeltaEvent]:
    """テキストを擬似トークンに区切った `ResponseTextDeltaEvent` 列を生成する（UI 逐次表示用）。

    エンジンは最終値を返す構造のため、ここでの逐次化はエンジン完了後の post-execution
    streaming（進捗的ではない）。Runner はこのイベントをそのままユーザーの stream_events へ
    転送するため、ストリーミング UI は最終出力をトークン単位で受け取れる。
    """
    for index, start in enumerate(range(0, len(text), _WORKFLOW_STREAM_CHUNK)):
        yield ResponseTextDeltaEvent(
            content_index=0,
            item_id="msg_workflow",
            output_index=0,
            delta=text[start : start + _WORKFLOW_STREAM_CHUNK],
            logprobs=[],
            sequence_number=index,
            type="response.output_text.delta",
        )


def _text_of(response: ModelResponse) -> str:
    """ModelResponse（単一テキストメッセージ）から本文テキストを取り出す。"""
    return "".join(
        part.text
        for item in response.output
        for part in getattr(item, "content", [])
        if getattr(part, "type", None) == "output_text"
    )


def latest_user_text(model_input: Any) -> Any:
    """SDK の `get_response` input から最新のユーザーテキストを取り出す（FR-10）。

    WorkflowModel に流入する input は文字列のことも input-list（`[{role, content}, ...]`）の
    こともある。先頭ワークフローノードが `[{CONTENT: ...}]` のような生メッセージ列ではなく
    素のテキストを受けられるよう、ItemHelpers で input-list へ正規化し末尾の user テキストを
    返す。文字列はそのまま返し、user テキストが見つからない場合は input をそのまま返す
    （AGENT ノード側で SDK input-list として処理できるため安全フォールバックになる）。

    Args:
        model_input: `Model.get_response` が受ける input（文字列 / input-list）。

    Returns:
        最新の user テキスト文字列、または正規化できないときは元の input。
    """
    if isinstance(model_input, str):
        return model_input
    if model_input is None:
        return ""
    try:
        items = ItemHelpers.input_to_new_input_list(model_input)
    except Exception:
        return model_input
    for item in reversed(items):
        if not isinstance(item, dict) or item.get("role") != "user":
            continue
        content = item.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = [
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") in ("input_text", "text")
            ]
            if texts:
                return "".join(texts)
    return model_input


def make_required_tool_choice_settings() -> ModelSettings:
    """`tool_choice='required'` を設定した ModelSettings を作る（経路A・FR-9）。

    tool_choice は Agent でなく ModelSettings のフィールドのため必ず model_settings 経由で
    設定する（extra に積むと build_agent の未知キーガードで ValueError になる）。

    Returns:
        tool_choice='required' の ModelSettings。
    """
    return ModelSettings(tool_choice=WORKFLOW_TOOL_CHOICE_REQUIRED)


def make_facade_extra() -> dict[str, Any]:
    """経路A ファサードの extra（`tool_use_behavior='stop_on_first_tool'`）を作る。

    tool_use_behavior は Agent フィールドなので extra で渡せる（tool_choice と非対称・FR-9）。

    Returns:
        `{"tool_use_behavior": "stop_on_first_tool"}`。
    """
    return {"tool_use_behavior": WORKFLOW_TOOL_USE_BEHAVIOR}
