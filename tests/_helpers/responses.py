"""ModelResponse 構築ヘルパー（FakeModel が返す応答を組み立てる）。"""

from __future__ import annotations

from agents import Usage
from agents.items import ModelResponse
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)


def text_response(text: str) -> ModelResponse:
    """単一のアシステントテキストメッセージを返す ModelResponse を作る。"""
    message = ResponseOutputMessage(
        id="msg_fake",
        type="message",
        role="assistant",
        status="completed",
        content=[ResponseOutputText(type="output_text", text=text, annotations=[])],
    )
    return ModelResponse(output=[message], usage=Usage(), response_id=None)


def tool_call_response(
    name: str, arguments: str = "{}", call_id: str = "call_fake"
) -> ModelResponse:
    """単一の関数ツール呼び出しを返す ModelResponse を作る（handoff/as_tool 誘発用）。"""
    call = ResponseFunctionToolCall(
        id="fc_fake",
        type="function_call",
        call_id=call_id,
        name=name,
        arguments=arguments,
    )
    return ModelResponse(output=[call], usage=Usage(), response_id=None)


def multi_tool_call_response(
    calls: list[tuple[str, str, str]],
) -> ModelResponse:
    """複数の関数ツール呼び出しを 1 応答で返す ModelResponse を作る（複数承認待ち FR-7 用）。

    各 ToolCall は異なる `call_id` を持てる。承認必須ツールを 1 ターンで複数呼ぶ
    （同時に複数の承認待ちを生む）シナリオの再現に使う。

    Args:
        calls: `(name, arguments, call_id)` のタプル列。`call_id` は一意にすること。

    Returns:
        複数の `ResponseFunctionToolCall` を output に並べた `ModelResponse`。
    """
    output = [
        ResponseFunctionToolCall(
            id=f"fc_{call_id}",
            type="function_call",
            call_id=call_id,
            name=name,
            arguments=arguments,
        )
        for name, arguments, call_id in calls
    ]
    return ModelResponse(output=output, usage=Usage(), response_id=None)
