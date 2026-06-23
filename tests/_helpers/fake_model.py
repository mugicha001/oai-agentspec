"""本物の LLM を呼ばずに Runner を駆動するための FakeModel。

openai-agents の `Model` ABC（`get_response` / `stream_response`）を継承し、
カンネドレスポンスを順に返しつつ呼び出しを記録する。SDK バージョン差に強いよう
`*args, **kwargs` で受ける。`stream_response` は本ライブラリのテストでは非対応。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from agents.items import ModelResponse
from agents.models.interface import Model

from .responses import text_response, tool_call_response


@dataclass
class _Call:
    system_instructions: str | None
    input: Any
    tools: Any
    handoffs: Any


@dataclass
class FakeModel(Model):
    """カンネドレスポンスを返し呼び出しを記録するテスト用 Model。

    Attributes:
        responses: 順に返す ModelResponse のキュー。空なら空テキストを返す。
        calls: 各 get_response 呼び出しの記録。
    """

    responses: list[ModelResponse] = field(default_factory=list)
    calls: list[_Call] = field(default_factory=list)

    def queue_text(self, text: str) -> FakeModel:
        """テキスト応答をキューに積む（自身を返す）。"""
        self.responses.append(text_response(text))
        return self

    def queue_tool_call(self, name: str, arguments: str) -> FakeModel:
        """function ToolCall 応答をキューに積む（自身を返す）。

        `queue_text` が `text_response` を使うのと対称に、共通ヘルパ `tool_call_response` で
        構築する。

        Args:
            name: 呼び出す tool 名。
            arguments: tool 引数の JSON 文字列。
        """
        self.responses.append(tool_call_response(name, arguments))
        return self

    async def get_response(
        self,
        system_instructions: str | None,
        input: Any,
        *args: Any,
        **kwargs: Any,
    ) -> ModelResponse:
        tools = args[1] if len(args) > 1 else kwargs.get("tools")
        handoffs = args[3] if len(args) > 3 else kwargs.get("handoffs")
        self.calls.append(
            _Call(
                system_instructions=system_instructions,
                input=input,
                tools=tools,
                handoffs=handoffs,
            )
        )
        if self.responses:
            return self.responses.pop(0)
        return text_response("")

    def stream_response(  # type: ignore[override]
        self,
        system_instructions: str | None,
        input: Any,
        *args: Any,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        raise NotImplementedError("FakeModel はストリーミング非対応")


@dataclass
class ChoiceAwareModel(Model):
    """tool_choice='required' のときだけ ToolCall を返し、それ以外は text を返す検証用 Model。

    `reset_tool_choice` による 2 ターン目の tool_choice 解除を観測するために使う。
    reset 既定（True）なら 2 ターン目は tool_choice が None に戻り text を返して終了する。
    reset=False なら 2 ターン目も required のままで ToolCall を返し続け max_turns 例外になる。

    Attributes:
        tool_name: required 時に呼ぶ tool 名。
        text: required でないとき返すテキスト。
        calls: get_response 呼び出し回数。
    """

    tool_name: str = "wf_tool"
    text: str = "done"
    calls: int = 0

    async def get_response(
        self,
        system_instructions: str | None,
        input: Any,
        *args: Any,
        **kwargs: Any,
    ) -> ModelResponse:
        self.calls += 1
        model_settings = args[0] if args else kwargs.get("model_settings")
        tool_choice = getattr(model_settings, "tool_choice", None)
        if tool_choice == "required":
            return tool_call_response(self.tool_name, '{"input": "x"}')
        return text_response(self.text)

    def stream_response(  # type: ignore[override]
        self,
        system_instructions: str | None,
        input: Any,
        *args: Any,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        raise NotImplementedError("ChoiceAwareModel はストリーミング非対応")
