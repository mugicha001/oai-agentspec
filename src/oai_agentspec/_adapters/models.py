"""ワークフロー結合の SDK Model / Tool 実装（SDK 結合を `_adapters` に閉じる・NFR-1）。

`WorkflowModel`（経路C）/ `DeterministicToolCallModel`（経路D 入口）/ `workflow_as_tool`
（経路A / D）を提供する。`ModelResponse` / ストリームイベント構築は `responses` モジュールへ
委譲する。SDK 結合（`agents` の `Model` ABC / `FunctionTool` / `ToolContext`）は本モジュール内に
閉じる。
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from agents import (
    FunctionTool,
    Model,
)
from agents.items import ModelResponse
from agents.tool_context import ToolContext

from .responses import (
    _completed_event,
    _text_delta_events,
    _text_of,
    latest_user_text,
    text_response,
    tool_call_response,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

    from ..workflow import WorkflowResult

# ワークフロー tool / handoff の無入力スキーマ（任意文字列 input を 1 つ受ける）。
_WORKFLOW_TOOL_SCHEMA: dict[str, Any] = {
    "additionalProperties": False,
    "type": "object",
    "properties": {"input": {"type": "string", "description": "ワークフローへの入力"}},
    "required": ["input"],
}


class WorkflowModel(Model):
    """LLM を呼ばず内部インタプリタを回し最終出力を ModelResponse で返す Model（経路C）。

    Runner はこれを最終出力として扱いターンを終える（決定論起動）。`get_response` は
    SDK 仕様上 run context を受け取れないため、外側 context はワークフロー内ステップへ
    伝播しない（C-11）。`stream_response` はエンジンを回しきった後に最終出力を
    `ResponseTextDeltaEvent` + `ResponseCompletedEvent` として流す（`Runner.run_streamed` 対応。
    エンジンが最終値を返す構造のため進捗的ではない post-execution streaming）。SDK `Model` ABC
    （get_response / stream_response）へ結合する（NFR-7）。
    """

    def __init__(
        self,
        interpret: Callable[..., Awaitable[WorkflowResult]],
        *,
        output_extractor: Callable[[Any], str] | None = None,
    ) -> None:
        """WorkflowModel を生成する。

        Args:
            interpret: `(input, *, context=None) -> Awaitable[WorkflowResult]`。内部
                インタプリタを既定 runner で回すクロージャ。
            output_extractor: 最終出力を単一メッセージ文字列へ変換する関数。None で str 化。
        """
        self._interpret = interpret
        self._output_extractor = output_extractor

    async def get_response(
        self,
        system_instructions: Any = None,
        input: Any = None,  # noqa: A002 - SDK Model.get_response の引数名に追従
        *args: Any,
        **kwargs: Any,
    ) -> ModelResponse:
        """input でエンジンを回し最終出力を単一メッセージ ModelResponse で返す。

        SDK `Model.get_response(system_instructions, input, ...)` の実シグネチャに合わせ
        `input` を第 2 引数として明示束縛する（位置/キーワード両様で受かる）。残りの引数は
        `*args` / `**kwargs` で吸収し SDK の引数追加に追従する。context は受け取らない
        （C-11）。SDK が `input` を改名した場合は TypeError 相当で早期検知できる。

        Returns:
            単一テキストメッセージ・tool/handoff なしの ModelResponse。
        """
        # START 入力を正規化し、先頭ノードが生メッセージ列でなく素のテキストを受ける
        # ようにする（[{CONTENT:..}] 問題の解消・FR-10）。
        start_input = latest_user_text(input)
        result = await self._interpret(start_input)
        if self._output_extractor is not None:
            text = self._output_extractor(result.final_output)
        else:
            text = str(result.final_output)
        return text_response(text)

    async def stream_response(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        """エンジンを回し最終出力を text-delta + completed イベントで流す（run_streamed 対応）。

        `get_response` と同じくエンジンを回しきってから、最終テキストを
        `ResponseTextDeltaEvent`（逐次表示用）に区切り、最後に `ResponseCompletedEvent`
        （`Runner` が最終出力として取り出す終端）を yield する。エンジンが最終値を返す構造の
        ため進捗的ではない post-execution streaming（実行完了後にトークン化して流す）。

        Yields:
            ResponseTextDeltaEvent / ResponseCompletedEvent。
        """
        response = await self.get_response(*args, **kwargs)
        text = _text_of(response)
        seq = 0
        for event in _text_delta_events(text):
            yield event
            seq = event.sequence_number + 1
        yield _completed_event(response.output, seq)


class DeterministicToolCallModel(Model):
    """毎回ワークフロー tool を 1 回呼ぶ ToolCall だけを返すステートレス決定論 Model（経路D 入口）。

    保持するのは tool 名（不変設定）のみで、可変な実行状態を一切持たない。そのため同一
    インスタンスを並行 run で共有しても安全（ステートレス）。`get_response` は入力を
    `latest_user_text` で素テキスト化し、`{"input": ...}` を引数に当該ワークフロー tool を
    1 回呼ぶ ToolCall だけを返す。LLM を介さないため決定論的で実 LLM 呼び出しは 0 回。
    `tool_use_behavior='stop_on_first_tool'` 併用が前提（無いと tool 結果後に再び同じ ToolCall
    を返し無限ループになる）。SDK `Model` ABC（get_response / stream_response）へ結合する（NFR-7）。
    """

    def __init__(self, tool_name: str) -> None:
        """決定論モデルを生成する。

        Args:
            tool_name: 毎回呼び出すワークフロー tool の名前（不変設定）。
        """
        self._tool_name = tool_name

    async def get_response(
        self,
        system_instructions: Any = None,
        input: Any = None,  # noqa: A002 - SDK Model.get_response の引数名に追従
        *args: Any,
        **kwargs: Any,
    ) -> ModelResponse:
        """当該ワークフロー tool を 1 回呼ぶ ToolCall だけを持つ ModelResponse を返す。

        入力は `latest_user_text` で素テキスト化し `{"input": ...}` を引数に載せる
        （LLM を介さず決定論的に渡す）。`latest_user_text` が user テキストを抽出できず元の
        input（list/dict 等）を返した場合は str 化し、tool スキーマ（input: string）と整合させる。
        引数は SDK の実シグネチャに合わせ input を第 2 引数で束縛し、残りは `*args` / `**kwargs`
        で吸収する。

        Returns:
            単一の function ToolCall を持つ ModelResponse。
        """
        text = latest_user_text(input)
        if not isinstance(text, str):
            text = str(text)
        return tool_call_response(self._tool_name, json.dumps({"input": text}))

    async def stream_response(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        """ToolCall を `ResponseCompletedEvent` として流す（run_streamed 対応）。

        入口は tool 呼び出しを返すだけのため、テキスト delta は流れない（最終出力は tool
        実行＝ワークフロー結果。`stop_on_first_tool` で確定）。`Runner.run_streamed` で
        クラッシュせず動くようにするための終端イベントのみを yield する。

        Yields:
            ResponseCompletedEvent（function ToolCall を載せた終端）。
        """
        response = await self.get_response(*args, **kwargs)
        yield _completed_event(response.output, 0)


def workflow_as_tool(
    interpret: Callable[..., Awaitable[WorkflowResult]],
    *,
    tool_name: str,
    tool_description: str | None = None,
    output_extractor: Callable[[Any], str] | None = None,
) -> FunctionTool:
    """内部インタプリタを回す FunctionTool を作る（経路A・context 透過）。

    `on_invoke_tool` クロージャで `tool_context.context` を内部インタプリタへ受け渡す
    （不変条件。`as_tool` と異なり SDK が自動透過しないため・FR-10）。各 AGENT ノード内側
    run の暴走上限（max_turns）等の Runner kwarg はグラフ既定 `run_defaults` / ノード
    `run_options` で設定する（passthrough・FR-15）。

    Args:
        interpret: `(input, *, context=None) -> Awaitable[WorkflowResult]`。
        tool_name: ワークフロー tool 名。
        tool_description: ワークフロー tool の説明（任意）。
        output_extractor: 最終出力を文字列化する関数（None で str 化）。

    Returns:
        agents.FunctionTool。
    """

    async def on_invoke_tool(tool_context: ToolContext[Any], input_json: str) -> str:
        try:
            payload = json.loads(input_json) if input_json else {}
        except json.JSONDecodeError:
            payload = {"input": input_json}
        model_input = payload.get("input", "") if isinstance(payload, dict) else input_json
        # 関数ノード / router へは RunContextWrapper（ToolContext）をそのまま渡す。
        # AGENT ノードの runner シームが .context を取り出して Runner.run へ渡す。
        result = await interpret(model_input, context=tool_context)
        if output_extractor is not None:
            return output_extractor(result.final_output)
        return str(result.final_output)

    return FunctionTool(
        name=tool_name,
        description=tool_description or "",
        params_json_schema=dict(_WORKFLOW_TOOL_SCHEMA),
        on_invoke_tool=on_invoke_tool,
    )
