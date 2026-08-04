"""HITL（ツール実行承認）テスト用ヘルパ（承認必須ツール / 応答キュー付き FakeModel）。

承認必須ツールは `function_tool(..., needs_approval=True)` で作る（`oai_agentspec` から公開済み）。
ツールには「実行されたら call_id 単位で記録するフラグ」を持たせ、NFR-7（承認前は実行記録ゼロ・
approve 後のみ実行）を検証できるようにする。

FakeModel は「承認必須ツールへの ToolCall を返す応答」→（再開後）「テキストを返す応答」を順に
キューする。Runner が needs_approval ツールへの ToolCall を検知して `result.interruptions` を
生成する（FakeModel が interruptions を直接返すのではない）。本パッケージは tests 配下であり
NFR-1 の grep 計測対象外（`agents` を直接 import してよい）。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from agents.items import ModelResponse
from agents.models.interface import Model

from oai_agentspec.runtime.deterministic import text_response


@dataclass
class ToolRecorder:
    """承認必須ツールの実行記録（NFR-7 の実行副作用観測点）。

    Attributes:
        executed: 実行されたツール引数（`x`）の列。承認前は空のまま。
        calls: 実行された (tool_name, x) のタプル列（複数ツール識別用）。
    """

    executed: list[str] = field(default_factory=list)
    calls: list[tuple[str, str]] = field(default_factory=list)


def make_approval_tool(recorder: ToolRecorder, *, name: str = "danger") -> Any:
    """承認必須ツールを作る（実行されたら recorder に記録する・needs_approval=True）。

    SDK の `needs_approval=True` で中断発火する関数ツールを返す。ツール本体は recorder へ
    実行記録を残すため、承認前は実行されていない（記録ゼロ）こと・approve 後のみ実行された
    ことを検証できる（NFR-7）。

    Args:
        recorder: 実行記録を蓄積する `ToolRecorder`。
        name: ツール名（複数承認待ちで識別するため変更可）。

    Returns:
        registry の `AgentSpec(tools=[...])` に載せる承認必須ツール。
    """
    from oai_agentspec import function_tool

    @function_tool(name_override=name, needs_approval=True)
    def _tool(x: str) -> str:
        """承認必須ツール（実行されたら記録する）。"""
        recorder.executed.append(x)
        recorder.calls.append((name, x))
        return f"{name}:{x}"

    return _tool


@dataclass
class QueuedFakeModel(Model):
    """カンネド ModelResponse を順に返す FakeModel（承認待ち再現用・非ストリーミング）。

    承認必須ツールへの ToolCall を返す応答 → 再開後のテキスト応答、の順にキューして使う。
    Runner が ToolCall を検知して interruptions を生成するため、本 Model 自身は通常の
    ModelResponse を返すだけでよい。

    Attributes:
        responses: 順に返す ModelResponse のキュー。空なら空テキストを返す。
        inputs: 各 get_response の input 記録（履歴継続/却下反映の検証用）。
    """

    responses: list[ModelResponse] = field(default_factory=list)
    inputs: list[Any] = field(default_factory=list)

    def queue(self, response: ModelResponse) -> QueuedFakeModel:
        """ModelResponse をキューに積む（自身を返す）。"""
        self.responses.append(response)
        return self

    async def get_response(
        self,
        system_instructions: str | None = None,
        input: Any = None,  # noqa: A002 - SDK シグネチャに追従
        *args: Any,
        **kwargs: Any,
    ) -> ModelResponse:
        """キュー先頭の応答を返す（空なら空テキスト）。"""
        self.inputs.append(input)
        if self.responses:
            return self.responses.pop(0)
        return text_response("")

    async def stream_response(  # type: ignore[override]
        self,
        system_instructions: str | None = None,
        input: Any = None,  # noqa: A002 - SDK シグネチャに追従
        *args: Any,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """最終テキストを delta + completed イベントで流す（run_streamed 対応）。

        承認必須ツールへの ToolCall を含む応答では SDK 側が中断するため、本メソッドは
        ToolCall を含むキューでも text 抽出は空になり delta は流れない（承認待ちが先行する）。
        """
        from oai_agentspec._adapters import _completed_event, _text_delta_events, _text_of

        response = await self.get_response(system_instructions, input, *args, **kwargs)
        text = _text_of(response)
        seq = 0
        for event in _text_delta_events(text, item_id="msg_stream_fake"):
            yield event
            seq = event.sequence_number + 1
        yield _completed_event(
            response.output, seq, response_id="resp_stream_fake", model="stream-fake"
        )
