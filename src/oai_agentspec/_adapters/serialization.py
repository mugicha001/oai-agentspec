"""RunState シリアライズと会話ストリーミング実行の委譲アダプタ（SDK 結合を閉じる・NFR-1）。

`serialize_state` / `deserialize_state`（不透明 `RunState` の JSON 往復）/ `run_streamed_outcome`
（断片 + 終端 `RunOutcome` のストリーミング）/ `run_streamed_text`（断片 + 完了の plain イベント
ストリーミング）を提供する。SDK 結合（`agents` の `Runner` / `RunState` / streaming イベント型 /
`RunContextWrapper`）は本モジュール内に閉じ、外へは plain な値（テキスト断片・最終出力・plain
イベント）のみを渡す。

`run_streamed_text` が yield する plain イベント型（`StreamTextDelta` / `StreamTextDone`）は
`_adapters` 内に閉じた中立 dataclass で、`runtime` 層（`conversation` 等）へ依存しない（NFR-5）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from agents import (
    RawResponsesStreamEvent,
    RunContextWrapper,
    Runner,
    RunState,
)
from openai.types.responses import (
    ResponseTextDeltaEvent,
)

from .runner import _outcome_from_result

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from .runner import RunOutcome

__all__ = [
    "StreamTextDelta",
    "StreamTextDone",
    "deserialize_state",
    "run_streamed_outcome",
    "run_streamed_text",
    "serialize_state",
]


@dataclass(frozen=True)
class StreamTextDelta:
    """`run_streamed_text` が流すテキスト断片（`_adapters` 内に閉じた中立 plain 型）。

    Attributes:
        text: 直近に生成されたテキスト断片。
    """

    text: str


@dataclass(frozen=True)
class StreamTextDone:
    """`run_streamed_text` の完了イベント（`_adapters` 内に閉じた中立 plain 型）。

    Attributes:
        final_output: 会話ターンの最終出力（通常はテキスト全文）。
    """

    final_output: str


async def serialize_state(state: Any) -> str:
    """不透明 `RunState` を plain JSON 文字列へシリアライズする（`to_string` 流用）。

    Args:
        state: シリアライズする SDK `RunState`（不透明 `Any`）。

    Returns:
        `RunState.to_string()` の JSON 文字列。
    """
    return state.to_string()


async def deserialize_state(initial_agent: Any, state_str: str) -> Any:
    """plain JSON 文字列を不透明 `RunState` へ復元する（`from_string` 流用・async）。

    Args:
        initial_agent: 復元の起点 SDK Agent（registry 解決済み・D-Resume）。
        state_str: `serialize_state` が出力した JSON 文字列。

    Returns:
        復元した SDK `RunState`（不透明 `Any`）。
    """
    return await RunState.from_string(initial_agent, state_str)


async def run_streamed_outcome(
    agent: Any,
    input: Any,  # noqa: A002 - Runner.run_streamed の引数名に追従
    *,
    session: Any = None,
    context: Any = None,
    **runner_kwargs: Any,
) -> AsyncIterator[str | RunOutcome]:
    """`Runner.run_streamed` を回し断片を逐次 yield し、最後に `RunOutcome` を 1 件 yield する。

    `stream_events()` を回してテキスト断片（plain `str`）を逐次 yield し、ストリーム終了後に
    `interruptions` を確認した `RunOutcome`（最後の 1 件）を yield する。中断ありなら
    `RunOutcome.interrupted=True` で承認待ち一覧 + 不透明 `RunState` を、中断なしなら最終出力を
    載せる。`input` に `RunState` を渡せばストリーム再開になる（ストリーム再開両対応）。

    共有コアは「`str` が来たら断片、`RunOutcome` が来たら終端」と判別でき、中断ターンでは
    `StreamDone` 相当を流さず承認待ちを別経路（`ApprovalRequired`）で扱える（NFR-6）。

    Args:
        agent: 実行する SDK Agent（解決済み）。
        input: ターンの入力（文字列 / input-list / 再開時は `RunState`）。
        session: SDK `Session`（履歴保持）。None で履歴なし。
        context: 各実行へ素通しする共有 context（`RunContextWrapper` は `.context` を展開）。
        **runner_kwargs: `Runner.run_streamed` へ素通しする残りの kwarg（max_turns 等）。

    Yields:
        テキスト断片（`str`）を 0 件以上、最後に終端の `RunOutcome` を 1 件。
    """
    raw_context = context.context if isinstance(context, RunContextWrapper) else context
    streamed = Runner.run_streamed(
        agent, input, context=raw_context, session=session, **runner_kwargs
    )
    async for event in streamed.stream_events():
        if isinstance(event, RawResponsesStreamEvent) and isinstance(
            event.data, ResponseTextDeltaEvent
        ):
            yield event.data.delta
    yield _outcome_from_result(streamed)


async def run_streamed_text(
    agent: Any,
    input: Any,  # noqa: A002 - Runner.run_streamed の引数名に追従
    *,
    session: Any = None,
    context: Any = None,
    **runner_kwargs: Any,
) -> AsyncIterator[StreamTextDelta | StreamTextDone]:
    """`Runner.run_streamed` を回しテキスト断片 / 完了を plain イベントで逐次 yield する。

    `stream_events()` を `async for` で回し、`RawResponsesStreamEvent.data` が
    `ResponseTextDeltaEvent` のときその `.delta` を `StreamTextDelta` として yield する。
    完了時に最終出力を `StreamTextDone` で 1 回 yield する。SDK の型（イベント / RunResult）は
    本関数内に閉じ、呼び出し側へは `_adapters` 内に閉じた中立 plain 型のみを渡す（NFR-1/NFR-5）。

    Args:
        agent: 実行する SDK Agent（解決済み）。
        input: ターンの入力（文字列 or input-list）。
        session: SDK `Session`（履歴保持）。None で履歴なし。
        context: 各実行へ素通しする共有 context。`RunContextWrapper` の場合は `.context`
            （生オブジェクト）を取り出して渡す（SDK が再ラップするため）。
        **runner_kwargs: `Runner.run_streamed` へ素通しする残りの kwarg（max_turns 等）。

    Yields:
        `StreamTextDelta`（テキスト断片）に続き、最後に `StreamTextDone`（最終出力）。
    """
    raw_context = context.context if isinstance(context, RunContextWrapper) else context
    streamed = Runner.run_streamed(
        agent, input, context=raw_context, session=session, **runner_kwargs
    )
    async for event in streamed.stream_events():
        if isinstance(event, RawResponsesStreamEvent) and isinstance(
            event.data, ResponseTextDeltaEvent
        ):
            yield StreamTextDelta(text=event.data.delta)
    final = streamed.final_output
    yield StreamTextDone(final_output="" if final is None else str(final))
