"""決定的応答モデルのストリーミング例: `Runner.run_streamed` を実 API 抜きで回す。

`DeterministicResponseModel` は `stream_response` を実装しており、`Runner.run_streamed` でも
完走する。方式は既存の `WorkflowModel` と同じ **post-execution streaming** である。

- ルール関数が返した完成済みの応答を確定させてから、テキストを一定長で区切って差分イベント
  として流し、最後に終端イベントを流す。
- したがって **実 LLM の逐次生成とは異なり、進捗を表す情報量は無い**。UI の逐次表示が動くこと
  の確認や、ストリーミング経路を含む結線の検証に使う。
- ツール呼び出しのみの応答ではテキスト差分が流れず、終端イベントのみになる。

モデル呼び出しは実 API へ接続しない（ネットワーク不要）。

実行:
    uv run python examples/deterministic/03_streaming.py
"""

from __future__ import annotations

import asyncio
from typing import Any

from agents import Agent, Runner
from openai.types.responses import ResponseTextDeltaEvent

from oai_agentspec.runtime.deterministic import (
    DeterministicResponseModel,
    ModelRequest,
    text_response,
)


def rule(request: ModelRequest) -> Any:
    """入力をそのまま返すルール関数。

    Args:
        request: 1 回分のモデル呼び出し入力。

    Returns:
        テキスト応答。
    """
    return text_response(f"ストリーミング応答: {request.user_text}")


async def main() -> None:
    """`run_streamed` で差分イベントを受け取り、連結が最終出力と一致することを示す。"""
    model = DeterministicResponseModel(rule)
    agent = Agent(name="streamer", instructions="ストリーミングします", model=model)

    result = Runner.run_streamed(agent, input="こんにちは")

    deltas: list[str] = []
    async for event in result.stream_events():
        if event.type != "raw_response_event":
            continue
        if isinstance(event.data, ResponseTextDeltaEvent):
            deltas.append(event.data.delta)

    print(f"差分イベント数 = {len(deltas)}")
    print(f"差分の連結     = {''.join(deltas)!r}")
    print(f"final_output   = {result.final_output!r}")
    print(f"連結 == final_output: {''.join(deltas) == result.final_output}")
    print()
    print("注: これは post-execution streaming（応答を確定させてから区切って流す）であり、")
    print("    実 LLM の逐次生成のような進捗情報は持たない。")


if __name__ == "__main__":
    asyncio.run(main())
