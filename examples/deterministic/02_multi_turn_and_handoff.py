"""決定的応答モデルの多ターン例: `turn` による分岐とツール実行・ハンドオフ。

ステートレスな純関数方式では「N 回目の呼び出し」という概念を持たないため、ターンごとに違う
応答を返すには **入力から導出される値** で分岐する。`ModelRequest` はそのための判別
フィールドを持つ。

- `turn`: 入力に含まれるモデル応答の件数（初回 0）。**1 モデル応答 = 1 ターン**として数える。
- `tool_outputs`: 入力中の tool 実行結果アイテムの列。各アイテムは `call_id` / `output` /
  `type` のみを持ち tool 名は含まない（tool 名は `request.input` 側の `function_call`
  アイテムにしかない）。戻り値や `call_id` で分岐したいとき使う。

**`user_text` だけで分岐してはならない**。`user_text` は role が `user` の最新テキストであり、
tool 実行結果を受け取った次のターンでも変わらない。そのため `user_text` の部分一致だけで
ToolCall を返すルール関数は同じ ToolCall を返し続け、`max_turns` に達するまで回る。

`mixed_response` は 1 応答にアシスタントテキストとツール呼び出しの両方を載せる。「一言返して
から tool を呼ぶ」「発話しながらハンドオフする」を表現できる。ハンドオフは SDK 上では
`transfer_to_<エージェント名>` という関数呼び出しなので、ToolCall を返すだけで発火する。

モデル呼び出しは実 API へ接続しない（ネットワーク不要）。

実行:
    uv run python examples/deterministic/02_multi_turn_and_handoff.py
"""

from __future__ import annotations

import asyncio
from typing import Any

from agents import Agent, Runner, function_tool

from oai_agentspec.runtime.deterministic import (
    DeterministicResponseModel,
    ModelRequest,
    mixed_response,
    text_response,
    tool_call_response,
)


@function_tool
def add_one(x: int) -> int:
    """1 を足す。

    Args:
        x: 入力値。

    Returns:
        `x + 1`。
    """
    return x + 1


def tool_rule(request: ModelRequest) -> Any:
    """初回は tool を呼び、tool 結果を受けた次ターンで最終応答を返す。

    Args:
        request: 1 回分のモデル呼び出し入力。

    Returns:
        応答オブジェクト。
    """
    if request.turn == 0:
        return tool_call_response("add_one", '{"x": 41}')
    values = [getattr(item, "output", None) or item.get("output") for item in request.tool_outputs]
    return text_response(f"tool の結果は {values} でした")


def handoff_rule(request: ModelRequest) -> Any:
    """初回は一言添えてハンドオフし、遷移先で最終応答を返す。

    Args:
        request: 1 回分のモデル呼び出し入力。

    Returns:
        応答オブジェクト。
    """
    if request.turn == 0:
        # テキストと ToolCall を 1 応答に載せる（mixed_response）。
        # ハンドオフ tool 名は `transfer_to_<エージェント名>`。
        return mixed_response(
            "専門の担当へおつなぎします。",
            [("transfer_to_specialist", "{}", "call_handoff")],
        )
    who = (request.system_instructions or "?").strip()
    return text_response(f"[turn={request.turn}][{who}] 承りました")


async def main() -> None:
    """tool 実行とハンドオフの 2 パターンで多ターンの分岐を示す。"""
    print("--- tool 実行（turn 0 で ToolCall、turn 1 で結果を読む）")
    tool_model = DeterministicResponseModel(tool_rule)
    tool_agent = Agent(
        name="calculator", instructions="計算します", model=tool_model, tools=[add_one]
    )
    tool_result = await Runner.run(tool_agent, input="足して", max_turns=2)
    print(f"final_output = {tool_result.final_output!r}")

    print("--- ハンドオフ（mixed_response でテキストと ToolCall を同時に返す）")
    # 同一インスタンスを 2 体で共有する（ステートレスなので安全）
    handoff_model = DeterministicResponseModel(handoff_rule)
    specialist = Agent(name="specialist", instructions="専門担当です", model=handoff_model)
    reception = Agent(
        name="reception",
        instructions="受付です",
        model=handoff_model,
        handoffs=[specialist],
    )
    handoff_result = await Runner.run(reception, input="相談したい", max_turns=2)
    print(f"final_output = {handoff_result.final_output!r}")
    print(f"last_agent   = {handoff_result.last_agent.name!r}（ハンドオフが発火した）")


if __name__ == "__main__":
    asyncio.run(main())
