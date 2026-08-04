"""決定的応答モデルの落とし穴例: 1 つのルール関数で tool 実行とハンドオフを併用する。

`02_multi_turn_and_handoff.py` は tool 用とハンドオフ用でルール関数を分けているが、実運用では
1 つのルール関数が両方を扱うことが多い。そのとき踏みやすい落とし穴がある。

**ハンドオフも SDK 上は `transfer_to_<エージェント名>` という関数呼び出しであり、その結果も
`ModelRequest.tool_outputs` に載る。** そのため次のように無条件で分岐すると、ハンドオフ後の
応答まで tool 分岐が乗っ取る。

    if request.tool_outputs:            # <- ハンドオフの結果にも当たってしまう
        return text_response(...)

正しくは、呼んだ tool を **`call_id` で絞り込む**。応答ビルダは `call_id` を指定できるので、
自分が発行した ToolCall だけを識別できる。`ModelRequest.tool_outputs` の
`function_call_output` アイテムは `call_id` / `output` / `type`（+ `id` / `status`）のみを
持ち、tool 名フィールドは無い。tool 名で絞りたい場合は 2 段階の手順が要る:
`request.input` を走査して `type == "function_call"` のアイテムから `name` -> `call_id` の
対応を作り、その `call_id` で `tool_outputs` を絞り込む。

本例は「乗っ取られる版」と「絞り込む版」を同じ入力で実行し、出力の差を示す。

モデル呼び出しは実 API へ接続しない（ネットワーク不要）。

実行:
    uv run python examples/deterministic/04_tool_and_handoff_in_one_rule.py
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

#: 本ルール関数が発行する ToolCall の call_id（自分の呼び出しを識別するために固定する）。
ECHO_CALL_ID = "call_echo_1"


@function_tool
def echo_tool(value: str) -> str:
    """受け取った値をそのまま返す。

    Args:
        value: 任意の文字列。

    Returns:
        `echoed:<value>`。
    """
    return f"echoed:{value}"


def _outputs_of(request: ModelRequest) -> list[Any]:
    """`tool_outputs` から output 値を取り出す（dict / 属性の双方に対応）。

    Args:
        request: 1 回分のモデル呼び出し入力。

    Returns:
        output 値のリスト。
    """
    return [(getattr(item, "output", None) or item.get("output")) for item in request.tool_outputs]


def naive_rule(request: ModelRequest) -> Any:
    """落とし穴版: `tool_outputs` があるかどうかだけで分岐する。

    ハンドオフの結果も `tool_outputs` に載るため、ハンドオフ後の応答がこの分岐に
    乗っ取られる。

    Args:
        request: 1 回分のモデル呼び出し入力。

    Returns:
        応答オブジェクト。
    """
    if request.tool_outputs:
        return text_response(f"[tool 分岐] {_outputs_of(request)}")
    if request.turn == 0 and "ツール" in request.user_text:
        return tool_call_response("echo_tool", '{"value": "x"}', call_id=ECHO_CALL_ID)
    if request.turn == 0:
        return mixed_response(
            "担当へおつなぎします。", [("transfer_to_specialist", "{}", "call_h")]
        )
    who = (request.system_instructions or "?").strip()
    return text_response(f"[{who}] 応答しました")


def _call_id_of(item: Any) -> str:
    """`tool_outputs` のアイテムから call_id を取り出す（dict / 属性の双方に対応）。

    Args:
        item: tool 実行結果アイテム。

    Returns:
        call_id（取得できなければ空文字）。
    """
    return str(
        getattr(item, "call_id", None)
        or (item.get("call_id") if isinstance(item, dict) else "")
        or ""
    )


def filtered_rule(request: ModelRequest) -> Any:
    """正しい版: 自分が発行した `call_id` の結果だけを tool 分岐として扱う。

    Args:
        request: 1 回分のモデル呼び出し入力。

    Returns:
        応答オブジェクト。
    """
    echoed = [
        (getattr(item, "output", None) or item.get("output"))
        for item in request.tool_outputs
        if _call_id_of(item) == ECHO_CALL_ID
    ]
    if echoed:
        return text_response(f"[tool 分岐] {echoed}")
    if request.turn == 0 and "ツール" in request.user_text:
        return tool_call_response("echo_tool", '{"value": "x"}', call_id=ECHO_CALL_ID)
    if request.turn == 0:
        return mixed_response(
            "担当へおつなぎします。", [("transfer_to_specialist", "{}", "call_h")]
        )
    who = (request.system_instructions or "?").strip()
    return text_response(f"[{who}] 応答しました")


async def _run_both(rule: Any, label: str) -> None:
    """同じルール関数で tool 経路とハンドオフ経路を実行して出力を表示する。

    Args:
        rule: 検証するルール関数。
        label: 表示用のラベル。
    """
    model = DeterministicResponseModel(rule)
    specialist = Agent(name="specialist", instructions="専門担当です", model=model)
    reception = Agent(
        name="reception",
        instructions="受付です",
        model=model,
        tools=[echo_tool],
        handoffs=[specialist],
    )

    tool_result = await Runner.run(reception, input="ツールを使って", max_turns=3)
    handoff_result = await Runner.run(reception, input="相談したい", max_turns=3)

    print(f"--- {label}")
    print(f"tool 経路     = {tool_result.final_output!r}")
    print(f"ハンドオフ経路 = {handoff_result.final_output!r}")
    print(f"last_agent    = {handoff_result.last_agent.name!r}")


async def main() -> None:
    """落とし穴版と絞り込み版を同じ入力で比較する。"""
    await _run_both(naive_rule, "落とし穴版（tool_outputs の有無だけで分岐）")
    print("  ハンドオフ経路の応答が tool 分岐に乗っ取られている")
    print("  （transfer_to_specialist の結果も tool_outputs に載るため）")
    print()
    await _run_both(filtered_rule, "絞り込み版（自分が発行した call_id だけを見る）")
    print("  ハンドオフ経路が専門担当の応答になっている")


if __name__ == "__main__":
    asyncio.run(main())
