"""Runner.run の結果から「どういう経緯で回答に至ったか」を表示する共有ヘルパー。

注: SDK の tracing（spans / トレーシングダッシュボード）とは無関係。ここでは run の
実行経緯（run path）を `RunResult.new_items` から読み取って表示するだけのもの。

`RunResult.final_output` は最終文しか表さない。どのエージェントが喋り、どこでハンドオフや
ツール呼び出し（サブエージェント含む）が起きたかは `RunResult.new_items` に時系列で入っている。
本モジュールはそれを読みやすい 1 行ずつに整形する。examples 共通で使う。
"""

from __future__ import annotations

from typing import Any

from agents import ItemHelpers


def _short(text: str, limit: int = 80) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def print_run_path(result: Any) -> None:
    """run の経緯（メッセージ / ハンドオフ / ツール呼び出し）を時系列で表示する。

    Args:
        result: `Runner.run` の戻り値（RunResult）。
    """
    print("--- run path ---")
    for item in result.new_items:
        agent = getattr(getattr(item, "agent", None), "name", "?")
        if item.type == "handoff_output_item":
            print(f"  [handoff]     {item.source_agent.name} -> {item.target_agent.name}")
        elif item.type == "handoff_call_item":
            name = getattr(item.raw_item, "name", "")
            print(f"  [handoff_req] {agent}: {name}")
        elif item.type == "tool_call_item":
            name = getattr(item.raw_item, "name", "tool")
            print(f"  [tool_call]   {agent}: {name}")
        elif item.type == "tool_call_output_item":
            print(f"  [tool_out]    {agent}: {_short(str(item.output))}")
        elif item.type == "message_output_item":
            print(f"  [message]     {agent}: {_short(ItemHelpers.text_message_output(item))}")
        elif item.type == "reasoning_item":
            print(f"  [reasoning]   {agent}")
        else:
            print(f"  [{item.type}] {agent}")
    print(f"--- 最終回答エージェント: {result.last_agent.name} ---")
