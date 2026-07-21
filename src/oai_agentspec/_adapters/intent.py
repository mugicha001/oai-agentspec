"""意図予測用 LLM 呼び出しのアダプタ（SDK 隔離窓口）。

runtime/intent/ の非 _adapters ファイルは agents を import せず、この薄いラッパを介して
SDK に触れる（judge.py と同型）。agents.Model を Any で受け、Agent + Runner.run で
文字列応答を得る最小関数のみを提供する。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


async def run_intent_prompt(
    model: Any,
    system: str,
    history_items: tuple[Mapping[str, Any], ...],
    user_content: str,
    *,
    context: Any = None,
) -> str:
    """agents.Agent + Runner.run を薄くラップして意図予測 LLM 応答を str で返す。

    Args:
        model: agents.Model 相当（呼び出し側 DI）。lib 内では Any として扱う。
        system: LLM に渡す system instructions。空文字は `instructions=None` として扱う。
        history_items: 過去 turn の SDK 互換 dict tuple。
        user_content: 現在発話の user content。空文字の場合は user turn を追加せず
            履歴のみを送る。
        context: RunContext。`RunContextWrapper` の場合は `.context` を展開して forward、
            None も可（keyword-only）。

    Returns:
        LLM の final_output を str 化したもの。None は空文字。

    Raises:
        ValueError: user_content と history_items の両方が空の場合（utterance が空でも
            prompt callable が非空を返せば送信は行われる点に注意）。
        Exception: モデル呼び出しで発生した例外はそのまま伝播する（catch しない）。
    """
    from agents import Agent, Runner  # 関数内遅延 import（NFR-1）

    from .run_context import unwrap_run_context

    agent = Agent(
        name="intent-classifier",
        instructions=system or None,
        model=model,
    )
    input_items: list[Mapping[str, Any]] = list(history_items)
    if user_content:
        input_items.append({"role": "user", "content": user_content})
    if not input_items:
        raise ValueError("intent classification requires a non-empty utterance or history items")
    raw_ctx = unwrap_run_context(context)
    result = await Runner.run(agent, input=input_items, context=raw_ctx)
    output = result.final_output
    return "" if output is None else str(output)
