"""CLI 非同期入力ヘルパ（chat サブコマンド用・rich Prompt を executor で待つ）。

ブロッキングな rich `Prompt.ask` をイベントループを塞がずに待つ `_ainput` を提供する。rich は
cli extra のため、本モジュールは cli extra 導入時のみ import できる（`chat` 経由でのみ import
され、`cli/__init__` や `main` のトップレベルには載らない）。SDK（`agents`）は import しない。
"""

from __future__ import annotations

import asyncio

from rich.console import Console
from rich.prompt import Prompt

__all__ = ["_ainput"]


async def _ainput(console: Console, prompt_markup: str) -> str | None:
    """rich の Prompt をイベントループを塞がずに待つ。

    Args:
        console: 表示先 Console。
        prompt_markup: プロンプト表示（rich マークアップ可）。

    Returns:
        入力文字列。EOF / 中断（Ctrl-C / Ctrl-D）では None。
    """
    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(None, lambda: Prompt.ask(prompt_markup, console=console))
    except (EOFError, KeyboardInterrupt):
        return None
