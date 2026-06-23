"""CLI HITL 承認 UI の表示ヘルパ（chat サブコマンド用・rich UI）。

承認待ち一覧の純表示ヘルパ（`_show_pending_panel`）を提供する。入力を伴う承認の対話的収集
（`_prompt_decision` / `_collect_decisions` / `_drain_pending_approvals`）は、入力ヘルパ
`_ainput` を `chat` 名前空間で解決する必要があるため `chat` 本体に残る。rich は cli extra の
ため、本モジュールは `chat` 経由でのみ import される（`cli/__init__` や `main` のトップレベルには
載らない）。SDK（`agents`）は import しない。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel

if TYPE_CHECKING:
    from ._models import PendingApproval

__all__ = ["_show_pending_panel"]


def _show_pending_panel(console: Console, pending: list[PendingApproval]) -> None:
    """承認待ち一覧を黄色 Panel で提示する（tool_name / call_id・FR-9）。"""
    lines = [
        f"  [bold]{index}.[/bold] ツール [bold yellow]{p.tool_name}[/bold yellow] "
        f"[dim](call_id: {p.call_id})[/dim]"
        for index, p in enumerate(pending, start=1)
    ]
    console.print(
        Panel(
            "ツール実行の承認が必要です:\n" + "\n".join(lines),
            title="[bold yellow]承認待ち[/bold yellow]",
            border_style="yellow",
            padding=(1, 1),
        )
    )
