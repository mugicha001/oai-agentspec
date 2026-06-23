"""CLI セッション選択画面の表示ヘルパ（chat サブコマンド用・rich UI）。

セッション選択画面（外側ループ）の純表示ヘルパ（`_show_header` / `_show_session_table` /
`_show_menu`）を提供する。入力を伴うセッション選択（`_select_session`）は `chat` 本体に残る。
rich は cli extra のため、本モジュールは `chat` 経由でのみ import される（`cli/__init__` や
`main` のトップレベルには載らない）。SDK（`agents`）は import しない。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

if TYPE_CHECKING:
    from ._models import SessionMeta

__all__ = [
    "_show_header",
    "_show_menu",
    "_show_session_table",
]


def _show_header(console: Console, base_url: str, entry: str | None) -> None:
    """ヘッダーパネルを表示する（サーバ URL とエントリエージェント）。"""
    entry_line = f"\nエントリ: {entry}" if entry else ""
    console.print(
        Panel(
            f"[bold]oai-agentspec 会話 CLI[/bold]\nサーバー: {base_url}{entry_line}",
            border_style="cyan",
            padding=(0, 2),
        )
    )
    console.print()


def _show_session_table(console: Console, sessions: list[SessionMeta]) -> None:
    """過去セッション一覧を rich Table で表示する。"""
    if not sessions:
        console.print("  [dim]過去のセッションはありません[/dim]\n")
        return
    table = Table(title="セッション一覧", border_style="blue", show_lines=False, pad_edge=True)
    table.add_column("#", style="bold", width=4, justify="right")
    table.add_column("セッション ID", min_width=16)
    table.add_column("最終更新", min_width=16)
    table.add_column("ターン", justify="right", width=6)
    table.add_column("プレビュー", min_width=20)
    for index, meta in enumerate(sessions, start=1):
        table.add_row(
            str(index),
            meta.session_id,
            meta.updated_at or "—",
            str(meta.turn_count),
            meta.preview or "[dim](なし)[/dim]",
        )
    console.print(table)
    console.print()


def _show_menu(console: Console, session_count: int) -> None:
    """セッション選択のコマンドヒントを表示する。"""
    if session_count > 0:
        hint = (
            f"  [bold]n[/bold] 新規会話  [bold]1-{session_count}[/bold] セッション復元  "
            "[bold]q[/bold] 終了"
        )
    else:
        hint = "  [bold]n[/bold] 新規会話  [bold]q[/bold] 終了"
    console.print(hint)
    console.print()
