"""CLI 会話画面の表示ヘルパとテキスト抽出（chat サブコマンド用・rich UI）。

会話画面（内側ループ）の純表示ヘルパ（`_render_history` / `_print_error` / `_print_assistant` /
`_show_help`）と履歴 content のテキスト抽出（`_extract_text`）を提供する。入力を伴う会話ループ
本体は `chat` 本体に残る。rich は cli extra のため、本モジュールは `chat` 経由でのみ import
される（`cli/__init__` や `main` のトップレベルには載らない）。SDK（`agents`）は import しない。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.panel import Panel

if TYPE_CHECKING:
    from ._models import ConversationClientError

__all__ = [
    "_extract_text",
    "_print_assistant",
    "_print_error",
    "_render_history",
    "_show_help",
]


def _extract_text(content: Any) -> str:
    """履歴アイテムの content フィールドをテキスト文字列へ変換する。

    SDK の content は `str` または `[{"text": "..."}]` 形式の `list[dict]` を取り得る。
    両者を吸収して連結した文字列を返す。

    Args:
        content: 履歴アイテムの content（str / list / その他）。

    Returns:
        テキスト化した文字列。
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(content)


def _render_history(console: Console, items: list[dict[str, Any]], agent_label: str) -> None:
    """復元した過去履歴（直近 N 件）を Panel で表示する。"""
    lines: list[str] = []
    for entry in items:
        role = entry.get("role")
        if role == "user":
            text = _extract_text(entry.get("content", ""))
            lines.append(f"  [bold blue]You:[/bold blue] {text}")
        elif role == "assistant":
            text = _extract_text(entry.get("content", ""))
            if text:
                lines.append(f"  [green]{agent_label}:[/green] {text}")
        elif entry.get("type") == "function_call":
            name = str(entry.get("name", ""))
            if name.startswith("transfer_to_"):
                target = name.replace("transfer_to_", "").replace("_", " ")
                lines.append(f"  [dim italic]-> {target} にハンドオフ[/dim italic]")
    if not lines:
        return
    console.print(
        Panel(
            "\n".join(lines),
            title=f"過去の会話履歴（直近 {len(items)} 件）",
            border_style="dim",
            padding=(1, 1),
        )
    )
    console.print()


def _print_error(console: Console, exc: ConversationClientError) -> None:
    """会話クライアントエラーを赤 Panel で表示する。"""
    title = f"[bold red]{exc.code}[/bold red]" if exc.code else "[bold red]エラー[/bold red]"
    console.print(Panel(exc.message, title=title, border_style="red", padding=(0, 1)))
    console.print()


def _print_assistant(console: Console, agent_label: str, output: str) -> None:
    """assistant の最終応答を表示する。"""
    console.print(f"[bold green]{agent_label}[/bold green]: ", end="")
    console.print(output, markup=False)


def _show_help(console: Console) -> None:
    """会話画面のコマンドヘルプを表示する。"""
    console.print(
        Panel(
            "  [bold]/back[/bold]   セッション選択へ戻る\n"
            "  [bold]/quit[/bold]   終了\n"
            "  [bold]/help[/bold]   このヘルプを表示",
            title="コマンド一覧",
            border_style="cyan",
            padding=(0, 1),
        )
    )
    console.print()
