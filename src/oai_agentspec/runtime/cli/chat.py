"""CLI 対話ループ（chat サブコマンド本体・rich 2層UI 統括）。

起動時にセッション選択画面（過去 session の rich Table）を表示し、新規会話 / 過去
セッションからの復元を選ばせる（2層ループ: 選択画面 -> 会話 -> /back で戻る）。会話は
エントリエージェント起点（サーバが起点を決める）で進み、CLI 側でエージェントを選ばせない。
会話画面のコマンドは /back（選択へ戻る）/quit（終了）/help。ストリーミング（既定）/
非ストリーミング両対応。表示は rich（Console / Panel / Table / Prompt）のみで行う
（prompt_toolkit は使わない）。入力はブロッキング Prompt を executor 上で待つ。

本モジュールは 2層ループの統括（`run_chat` / `_run_conversation` / `_start_conversation`）と
ターン実行（`_turn_streaming` / `_turn_non_streaming`）に加え、入力ヘルパ `_ainput` を
名前空間で解決して呼ぶ対話的ヘルパ（セッション選択 `_select_session`、承認の対話的収集
`_prompt_decision` / `_collect_decisions` / `_drain_pending_approvals`）を保持する。純表示
ヘルパは `_session_ui` / `_conversation_ui` / `_approval_ui`、入力は `_input` に分離する。
"""

from __future__ import annotations

import sys
import uuid
from typing import Any

from rich.console import Console
from rich.panel import Panel

from ._approval_ui import _show_pending_panel
from ._conversation_ui import (
    _print_assistant,
    _print_error,
    _render_history,
    _show_help,
)
from ._input import _ainput
from ._session_ui import _show_header, _show_menu, _show_session_table
from .client import (
    ApprovalRequired,
    ConversationClient,
    ConversationClientError,
    PendingApproval,
    SessionMeta,
    StreamDone,
    StreamToken,
)

# 終了コマンド（大文字小文字を無視して判定）。
_QUIT_COMMANDS = frozenset({"/quit", "/exit"})

# 復元時に表示する過去履歴の取得件数（直近 N 件）。
_HISTORY_LIMIT = 10


# ---------------------------------------------------------------------------
# セッション選択（外側ループ・入力を伴うため chat に残す）
# ---------------------------------------------------------------------------
async def _select_session(
    console: Console, sessions: list[SessionMeta]
) -> str | SessionMeta | None:
    """セッション選択入力を受け付ける。

    Returns:
        "new"（新規）/ "quit"（終了）/ 選択された `SessionMeta`。無効入力は None（再表示）。
    """
    raw = await _ainput(console, "[bold]選択[/bold]")
    if raw is None:
        return "quit"
    choice = raw.strip().lower()
    if choice in ("q", "quit", "exit"):
        return "quit"
    if choice in ("n", "new", ""):
        return "new"
    if choice.isdigit():
        idx = int(choice)
        if 1 <= idx <= len(sessions):
            return sessions[idx - 1]
    console.print("[bold red]無効な入力です[/bold red]\n")
    return None


# ---------------------------------------------------------------------------
# HITL 承認の対話的収集（入力を伴うため chat に残す）
# ---------------------------------------------------------------------------
async def _prompt_decision(console: Console, approval: PendingApproval) -> dict[str, Any]:
    """1 件の承認待ちについて approve/reject を対話的に選ばせ decision を組み立てる（FR-9）。

    承認（y）/ 却下（n）を選ばせ、却下なら任意で却下理由を入力させる。既定（Enter）・EOF /
    中断・認識できない入力はすべて安全側に倒して却下扱いとする（未承認なら実行されない・
    NFR-7）。`a`/`approve` / `r`/`reject` も別名として受理する。

    Args:
        console: 表示先 Console。
        approval: 対象の承認待ち（tool_name / call_id）。

    Returns:
        decisions の 1 要素（`{"call_id", "decision", "rejection_message"}`）。
    """
    raw = await _ainput(
        console,
        f"[bold yellow]{approval.tool_name}[/bold yellow] を承認しますか? "
        "[bold]\\[y/N][/bold]（Enter=却下）",
    )
    choice = (raw or "n").strip().lower()
    if choice in ("y", "yes", "a", "approve"):
        return {"call_id": approval.call_id, "decision": "approve", "rejection_message": None}
    if choice not in ("r", "reject", "n", "no"):
        # 認識できない入力は安全側（却下）に倒す旨を 1 行提示する（UX・NFR-7）。
        console.print("[dim]認識できない入力のため却下として扱います[/dim]")
    reason = await _ainput(console, "[dim]却下理由（任意・Enter でスキップ）[/dim]")
    rejection_message = (reason or "").strip() or None
    return {
        "call_id": approval.call_id,
        "decision": "reject",
        "rejection_message": rejection_message,
    }


async def _collect_decisions(
    console: Console, pending: list[PendingApproval]
) -> list[dict[str, Any]]:
    """承認待ち一覧を提示し call_id ごとに approve/reject を集めて decisions を返す（FR-9）。

    複数の承認待ちは call_id ごとに個別に選ばせる。返した decisions が再開後に部分解決で
    残った場合は、呼び出し側が再度本関数を通じて残りを選ばせる（段階解決・FR-7）。

    Args:
        console: 表示先 Console。
        pending: 承認待ち一覧（call_id 単位）。

    Returns:
        decisions（plain dict の列）。空なら継続しない。
    """
    _show_pending_panel(console, pending)
    decisions: list[dict[str, Any]] = []
    for approval in pending:
        decisions.append(await _prompt_decision(console, approval))
    return decisions


async def _drain_pending_approvals(
    client: ConversationClient,
    console: Console,
    agent_label: str,
    *,
    conversation_id: str,
) -> bool:
    """会話入口で未解決の承認待ちをドレイン（提示→解決→再開）する（P2-2・復元時の取りこぼし防止）。

    `get_approvals` で現在の承認待ちを確認し、あれば承認 UI（`_collect_decisions`）で解決して
    `resolve_approvals` で再開する。部分解決で残れば残りを繰り返す（段階解決・FR-7）。承認待ちが
    無ければ no-op（新規会話や中断なし復元では何もしない）。これを入力ループ前に呼ぶことで、復元
    会話の先頭メッセージが古い承認待ちで黙って捨てられる事故を防ぐ。

    Args:
        client: 接続済みクライアント。
        console: 表示先 Console。
        agent_label: 応答表示に使うエージェントラベル。
        conversation_id: 対象会話 ID。

    Returns:
        承認待ちを提示・解決した（=ユーザーへ何か出した）なら True、no-op なら False。

    Raises:
        ConversationClientError: get_approvals / resolve_approvals が失敗した場合。
    """
    pending = await client.get_approvals(conversation_id)
    if not pending:
        return False
    while pending:
        decisions = await _collect_decisions(console, pending)
        if not decisions:
            # 解決を選ばなかった（EOF 等）。未解決のまま中断する。
            return True
        result = await client.resolve_approvals(conversation_id, decisions)
        if result.status == "pending":
            pending = result.pending
            continue
        _print_assistant(console, agent_label, result.output or "")
        return True
    return True


# ---------------------------------------------------------------------------
# ターン実行
# ---------------------------------------------------------------------------
async def _turn_streaming(
    client: ConversationClient,
    console: Console,
    agent_label: str,
    text: str,
    *,
    conversation_id: str,
) -> None:
    """ストリーミングで 1 ターン会話し token を逐次表示する（承認待ち対応・FR-9）。

    承認待ちが来たら一覧を提示し call_id ごとに approve/reject を選ばせ、同一 WS 接続で
    decisions を送って再開し、残りの token/done を表示する（複数/段階解決対応）。
    """
    label_shown = False
    got_output = False

    async def _on_approval(pending: list[PendingApproval]) -> list[dict[str, Any]]:
        # 承認 UI を出す前に、ここまでの token 出力行を閉じる。
        nonlocal label_shown
        if label_shown:
            sys.stdout.write("\n")
            sys.stdout.flush()
            label_shown = False
        return await _collect_decisions(console, pending)

    def _ensure_label() -> None:
        nonlocal label_shown
        if not label_shown:
            console.print(f"[bold green]{agent_label}[/bold green]: ", end="")
            label_shown = True

    async for event in client.stream(
        None, text, conversation_id=conversation_id, approval_handler=_on_approval
    ):
        if isinstance(event, StreamToken):
            _ensure_label()
            sys.stdout.write(event.text)
            sys.stdout.flush()
            got_output = True
        elif isinstance(event, StreamDone):
            if not got_output:
                _ensure_label()
                sys.stdout.write(event.output)
                sys.stdout.flush()
            got_output = True
        elif isinstance(event, ApprovalRequired) and not event.pending:
            # 承認待ち通知だが対象が空（理論上は来ない）。何も提示しない。
            continue
    if label_shown:
        sys.stdout.write("\n")
        sys.stdout.flush()


async def _turn_non_streaming(
    client: ConversationClient,
    console: Console,
    agent_label: str,
    text: str,
    *,
    conversation_id: str,
) -> None:
    """非ストリーミングで 1 ターン会話し最終応答を表示する（承認待ち対応・FR-9）。

    承認待ちが返ったら一覧を提示し call_id ごとに approve/reject を選ばせ、`resolve_approvals`
    で再開する。再開後に部分解決で残ったら再度選ばせる（段階解決・FR-7）。全解決で最終応答を表示。
    """
    result = await client.send(None, text, conversation_id=conversation_id)
    while result.status == "pending":
        decisions = await _collect_decisions(console, result.pending)
        if not decisions:
            return
        result = await client.resolve_approvals(conversation_id, decisions)
    _print_assistant(console, agent_label, result.output or "")


async def _run_conversation(
    client: ConversationClient,
    console: Console,
    *,
    conversation_id: str,
    session_id: str,
    agent_label: str,
    history_items: list[dict[str, Any]],
    stream: bool,
) -> bool:
    """会話ループ（内側）を実行する。

    Returns:
        True でアプリ終了、False でセッション選択へ戻る。
    """
    header = f"[bold]会話[/bold] session: {session_id}\n[dim]エントリ: {agent_label}[/dim]"
    console.print(Panel(header, border_style="cyan", padding=(0, 2)))
    console.print("[dim]コマンド: /back=選択へ戻る  /quit=終了  /help=ヘルプ[/dim]\n")
    if history_items:
        _render_history(console, history_items, agent_label)

    # 入力ループ前に未解決の承認待ちをドレインする（復元会話の先頭入力が黙って捨てられるのを
    # 防ぐ・P2-2）。新規会話・中断なし復元では承認待ちが無く no-op。
    try:
        await _drain_pending_approvals(
            client, console, agent_label, conversation_id=conversation_id
        )
    except ConversationClientError as exc:
        _print_error(console, exc)

    while True:
        line = await _ainput(console, "[bold blue]You[/bold blue]")
        if line is None:
            return True
        text = line.strip()
        if text.startswith("/"):
            cmd = text.lower()
            if cmd == "/back":
                return False
            if cmd in _QUIT_COMMANDS:
                return True
            if cmd == "/help":
                _show_help(console)
                continue
            console.print(f"[bold red]不明なコマンド: {text}[/bold red]\n")
            continue
        if not text:
            continue
        try:
            if stream:
                await _turn_streaming(
                    client, console, agent_label, text, conversation_id=conversation_id
                )
            else:
                await _turn_non_streaming(
                    client, console, agent_label, text, conversation_id=conversation_id
                )
        except ConversationClientError as exc:
            _print_error(console, exc)


# ---------------------------------------------------------------------------
# エントリポイント（2層ループの統括）
# ---------------------------------------------------------------------------
async def run_chat(*, base_url: str, stream: bool) -> int:
    """chat の 2層ループ（セッション選択 -> 会話 -> /back で戻る）を実行する。

    Args:
        base_url: 会話サーバのベース URL。
        stream: True でストリーミング、False で非ストリーミング。

    Returns:
        プロセス終了コード（正常 0、接続/エントリ不能 1）。
    """
    console = Console()
    async with ConversationClient(base_url) as client:
        try:
            entry = await client.get_entry()
        except ConversationClientError as exc:
            _print_error(console, exc)
            return 1
        if entry is None:
            console.print(
                "[bold red]エントリエージェントが見つかりません。"
                "サーバの registry を確認してください。[/bold red]"
            )
            return 1

        while True:
            console.clear()
            _show_header(console, base_url, entry)
            try:
                sessions = await client.list_sessions()
            except ConversationClientError as exc:
                _print_error(console, exc)
                return 1
            _show_session_table(console, sessions)
            _show_menu(console, len(sessions))

            choice = await _select_session(console, sessions)
            if choice == "quit":
                console.print("[dim]終了します[/dim]")
                return 0
            if choice is None:
                continue

            try:
                conversation_id, session_id, history_items = await _start_conversation(
                    client, choice
                )
            except ConversationClientError as exc:
                _print_error(console, exc)
                await _ainput(console, "[dim]Enter で戻る[/dim]")
                continue

            should_quit = await _run_conversation(
                client,
                console,
                conversation_id=conversation_id,
                session_id=session_id,
                agent_label=entry,
                history_items=history_items,
                stream=stream,
            )
            if should_quit:
                console.print("[dim]終了します[/dim]")
                return 0


async def _start_conversation(
    client: ConversationClient, choice: str | SessionMeta
) -> tuple[str, str, list[dict[str, Any]]]:
    """選択（新規 / 復元）に応じて会話を作成し、履歴を取得する。

    新規は新しい session_id を採番して作成（サーバが永続化方針に従い保存）。復元は選択した
    session_id に紐づけて作成し、直近 N 件の履歴を取得して表示用に返す。

    Args:
        client: 接続済みクライアント。
        choice: "new" または復元対象の `SessionMeta`。

    Returns:
        `(conversation_id, session_id, history_items)`。

    Raises:
        ConversationClientError: 作成 / 履歴取得に失敗した場合。
    """
    if choice == "new":
        session_id = f"sess-{uuid.uuid4().hex[:12]}"
        conversation_id = await client.create_conversation(session_id=session_id)
        return conversation_id, session_id, []
    assert isinstance(choice, SessionMeta)
    session_id = choice.session_id
    conversation_id = await client.create_conversation(session_id=session_id)
    history_items = await client.get_history(session_id, limit=_HISTORY_LIMIT)
    return conversation_id, session_id, history_items


__all__ = ["run_chat"]
