"""宣言 spec の共有バリデーションヘルパ（agents 非依存・最下層）。

通常ルート（`AgentRegistry`）と Realtime 専用ルート（`RealtimeAgentRegistry` /
`_adapters/realtime.py`）が共有する検証ロジックを一元化する。両ルートは宣言型・registry を
共用しない設計だが、spec レベルの不変条件（callable instructions の呼び出し可能性等）は
単一の実装・単一のエラーメッセージで維持する（片側だけの修正による挙動乖離を防ぐ）。
"""

from __future__ import annotations

import inspect
from typing import Any


def validate_instructions_callable(agent_name: str, instructions: Any) -> None:
    """callable instructions が (context, agent) の 2 引数で呼び出せることを検証する。

    SDK は instructions を `(context, agent)` の 2 位置引数で呼び出すため、bind による
    呼び出し可能性のみを検証する（デフォルト引数・可変長は許容）。シグネチャ取得不能な
    callable（builtin 等）は検証をスキップし実行時に委ねる。

    Args:
        agent_name: エラーメッセージに含めるエージェント名。
        instructions: spec の instructions 値（callable でなければ何もしない）。

    Raises:
        ValueError: callable が 2 引数で呼び出せない場合。
    """
    if not callable(instructions):
        return
    try:
        sig = inspect.signature(instructions)
    except (ValueError, TypeError):
        # シグネチャ取得不能な callable（builtin 等）は検証をスキップし実行時に委ねる
        return
    try:
        sig.bind(object(), object())
    except TypeError:
        raise ValueError(
            f"agent {agent_name!r}: instructions callable は (context, agent) の "
            f"2 引数で呼び出せる必要があります"
        ) from None


def ensure_static_prompt(agent_name: str, prompt: Any) -> None:
    """prompt が callable（DynamicPromptFunction）でないことを検証する（Realtime ルート用）。

    RealtimeAgent は `Prompt | None` のみ対応で callable prompt を解決しないため、
    register 時（前倒し検証）と build 時（第二防御）の両層が本ヘルパを共有し、
    同一の判定・同一のエラーメッセージを維持する。

    Args:
        agent_name: エラーメッセージに含めるエージェント名。
        prompt: spec の prompt 値。

    Raises:
        ValueError: prompt が callable の場合。
    """
    if callable(prompt):
        raise ValueError(
            f"agent {agent_name!r}: prompt に callable（DynamicPromptFunction）は"
            f"指定できません（RealtimeAgent は静的 Prompt のみ対応）"
        )
