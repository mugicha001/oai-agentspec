"""宣言 spec の共有バリデーションヘルパ（agents 非依存・最下層）。

通常ルート（`AgentRegistry`）と Realtime 専用ルート（`RealtimeAgentRegistry` /
`_adapters/realtime.py`）が共有する検証ロジックを一元化する。両ルートは宣言型・registry を
共用しない設計だが、spec レベルの不変条件（callable instructions の呼び出し可能性等）は
単一の実装・単一のエラーメッセージで維持する（片側だけの修正による挙動乖離を防ぐ）。
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    # 型ヒント専用の一方向参照であり実行時依存はない（_validation は最下層のまま）。
    from .realtime.spec import RealtimeHandoffConfig


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


def validate_extra_kwargs(
    agent_name: str,
    extra: Mapping[str, Any],
    *,
    dedicated: frozenset[str],
    field_names: frozenset[str],
    agent_label: str,
) -> None:
    """spec.extra の専用フィールド衝突・未知キーを検証する（両ルートのアダプタが共有）。

    通常ルート（`build_agent`）と Realtime 専用ルート（`build_realtime_agent`）の
    extra 検証を一元化する。SDK 型の有効 kwarg 集合（`field_names`）と専用フィールド集合
    （`dedicated`）は各アダプタで算出済みの `frozenset` を渡すため、本ヘルパは agents 非依存を
    保つ。衝突検査を未知検査より先に行う順序・両メッセージ文字列を単一ソースで維持する。

    Args:
        agent_name: エラーメッセージに含めるエージェント名。
        extra: spec の extra（キー集合のみ参照する）。
        dedicated: spec 側で別扱いする専用フィールド名の集合（extra から除外する対象）。
        field_names: 対象 SDK Agent が受け付ける有効 kwarg 名の集合。
        agent_label: 未知キーメッセージに埋め込む SDK クラス表示名
            （`agents.Agent` / `agents.realtime.RealtimeAgent`）。

    Raises:
        ValueError: extra に専用フィールド名と同名のキー、または対象 Agent が受け付けない
            未知のキーが含まれる場合。
    """
    collisions = dedicated & extra.keys()
    if collisions:
        raise ValueError(
            f"agent {agent_name!r}: extra に専用フィールドと同名のキーが含まれます: "
            f"{sorted(collisions)}"
        )
    unknown = extra.keys() - field_names
    if unknown:
        raise ValueError(
            f"agent {agent_name!r}: extra に {agent_label} が受け付けないキーが含まれます: "
            f"{sorted(unknown)}"
        )


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


def validate_realtime_handoff_options(
    agent_name: str,
    handoffs: Sequence[str],
    handoff_options: Mapping[str, RealtimeHandoffConfig],
) -> None:
    """Realtime ルートの per-edge ハンドオフ設定（`handoff_options`）を検証する。

    `RealtimeAgentRegistry.register`（`_validate_spec` 経由）と `RealtimeHandoffGraph.apply`
    が共有する検証ロジックを一元化する。両者が同じバリデータを呼ぶことで、`apply -> register`
    でも `register -> apply` でも最終 spec が必ず同一規則・同一エラーメッセージで検証される
    （順序非依存）。

    検証項目:
        - `handoff_options` のキーが `handoffs` に存在すること（タイポによる per-edge 設定の
          silent drop を防ぐ）。
        - `input_type` 指定時に `on_handoff` が伴うこと。
        - `on_handoff` の引数個数（`input_type` ありで 2・なしで 1。SDK `realtime_handoff()`
          の必須制約）。get() 時の文脈なし UserError をエージェント名・エッジ名入り ValueError
          に前倒しする。シグネチャ取得不能な callable（builtin 等）は arity 検査をスキップする。

    Args:
        agent_name: エラーメッセージに含めるエージェント名（ハンドオフ元）。
        handoffs: ハンドオフ先エージェント名リスト。
        handoff_options: dst 名 -> RealtimeHandoffConfig の per-edge 設定。

    Raises:
        ValueError: キー不整合・input_type と on_handoff の不整合・arity 不一致のいずれか。
    """
    unknown_options = set(handoff_options) - set(handoffs)
    if unknown_options:
        raise ValueError(
            f"agent {agent_name!r}: handoff_options のキー {sorted(unknown_options)!r} が "
            f"handoffs に存在しません"
        )
    for dst, config in handoff_options.items():
        if config.input_type is not None and config.on_handoff is None:
            raise ValueError(
                f"agent {agent_name!r} -> {dst!r}: input_type を指定する場合は "
                f"on_handoff が必須です"
            )
        if config.on_handoff is not None:
            # SDK realtime_handoff() は引数個数を厳格に検査する（bind ではなく len）ため
            # 同じ規則で前倒し検証する
            expected = 2 if config.input_type is not None else 1
            try:
                n_params = len(inspect.signature(config.on_handoff).parameters)
            except (ValueError, TypeError):
                continue
            if n_params != expected:
                form = "(context, input) の 2" if expected == 2 else "(context) の 1"
                raise ValueError(
                    f"agent {agent_name!r} -> {dst!r}: on_handoff は {form} 引数である"
                    f"必要がありますが {n_params} 引数です"
                )
