"""SDK function_tool 結線: `ToolSpec` → `agents.function_tool()` の kwargs 変換窓口。

`ToolRegistry` の `__getattr__` から関数内遅延 import 経由で呼ばれる（NFR-1 の
SDK 隔離を維持）。責務は以下:

- `ToolSpec` の型付きフィールドを SDK kwargs へ None-omission で流し込む
- `extra` 素通し dict を `validate_extra_kwargs`（subject_label="tool"）で検証
- `failure_error_function` の 3 値センチネル（`_UNSET` / callable / `None`）を判定し、
  未指定なら kwarg を渡さない（SDK 既定 formatter に委ねる）
- `is_enabled` に enabled_supplier を SDK 契約 `(ctx, agent) -> bool` にラップした
  closure を注入（bool 焼き込み禁止・FR-4）
- `function_tool(spec.func, **kwargs)` を呼んで `FunctionTool` を返す
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from agents import FunctionTool, function_tool

from .._validation import validate_extra_kwargs
from ..constants import TOOL_UNSET

if TYPE_CHECKING:
    # 型ヒント専用の一方向参照（実行時 import なし・W1 修正）。
    # `_adapters -> tool_registry` の実行時上向き参照を持たないことで、
    # 将来 tool_registry 側に `_adapters` のトップレベル import が入っても
    # 循環にならない構造を維持する。
    from ..tool_registry import ToolSpec

__all__ = ["build_function_tool"]


# 予約キー = 型付きフィールドの SDK 対応 kwarg 9 つ + `func`（10 個・設計判断 6）。
# extra に同名キーを積むと build 時に ValueError で弾く（silent override 防止）。
_DEDICATED_TOOL_KWARGS: frozenset[str] = frozenset(
    {
        "func",
        "is_enabled",
        "needs_approval",
        "timeout",
        "timeout_behavior",
        "timeout_error_function",
        "failure_error_function",
        "name_override",
        "description_override",
        "strict_mode",
    }
)

# 有効 kwarg 集合は `inspect.signature(function_tool)` から導出（builders.py L91 の
# dataclasses.fields 導出と同思想。SDK 引数追加に自動追随・設計判断 6）。
_FUNCTION_TOOL_PARAM_NAMES: frozenset[str] = frozenset(
    inspect.signature(function_tool).parameters.keys()
)


def build_function_tool(
    spec: ToolSpec,
    enabled_supplier: Callable[[], bool],
) -> FunctionTool:
    """`ToolSpec` を SDK `FunctionTool` へ結線する（Registry の遅延構築 hook）。

    Args:
        spec: 登録された Tool 宣言（`ToolRegistry._specs[name]`）。
        enabled_supplier: `spec.enabled` の現在値を返す 0 引数 callable
            （`ToolRegistry.__getattr__` から `lambda: spec.enabled` の形で渡される）。
            SDK `is_enabled(ctx, agent)` シグネチャへ本関数内でラップされる。

    Returns:
        メタデータ適用済み `FunctionTool`（SDK 実型）。

    Raises:
        ValueError: `spec.extra` に予約キー衝突または SDK function_tool が受け付けない
            未知キーが含まれる場合。
    """
    tool_name = spec.name if spec.name is not None else spec.func.__name__

    # extra 検証（subject_label="tool" で "tool 'xxx':" prefix メッセージを出す）
    validate_extra_kwargs(
        tool_name,
        spec.extra,
        dedicated=_DEDICATED_TOOL_KWARGS,
        field_names=_FUNCTION_TOOL_PARAM_NAMES,
        agent_label="agents.function_tool",
        subject_label="tool",
    )

    # None-omission で kwargs 組み立て（SDK 既定値を再現しない・設計判断 1）
    kwargs: dict[str, Any] = {}

    # name_override: spec.name_override 明示指定を最優先。次に Registry 登録キー
    # （spec.name が func.__name__ と異なる場合に SDK 提示名も登録キーに合わせる）。
    if spec.name_override is not None:
        kwargs["name_override"] = spec.name_override
    elif spec.name is not None:
        kwargs["name_override"] = spec.name

    if spec.description_override is not None:
        kwargs["description_override"] = spec.description_override

    if spec.needs_approval is not None:
        kwargs["needs_approval"] = spec.needs_approval

    if spec.timeout is not None:
        kwargs["timeout"] = spec.timeout
    if spec.timeout_behavior is not None:
        kwargs["timeout_behavior"] = spec.timeout_behavior
    if spec.timeout_error_function is not None:
        kwargs["timeout_error_function"] = spec.timeout_error_function

    if spec.strict_mode is not None:
        kwargs["strict_mode"] = spec.strict_mode

    # failure_error_function の 3 値: `TOOL_UNSET` なら渡さない（SDK 既定 formatter に委ねる）、
    # それ以外（None 明示 / callable）は渡す。SDK 側で None 明示は「例外を素通し」の意味に
    # なるため 3 値の区別が保持される（設計判断 4）。
    if spec.failure_error_function is not TOOL_UNSET:
        kwargs["failure_error_function"] = spec.failure_error_function

    # is_enabled: enabled_supplier を SDK シグネチャ (ctx, agent) -> bool にラップ。
    # spec.enabled は毎回 attribute lookup されるため FR-4 の動的トグルが成立する。
    def _is_enabled(_ctx: Any, _agent: Any) -> bool:
        return bool(enabled_supplier())

    kwargs["is_enabled"] = _is_enabled

    # extra 素通し（予約キー衝突は既に validate_extra_kwargs で検証済み）
    kwargs.update(spec.extra)

    return function_tool(spec.func, **kwargs)
