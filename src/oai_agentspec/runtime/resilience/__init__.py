"""Resilience 系宣言型の公開窓口（`oai-agentspec[resilience]` extra・agents 非依存の窓口）。

宣言型 `ModelRetryPolicy` / `RunBudgetPolicy` と例外 `RunBudgetExceeded`（すべて
`agents` に依存しない）、build 関数 2 種（`build_model_retry` / `build_run_budget_hooks`）、
および SDK 生型 10 種（`ModelRetrySettings` 系 / `RunErrorHandlers` 系）を再エクスポートする。

`_types` / `_errors` は追加の外部依存を持たない（`agents` / `pydantic` などを import しない）
ため**直 import** で `__all__` へ載せる。intent 窓口は pydantic 依存のため全シンボルを
PEP 562 で遅延化しているが、本窓口では依存を持たないシンボルは直 import としてよい
（差分理由）。`build_*` と SDK 生型 10 種は SDK への上向き参照を持つため、`__getattr__` で
`_adapters.resilience` 経由の**遅延取得**とし、窓口の import 自体は `agents` を発火させない
（NFR-1 の隔離を利用者側の import タイミングでも維持）。

窓口 import 自体は `resilience` extra 未導入でも壊れない（extra は追加依存ゼロ）。
"""

from __future__ import annotations

from typing import Any

from ._errors import RunBudgetExceeded
from ._types import ModelRetryPolicy, RunBudgetPolicy

__all__ = [
    "ModelRetryBackoffSettings",
    "ModelRetryNormalizedError",
    "ModelRetryPolicy",
    "ModelRetrySettings",
    "RetryDecision",
    "RetryPolicyContext",
    "RunBudgetExceeded",
    "RunBudgetPolicy",
    "RunErrorData",
    "RunErrorHandlerInput",
    "RunErrorHandlerResult",
    "RunErrorHandlers",
    "build_model_retry",
    "build_run_budget_hooks",
    "retry_policies",
]


# `_adapters.resilience` 経由で遅延取得するシンボル集合（build 関数 + SDK 生型 10 種）。
_DEFERRED_SYMBOLS = frozenset(
    {
        "ModelRetryBackoffSettings",
        "ModelRetryNormalizedError",
        "ModelRetrySettings",
        "RetryDecision",
        "RetryPolicyContext",
        "RunErrorData",
        "RunErrorHandlerInput",
        "RunErrorHandlerResult",
        "RunErrorHandlers",
        "build_model_retry",
        "build_run_budget_hooks",
        "retry_policies",
    }
)

# 直 import 済みシンボル（`_types` / `_errors` 由来・agents 非依存）。通常は module import
# 時に `globals()` に載っており `__getattr__` に来ないが、テスト等で `pop` された場合の
# 再解決のためのフォールバック。
_DIRECT_SYMBOLS = frozenset({"ModelRetryPolicy", "RunBudgetPolicy", "RunBudgetExceeded"})


def __getattr__(name: str) -> Any:
    """PEP 562: `_adapters.resilience` 経由でシンボルを遅延取得しキャッシュする。

    窓口 import 時に `agents` を発火させないため、SDK 生型と build 関数は本関数で初めて
    ロードする。取得済み値は `globals()` に載せて 2 回目以降を高速化する。

    Args:
        name: アクセスされた属性名。

    Returns:
        該当する公開シンボル（SDK 生型または build 関数）。

    Raises:
        AttributeError: `__all__` に含まれない属性名の場合。
    """
    if name in _DEFERRED_SYMBOLS:
        from ..._adapters import resilience as _resilience

        value = getattr(_resilience, name)
    elif name in _DIRECT_SYMBOLS:
        # 通常は module import 時に globals() に載っているため本 branch には来ない。
        # `pop` されたケース（テスト等）のフォールバックとして _types / _errors から再解決する。
        if name in {"ModelRetryPolicy", "RunBudgetPolicy"}:
            from . import _types

            value = getattr(_types, name)
        else:
            from . import _errors

            value = getattr(_errors, name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """`dir()` に遅延再エクスポート分を未 import でも含める。"""
    return sorted(set(globals()) | set(__all__))
