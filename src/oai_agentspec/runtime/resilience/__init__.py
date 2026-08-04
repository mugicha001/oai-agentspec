"""Resilience 系宣言型の公開窓口（`oai-agentspec[resilience]` extra・agents 非依存の窓口）。

宣言型 `ModelRetryPolicy` / `RunBudgetPolicy` / `FailsafeHandler` / `FailsafePolicy` /
`FailsafeResult`、sentinel `RUNNING_AGENT`、関数 `failsafe_call`（いずれも `agents` に
依存しない）、build 関数 2 種（`build_model_retry` / `build_run_budget_hooks`）、および
SDK 生型 10 種（`ModelRetrySettings` 系 / `RunErrorHandlers` 系）を再エクスポートする。

例外 `RunBudgetExceeded` は本窓口からは撤去済み（Breaking Change）。正規の取得経路は
`oai_agentspec.exceptions`（lib 独自例外 9 種の統一窓口）を参照する。

`_types` / `_failsafe` は追加の外部依存を持たない（`agents` / `pydantic` などを
import しない）ため**直 import** で `__all__` へ載せる。intent 窓口は pydantic 依存の
ため全シンボルを PEP 562 で遅延化しているが、本窓口では依存を持たないシンボルは直
import としてよい（差分理由）。`build_*` と SDK 生型 10 種は SDK への上向き参照を持つ
ため、`__getattr__` で `_adapters.resilience` 経由の**遅延取得**とし、窓口 import 時点では
実装実体の `_adapters.resilience` をロードしない（`hooks` 窓口と同型の遅延パターン）。
`agents` 自体はコア依存であり、`oai_agentspec/__init__.py` -> `_adapters/__init__.py` の
連鎖で本窓口の import より前にロード済みになる。遅延の対象は実装実体のモジュールで
あって SDK ではない。

`_DIRECT_SYMBOLS` は直 import 済みシンボルの再解決フォールバック用にシンボル名から
所属モジュール名（`_types` / `_failsafe`）への対応を dict で持つ（複数モジュールに
分散するため）。`_DEFERRED_SYMBOLS` は取得元が `_adapters.resilience` 単一のため
frozenset のままでよく、両者は非対称。

窓口 import 自体は `resilience` extra 未導入でも壊れない（extra は追加依存ゼロ）。
"""

from __future__ import annotations

import importlib
from typing import Any

from ._failsafe import RUNNING_AGENT, FailsafeHandler, FailsafePolicy, FailsafeResult, failsafe_call
from ._types import ModelRetryPolicy, RunBudgetPolicy

__all__ = [
    "FailsafeHandler",
    "FailsafePolicy",
    "FailsafeResult",
    "ModelRetryBackoffSettings",
    "ModelRetryNormalizedError",
    "ModelRetryPolicy",
    "ModelRetrySettings",
    "RUNNING_AGENT",
    "RetryDecision",
    "RetryPolicyContext",
    "RunBudgetPolicy",
    "RunErrorData",
    "RunErrorHandlerInput",
    "RunErrorHandlerResult",
    "RunErrorHandlers",
    "build_model_retry",
    "build_run_budget_hooks",
    "failsafe_call",
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

# 直 import 済みシンボル名からその所属モジュール名（`_types` / `_failsafe`）への対応。
# 通常は module import 時に `globals()` に載っており `__getattr__` に来ないが、
# テスト等で `pop` された場合の再解決のためのフォールバック。
_DIRECT_SYMBOLS: dict[str, str] = {
    "ModelRetryPolicy": "_types",
    "RunBudgetPolicy": "_types",
    "FailsafeHandler": "_failsafe",
    "FailsafePolicy": "_failsafe",
    "FailsafeResult": "_failsafe",
    "failsafe_call": "_failsafe",
    "RUNNING_AGENT": "_failsafe",
}


def __getattr__(name: str) -> Any:
    """PEP 562: `_adapters.resilience` 経由でシンボルを遅延取得しキャッシュする。

    SDK 生型と build 関数の実装実体である `_adapters.resilience` のロードを属性アクセス時
    まで遅らせる。`agents` 自体はコア依存で本窓口の import より前にロード済みのため、遅延の
    対象ではない。取得済み値は `globals()` に載せて 2 回目以降を高速化する。

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
        # `pop` されたケース（テスト等）のフォールバックとして所属モジュールから再解決する。
        module = importlib.import_module(f".{_DIRECT_SYMBOLS[name]}", __package__)
        value = getattr(module, name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """`dir()` に遅延再エクスポート分を未 import でも含める。"""
    return sorted(set(globals()) | set(__all__))
