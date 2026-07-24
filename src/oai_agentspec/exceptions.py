"""lib 独自例外（9 種）の統一窓口（再エクスポート専用・agents 非依存）。

利用者が例外の定義元モジュールを把握しなくても ``oai_agentspec.exceptions`` 単体を
import すれば全種を捕捉できるようにする窓口。定義元は変更せず（既存の公開契約を維持）、
本モジュールは再エクスポートに徹する。

シンボルは取得方法で 2 系統に分かれる。

- 直 import（7 種）: ``RegistryFrozenError`` / ``IntegrityError`` /
  ``PromptTemplateIntegrityError`` / ``PromptResolutionError`` / ``WorkflowFrozenError`` /
  ``RunBudgetExceeded`` / ``ConversationError``。定義元モジュール（``.registry`` /
  ``.integrity`` / ``.prompts`` / ``.workflow.graph`` /
  ``.runtime.resilience._errors`` / ``.runtime.conversation.types``）はいずれも
  ``agents`` / ``openai`` は非依存で、追加の外部依存も持たない。窓口 import 時に
  発火しても NFR-1（SDK 隔離）にも extra 未導入耐性にも抵触しないため、
  ``runtime/resilience/__init__.py`` の直 import 分（``ModelRetryPolicy`` 等）と
  同じ理由で直 import としてよい。

- PEP 562 遅延取得（2 種）: ``OptimizeError``（``.runtime.lightning.types``）/
  ``ConversationClientError``（``.runtime.cli._models``）。この 2 種は
  ``lightning`` extra（追加の重い依存を伴う最適化系）と ``cli`` extra
  （別プロセス・httpx/websockets 依存）に属し、窓口 import だけでこれら extra 未導入時に
  ImportError を起こしてはならない。他窓口の遅延分（``_adapters.resilience`` 経由）は
  SDK 生型を SDK アダプタ層から取得するためのものだが、本窓口の遅延 2 種はそれぞれの
  extra 内で完結する plain 例外であり `_adapters` を経由しない。遅延先を
  ``.runtime.lightning.types`` / ``.runtime.cli._models`` という**定義元モジュール自体**に
  したのは、`_adapters` 層が存在しない（SDK ラップ不要な独立 extra である）ためで、
  `__getattr__` はそれぞれの定義元モジュールへ直接遅延する。

窓口 import 自体は ``lightning`` / ``cli`` extra 未導入でも壊れない
（未導入時に遅延 2 種へアクセスした場合のみ ImportError が伝播する）。
"""

from __future__ import annotations

from typing import Any

from .integrity import IntegrityError, PromptTemplateIntegrityError
from .prompts import PromptResolutionError
from .registry import RegistryFrozenError
from .runtime.conversation.types import ConversationError
from .runtime.resilience._errors import RunBudgetExceeded
from .workflow.graph import WorkflowFrozenError

__all__ = [
    "ConversationClientError",  # noqa: F822 - PEP 562 __getattr__ による遅延解決（下記参照）
    "ConversationError",
    "IntegrityError",
    "OptimizeError",  # noqa: F822 - PEP 562 __getattr__ による遅延解決（下記参照）
    "PromptResolutionError",
    "PromptTemplateIntegrityError",
    "RegistryFrozenError",
    "RunBudgetExceeded",
    "WorkflowFrozenError",
]


# `lightning` / `cli` extra の定義元モジュールへ直接遅延取得するシンボル集合。
_DEFERRED_SYMBOLS = frozenset(
    {
        "OptimizeError",
        "ConversationClientError",
    }
)


def __getattr__(name: str) -> Any:
    """PEP 562: `lightning` / `cli` extra の定義元モジュールから遅延取得しキャッシュする。

    窓口 import 時に `lightning` / `cli` extra の依存を発火させないため、該当 2 種は
    本関数で初めてロードする。取得済み値は `globals()` に載せて 2 回目以降を高速化する。

    Args:
        name: アクセスされた属性名。

    Returns:
        該当する公開例外クラス（`OptimizeError` または `ConversationClientError`）。

    Raises:
        AttributeError: `__all__` に含まれない属性名の場合。
    """
    if name not in _DEFERRED_SYMBOLS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if name == "OptimizeError":
        from .runtime.lightning import types as _lightning_types

        value = getattr(_lightning_types, name)
    else:
        from .runtime.cli import _models as _cli_models

        value = getattr(_cli_models, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """`dir()` に遅延再エクスポート分を未 import でも含める。"""
    return sorted(set(globals()) | set(__all__))
