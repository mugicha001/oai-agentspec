"""`RunHooksBase` 合成ヘルパー `chain_hooks` の公開窓口（agents はコア依存・extra 不要）。

`Runner.run(hooks=...)` が単数の `RunHooksBase` しか受け付けないため、`build_run_budget_hooks`
と他 hooks を併用したい場合の合成窓口として `chain_hooks(*hooks)` を再エクスポートする。実装実体は
`_adapters/hooks.py`（`RunHooksBase` サブクラス定義のため `agents.lifecycle` の import が不可避で、
SDK 隔離 NFR-1 に従い `_adapters/` に配置）。本窓口は PEP 562 の module `__getattr__` で
`chain_hooks` を遅延取得し、`import oai_agentspec.runtime.hooks` 時点では `agents` を発火させない
（`governance` 窓口と同型の遅延パターン）。

合成仕様（詳細は `_adapters/hooks.py` の module docstring と `docs/architecture.md` の
「hooks 合成（chain_hooks）」節を参照）:

- 0 引数: 素の `RunHooksBase()` を返す（no-op）。
- 1 引数: 渡した hook をそのまま返す（`is` 一致）。
- 2 引数以上: 全 7 hook メソッドを宣言順に順次 `await`・fail-fast・引数無変更転送。

コア `__all__` には載せない独立窓口（実行寄り層のため）。利用者は `from oai_agentspec.runtime.hooks
import chain_hooks` で参照する。
"""

from __future__ import annotations

from typing import Any

__all__ = ["chain_hooks"]


def __getattr__(name: str) -> Any:
    """`chain_hooks` を `_adapters.hooks` から遅延取得して再エクスポートする（PEP 562）。

    `agents.lifecycle` の import を属性アクセス時まで遅らせ、`import oai_agentspec.runtime.hooks`
    時点では SDK を発火させない。取得済みの値は module 属性へキャッシュし、以降のアクセスは通常の
    属性解決で返す。

    Args:
        name: アクセスされた属性名。

    Returns:
        `_adapters.hooks.chain_hooks`（合成 factory 関数）。

    Raises:
        AttributeError: 本窓口が公開しない属性名の場合。
    """
    if name == "chain_hooks":
        from ..._adapters.hooks import chain_hooks as _chain_hooks

        globals()[name] = _chain_hooks
        return _chain_hooks
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """`dir()` に遅延再エクスポート分（`chain_hooks`）を未 import でも含める。"""
    return sorted(set(globals()) | set(__all__))
