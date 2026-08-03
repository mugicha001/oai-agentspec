"""hooks 合成ヘルパー（`chain_hooks` / `chain_agent_hooks`）の公開窓口（agents はコア依存）。

`Runner.run(hooks=...)` が単数の `RunHooksBase` しか受け付けず、`AgentSpec.hooks` も単一スロットの
ため、複数 hooks を併用したい場合の合成窓口として run 単位 `chain_hooks(*hooks)` と agent 単位
`chain_agent_hooks(*hooks)` を再エクスポートする。実装実体は `_adapters/hooks.py`（`RunHooksBase` /
`AgentHooksBase` サブクラス定義のため `agents.lifecycle` の import が不可避で、SDK 隔離 NFR-1 に従い
`_adapters/` に配置）。本窓口は PEP 562 の module `__getattr__` で両シンボルを遅延取得し、
`import oai_agentspec.runtime.hooks` 時点では合成クラス定義を含む `_adapters.hooks` をロードしない
（`governance` 窓口と同型の遅延パターン）。`agents` / `agents.lifecycle` 自体はコア依存であり、
`oai_agentspec/__init__.py` -> `_adapters/__init__.py` の連鎖で本窓口の import より前にロード済みに
なる。遅延の対象は実装実体のモジュールであって SDK ではない。

合成仕様（詳細は `_adapters/hooks.py` の module docstring と `docs/architecture.md` の
「hooks 合成（chain_hooks）」節を参照）:

- `chain_hooks`（run 単位・`build_run_budget_hooks` 等との併用）
    - 0 引数: 素の `RunHooksBase()` を返す（no-op）。
    - 1 引数: 渡した hook をそのまま返す（`is` 一致）。
    - 2 引数以上: 全 7 hook メソッドを宣言順に順次 `await`・fail-fast・引数無変更転送。
- `chain_agent_hooks`（agent 単位・`AgentSpec(hooks=...)` へ素通しできる）
    - `AgentHooksBase` インスタンス / `on_*` の一部だけを持つ部分実装 / `None` を混在で受理する。
    - `None` を除いた実効 0 件: 素の `AgentHooksBase()` を返す（no-op）。
    - 実効 1 件かつ `AgentHooksBase` インスタンス: そのフックをそのまま返す（`is` 一致）。
    - それ以外: 全 7 hook メソッドを宣言順に順次委譲・fail-fast・引数無変更転送（当該メソッドを
      持たない要素は skip）。

コア `__all__` には載せない独立窓口（実行寄り層のため）。利用者は `from oai_agentspec.runtime.hooks
import chain_hooks, chain_agent_hooks` で参照する。
"""

from __future__ import annotations

from typing import Any

__all__ = ["chain_agent_hooks", "chain_hooks"]


def __getattr__(name: str) -> Any:
    """`chain_hooks` / `chain_agent_hooks` を `_adapters.hooks` から遅延取得する（PEP 562）。

    `_adapters.hooks`（`RunHooksBase` / `AgentHooksBase` のサブクラス定義を含む）のロードを属性
    アクセス時まで遅らせる。`agents` / `agents.lifecycle` 自体はコア依存で本窓口の import より前に
    ロード済みなので、遅延の対象ではない。取得済みの値は module 属性へキャッシュし、以降のアクセスは
    通常の属性解決で返す。

    Args:
        name: アクセスされた属性名。

    Returns:
        `_adapters.hooks.chain_hooks` または `_adapters.hooks.chain_agent_hooks`
        （いずれも合成 factory 関数）。

    Raises:
        AttributeError: 本窓口が公開しない属性名の場合。
    """
    if name == "chain_hooks":
        from ..._adapters.hooks import chain_hooks as _chain_hooks

        globals()[name] = _chain_hooks
        return _chain_hooks
    elif name == "chain_agent_hooks":
        from ..._adapters.hooks import chain_agent_hooks as _chain_agent_hooks

        globals()[name] = _chain_agent_hooks
        return _chain_agent_hooks
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """`dir()` に遅延再エクスポート分（`__all__` の 2 シンボル）を未 import でも含める。"""
    return sorted(set(globals()) | set(__all__))
