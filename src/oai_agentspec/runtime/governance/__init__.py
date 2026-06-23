"""AGT ガバナンスの公開窓口（`oai-agentspec[governance]` extra・agents 非依存・公開 API）。

宣言したエージェントが「何をできるか」をツール単位のポリシーで許可 / 拒否し、許可 / 拒否を監査ログ
へ記録する装飾 builder（`GovernedAgentBuilder`）を再エクスポートする。`AgentRegistry(agent_builder=
...)` へ注入すると、registry の遅延構築経路を通る全 spec の tools が govern ラップされ、監査
`AgentHooks` が装着される。`AgentSpec` / `tools` / `AgentBuilder` Protocol の宣言面は不変。

SDK 型（`agents`）と AGT（`agent-governance-toolkit`）の import は `_adapters/governance.py` に閉
じ、本窓口は不透明値（policy / audit_sink）のみ扱う（SDK / 外部クライアント隔離・NFR-1）。コア
`__init__` の `__all__` には載せない（公開 API は本窓口に集約・装飾 builder は実行寄りであり宣言層
シンボルのみのコア `__all__` 原則に従う）。AGT の import は `govern_spec` 内の関数内遅延に閉じるの
で、本窓口からの import は governance extra 未導入でも壊れない（build 時に必要 extra を案内する）。

拒否例外 `PolicyViolationError`（AGT 由来・isinstance 互換）は本窓口から遅延再エクスポートする
（PEP 562 の module `__getattr__`）。窓口 import 自体は extra 未導入でも壊れず、属性アクセス時に
初めて AGT を遅延 import する（未導入時は install hint 付き `ImportError`）。AGT 内部パッケージ
からの直接 import や DeprecationWarning 抑制のボイラープレートは不要になる。

extra 未導入時の挙動の含意（仕様）: 属性アクセスは `AttributeError` でなく案内付き
`ImportError` を送出するため、`hasattr(...)` での存在プローブや
`from oai_agentspec.runtime.governance import *` も未導入環境では `ImportError` になる
（無言の失敗をしない fail-fast を優先する設計）。`dir()` には `__dir__` により未 import でも
公開シンボルが列挙される。
"""

from __future__ import annotations

from typing import Any

from .builder import GovernedAgentBuilder

__all__ = [
    "GovernedAgentBuilder",
    "PolicyViolationError",
]


def __getattr__(name: str) -> Any:
    """`PolicyViolationError` を遅延再エクスポートする（PEP 562）。

    AGT の import を属性アクセス時まで遅らせることで、governance extra 未導入でも本窓口の
    import 自体は壊れない。取得済みの値は module 属性へキャッシュし、以降のアクセスは通常の
    属性解決で返す。

    Args:
        name: アクセスされた属性名。

    Returns:
        `PolicyViolationError` クラス（AGT が送出する例外クラスそのもの）。

    Raises:
        AttributeError: 本窓口が公開しない属性名の場合。
        ImportError: governance extra（agent-governance-toolkit）が未導入の場合（案内付き）。
    """
    if name == "PolicyViolationError":
        from ..._adapters import policy_violation_error_type

        exc_type = policy_violation_error_type()
        globals()[name] = exc_type
        return exc_type
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """`dir()` に遅延再エクスポート分（`PolicyViolationError`）を未 import でも含める。"""
    return sorted(set(globals()) | set(__all__))
