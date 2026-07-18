"""DI 注入点の Protocol 定義。

`agents` をランタイム import しない。SDK 型（`Agent`）は `TYPE_CHECKING` ブロックで
`._adapters` の型エイリアス経由で参照する（詳細は docs/architecture.md「依存性注入（DI）」）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ._adapters import Agent
    from .spec import AgentSpec


@runtime_checkable
class AgentBuilder(Protocol):
    """単一 Agent を構築する責務（DI 注入点）。

    handoffs は空で構築し、サブエージェントのツールも注入しない。これらの結線は
    registry の局所 2 パス遅延バインドが担う。テストでは本物の `agents.Agent` を
    構築しないフェイクを注入できる。

    spec は `AgentSpec` のサブクラス（`SandboxAgentSpec` 等）でありうる。デフォルト
    実装（`_adapters.build_agent`）は `SandboxAgentSpec` に対して
    `agents.sandbox.SandboxAgent` を構築する。カスタム実装が構築先クラスをどう扱うかは
    実装側の責務となる。デフォルト実装が構築へ反映するのはライブラリが宣言する
    フィールドのみで、利用者定義のサブクラスが追加した独自フィールドは関知しない
    （反映したい場合はカスタム builder を注入する）。
    """

    def build(self, spec: AgentSpec) -> Agent:
        """spec から handoffs 空の Agent を 1 つ構築する。

        Args:
            spec: 構築対象の AgentSpec。

        Returns:
            構築された agents.Agent（handoffs は空・サブツール未注入）。

        Raises:
            ValueError: extra に専用フィールドと同名キー、または未知のキーがある場合。
        """
        ...


__all__ = ["AgentBuilder"]
