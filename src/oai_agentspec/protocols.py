"""DI 注入点の Protocol 定義。

`agents` をランタイム import しない。SDK 型（`Agent`）は `TYPE_CHECKING` ブロックで
`._adapters` の型エイリアス経由で参照する（詳細は docs/architecture.md「依存性注入（DI）」）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

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


@runtime_checkable
class GuardrailProvider(Protocol):
    """名前から guardrail を解決する責務（DI 注入点）。

    `AgentSpec.guardrails` に宣言された名前を実体へ解決し、宣言された適用境界を答える。
    `AgentRegistry` は build 時にこの 2 照会だけを使い、解決元の実装型には触れない。

    照会を 2 つに絞り、戻り値を plain（不透明型と str）に閉じているのは、コア層から
    `runtime/guardrails` への依存辺を作らないためである。境界の値域型や宣言型を戻り値の
    型注釈に用いると、コア層が実装型を参照して単方向依存が崩れる。

    登録簿が持つ他の照会（宣言 1 件の取得・一覧・危険度）は本 Protocol の契約に含めない。
    自作の解決元へ実装を要求せず、必要な利用者が登録簿を直接使う形にする。
    """

    def get(self, name: str) -> Any:
        """登録名から guardrail 実体を返す。

        Args:
            name: 解決する guardrail 名。

        Returns:
            SDK 互換の guardrail 実体（不透明型として扱う）。

        Raises:
            KeyError: name が未登録の場合。
        """
        ...

    def boundary_of(self, name: str) -> str:
        """登録名から宣言された適用境界を返す。

        Args:
            name: 解決する guardrail 名。

        Returns:
            適用境界の文字列（`"input"` / `"output"` / `"tool_input"` / `"tool_output"`）。

        Raises:
            KeyError: name が未登録の場合。
        """
        ...


__all__ = ["AgentBuilder", "GuardrailProvider"]
