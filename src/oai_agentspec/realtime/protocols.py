"""Realtime ルートの DI 注入点の Protocol 定義。

`agents` をランタイム import しない。SDK 型（`RealtimeAgent` / `Handoff`）は不透明型
（`Any`）として扱い、`_adapters` 側で SDK 結合を閉じる（詳細は
docs/architecture.md「依存性注入（DI）」）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .spec import RealtimeAgentSpec, RealtimeHandoffConfig


@runtime_checkable
class RealtimeAgentBuilder(Protocol):
    """単一 RealtimeAgent の構築とハンドオフ結線を担う責務（DI 注入点）。

    `build` は handoffs 空で構築し、ハンドオフの結線は registry の局所 2 パス遅延バインドが
    `make_handoff` を用いて担う。テストでは本物の `agents.realtime.RealtimeAgent` を構築しない
    フェイクを注入できる。
    """

    def build(self, spec: RealtimeAgentSpec) -> Any:
        """spec から handoffs 空の RealtimeAgent を 1 つ構築する。

        Args:
            spec: 構築対象の RealtimeAgentSpec。

        Returns:
            構築された agents.realtime.RealtimeAgent（handoffs は空）を表す不透明型。

        Raises:
            ValueError: extra に専用フィールドと同名キー、または未知のキーがある場合。
        """
        ...

    def make_handoff(self, agent: Any, config: RealtimeHandoffConfig | None) -> Any:
        """構築済みの target agent とエッジ設定からハンドオフオブジェクトを生成する。

        Args:
            agent: ハンドオフ先の構築済み RealtimeAgent（不透明型）。
            config: エッジ設定。省略（None）時は既定設定で結線する。

        Returns:
            SDK `realtime_handoff()` が返す Handoff を表す不透明型。

        Note:
            `RealtimeHandoffConfig` は `input_filter` を型として持たないため、非対応の
            `input_filter` は指定不能（型レベル排除で完結し、実行時 reject は不要）。
        """
        ...


__all__ = ["RealtimeAgentBuilder"]
