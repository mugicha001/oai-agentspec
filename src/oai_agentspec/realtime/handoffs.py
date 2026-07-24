"""Realtime 用の宣言的ハンドオフトポロジ（`RealtimeHandoffEdge` / `RealtimeHandoffGraph`）。

コアの `HandoffGraph` と対称だが統合しない独立アーティファクト。ハンドオフトポロジを
`RealtimeAgentSpec` とは別に宣言し、`apply(specs)` で各 spec の `handoffs` /
`handoff_options` へ in-place 反映する。反映直後に共有バリデータ
（`_validation.validate_realtime_handoff_options`）で検証するため、`apply -> register`
でも `register -> apply` でも最終 spec が必ず検証される（順序非依存）。

`agents` / `openai` には一切依存しない（NFR-1）。エージェント名（str）と
`RealtimeHandoffConfig`（plain データ）のみを扱い、SDK 結合は registry / `_adapters` に委ねる。
本モジュールは `realtime/registry` を import しない（`entry_agent` の registry は引数受けで
型は `Any`）。コアの `handoffs.py` と異なり、動的破線（`dynamic_edge`）・`input_filter` は
持たない（`RealtimeHandoffConfig` が型として持たないため）。
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from .._mermaid import render_flowchart
from .._validation import validate_realtime_handoff_options
from .spec import RealtimeAgentSpec, RealtimeHandoffConfig


@dataclass
class RealtimeHandoffEdge:
    """Realtime 静的ハンドオフ 1 エッジ（固定 1 ターゲット）。

    Attributes:
        src: ハンドオフ元エージェント名。
        dst: ハンドオフ先エージェント名。
        config: per-edge のハンドオフ設定。`None` はデフォルト（設定なし）を表す。
    """

    src: str
    dst: str
    config: RealtimeHandoffConfig | None = None


@dataclass
class RealtimeHandoffGraph:
    """Realtime 用の宣言的ハンドオフトポロジ。`apply(specs)` で spec 群へ結線する。

    グラフは宣言アーティファクトとして spec を組み立てるだけで registry を変更しない
    （MVP が意図的に除外した update / built 無効化基盤を導入しない）。`register` の
    前後どちらの順序で `apply` しても、共有バリデータにより最終 spec が必ず検証される
    （順序非依存）。再 apply でエッジが消えた src の `handoffs` は自動クリアしない
    （エッジを持つ src のみ replace する一回性反映）。

    使用例::

        graph = RealtimeHandoffGraph(entry="triage")
        graph.edge("triage", "billing", tool_description="請求")
        graph.apply([triage, billing])
        registry.register(triage)
        registry.register(billing)
    """

    entry: str | None = None
    edges: list[RealtimeHandoffEdge] = field(default_factory=list)

    def edge(
        self,
        src: str,
        dst: str,
        *,
        on_handoff: Any = None,
        input_type: Any = None,
        tool_name: str | None = None,
        tool_description: str | None = None,
        is_enabled: Any = True,
    ) -> RealtimeHandoffGraph:
        """静的エッジを追加する（自身を返す・fluent）。

        引数は SDK `realtime_handoff()` の対応引数（`RealtimeHandoffConfig` フィールド）へ
        マップされる: `tool_name` -> `tool_name_override` / `tool_description` ->
        `tool_description_override` / `on_handoff`・`input_type`・`is_enabled` は同名。

        Args:
            src: ハンドオフ元エージェント名。
            dst: ハンドオフ先エージェント名。
            on_handoff: ハンドオフ発火時のコールバック。
            input_type: 転送時に LLM が埋める構造化入力の型。
            tool_name: ハンドオフ tool 名（`tool_name_override`）。
            tool_description: ハンドオフ tool の説明（`tool_description_override`）。
            is_enabled: 動的有効化（bool または callable）。

        Returns:
            自身（連鎖可能）。
        """
        config = RealtimeHandoffConfig(
            on_handoff=on_handoff,
            input_type=input_type,
            tool_name_override=tool_name,
            tool_description_override=tool_description,
            is_enabled=is_enabled,
        )
        self.edges.append(RealtimeHandoffEdge(src=src, dst=dst, config=config))
        return self

    def extend(self, edges: Iterable[tuple[str, str]]) -> RealtimeHandoffGraph:
        """(src, dst) タプル列をまとめて追加する（自身を返す）。"""
        for src, dst in edges:
            self.edge(src, dst)
        return self

    def outgoing(self, src: str) -> list[str]:
        """src の静的出辺の dst 名リストを返す。"""
        return [e.dst for e in self.edges if e.src == src]

    def apply(self, specs: Iterable[RealtimeAgentSpec]) -> None:
        """各 src のエッジを検証してから spec 群へ一括反映する（all-or-nothing）。

        specs を名前で索引し、エッジを持つ src の `spec.handoffs` を replace・
        `spec.handoff_options` を書き込む。反映は 2 パスで行う: パス 1 で全 src の
        反映値を組み立てて `validate_realtime_handoff_options` で検証し、パス 2 で
        一括代入する。途中の `KeyError` / `ValueError` ではどの spec も変異しない
        （原子性）。最終 spec が必ず検証済みになるため、`register` の前後どちらの
        順序で apply しても検証は迂回されない。

        apply は build 前のワンショット反映を前提とする。`registry.get()` で構築済みの
        エージェントは registry のキャッシュから返るため、構築後に apply しても
        既存エージェントの結線は変わらない（apply は spec のみを書き換え、registry の
        キャッシュには関与しない）。apply は必ず最初の `get()` より前に行うこと。

        エッジを持たない src の spec は触らない（一回性反映）。再 apply でエッジが消えた src の
        `handoffs` は自動クリアしない（別グラフの apply が既存結線を消去しない）。

        Args:
            specs: 反映対象の RealtimeAgentSpec 群。

        Raises:
            KeyError: グラフに現れる src が specs に存在しない場合（spec は変異しない）。
            ValueError: specs に同名 spec が複数含まれる場合、または反映値の
                handoff_options が検証に失敗した場合（spec は変異しない）。
        """
        spec_by_name: dict[str, RealtimeAgentSpec] = {}
        for spec in specs:
            if spec.name in spec_by_name:
                # 辞書内包の暗黙後勝ちで片方を silent に無視しないよう明示的に弾く
                # （register() の重複 ValueError と整合する）。
                raise ValueError(f"specs に同名の spec が複数あります: {spec.name!r}")
            spec_by_name[spec.name] = spec
        srcs: list[str] = []
        for e in self.edges:
            if e.src not in srcs:
                srcs.append(e.src)
        # パス 1: 反映値の組み立てと検証（spec には触らない）。
        staged: list[tuple[RealtimeAgentSpec, list[str], dict[str, RealtimeHandoffConfig]]] = []
        for src in srcs:
            if src not in spec_by_name:
                raise KeyError(f"handoff source {src!r} は specs に存在しません")
            dsts: list[str] = []
            options_map: dict[str, RealtimeHandoffConfig] = {}
            for e in self.edges:
                if e.src != src:
                    continue
                if e.dst not in dsts:
                    dsts.append(e.dst)
                if e.config is not None and e.config != RealtimeHandoffConfig():
                    options_map[e.dst] = e.config
            validate_realtime_handoff_options(src, dsts, options_map)
            staged.append((spec_by_name[src], dsts, options_map))
        # パス 2: 全件が検証を通過してから一括反映する（途中失敗で部分変異を残さない）。
        for spec, dsts, options_map in staged:
            spec.handoffs = dsts
            spec.handoff_options = options_map

    def entry_agent(self, registry: Any) -> Any:
        """entry エージェントを registry から取得する。

        Args:
            registry: `get(name)` を持つ RealtimeAgentRegistry（import 辺を作らないため Any）。

        Returns:
            entry 名の構築済み RealtimeAgent。

        Raises:
            ValueError: entry が未設定の場合。
        """
        if self.entry is None:
            raise ValueError("no entry agent set on RealtimeHandoffGraph")
        return registry.get(self.entry)

    def mermaid(self) -> str:
        """グラフを Mermaid flowchart 文字列として返す（静的エッジのみ・破線なし）。

        ラベル源は `tool_description_override`（未設定時は無ラベル）。コアの `HandoffGraph`
        と異なり動的破線（`-.->`）を持たない。書式は共有フォーマッタ `_mermaid` が
        単一ソースで保つ。
        """
        static_edges = [
            (
                e.src,
                e.dst,
                e.config.tool_description_override if e.config is not None else None,
            )
            for e in self.edges
        ]
        return render_flowchart(self.entry, static_edges)


def from_specs(
    specs: Iterable[RealtimeAgentSpec], entry: str | None = None
) -> RealtimeHandoffGraph:
    """RealtimeAgentSpec 群の `handoffs` 宣言から RealtimeHandoffGraph を構築する。

    各 spec の `handoffs` から静的エッジを張り、`handoff_options` があれば config として
    引き継ぐ（コア `handoffs.from_specs` と対称）。

    Args:
        specs: 元にする RealtimeAgentSpec 群。
        entry: エントリエージェント名。

    Returns:
        構築済みの RealtimeHandoffGraph。

    Raises:
        ValueError: specs に同名 spec が複数含まれる場合（`apply` の重複検出と整合）。
    """
    graph = RealtimeHandoffGraph(entry=entry)
    seen: set[str] = set()
    for spec in specs:
        if spec.name in seen:
            raise ValueError(f"specs に同名の spec が複数あります: {spec.name!r}")
        seen.add(spec.name)
        for dst in spec.handoffs:
            config = spec.handoff_options.get(dst)
            graph.edges.append(RealtimeHandoffEdge(src=spec.name, dst=dst, config=config))
    return graph


__all__ = ["RealtimeHandoffEdge", "RealtimeHandoffGraph", "from_specs"]
