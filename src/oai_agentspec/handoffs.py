"""宣言的ハンドオフトポロジ（`HandoffEdge` / `HandoffGraph`）。

トポロジを AgentSpec とは別アーティファクトとして宣言し、registry の内部プリミティブ
（`_update_handoffs`）経由で反映する。registry の生の内部状態には直接アクセスしない。

エッジは 2 種類:
    - 静的エッジ（`edge`）: 固定 1 ターゲットへのハンドオフ。HandoffConfig で per-edge 設定。
    - 動的エッジ（`dynamic_edge`）: resolver が候補から転送先を実行時決定する（on_invoke）。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from dataclasses import fields as _dataclass_fields
from typing import TYPE_CHECKING, Any

from .spec import DynamicHandoff, HandoffConfig

if TYPE_CHECKING:
    from .registry import AgentRegistry


@dataclass
class HandoffEdge:
    """静的ハンドオフ 1 エッジ（固定 1 ターゲット）。

    Attributes:
        src: ハンドオフ元エージェント名。
        dst: ハンドオフ先エージェント名。
        config: per-edge のハンドオフ設定（description / on_handoff / input_type 等）。
    """

    src: str
    dst: str
    config: HandoffConfig = field(default_factory=HandoffConfig)


@dataclass
class DynamicHandoffEdge:
    """動的ハンドオフ 1 エッジ（候補から実行時に転送先を選ぶ）。

    `on_handoff` / `input_type` / `input_filter` / `is_enabled` / `options` は静的エッジの
    `HandoffConfig` と同じ意味を持ち、利用側に「静的・動的」の二重学習を強いない（型付き
    フィールド優先・`options` は SDK 固有の追加 kwarg 専用の裏口で、型付きフィールドと
    同義の予約キーは `_adapters` 側で `ValueError` として弾く）。

    Attributes:
        src: ハンドオフ元エージェント名。
        tool_name: ハンドオフ tool 名。
        candidates: 転送先候補のエージェント名。
        resolver: `(context, input_json) -> 転送先名`（候補内）。
        description: ハンドオフ tool の説明。
        on_handoff: ハンドオフ発火時のコールバック（転送先決定後・Agent 返却前に発火）。
            `input_type` 指定時は `(context, parsed_input)`、未指定時は `(context,)` を受ける。
        input_type: 転送時に LLM が埋める構造化入力の型（pydantic モデル等）。
        input_filter: 次エージェントへ渡す履歴の変換（`Handoff.input_filter` へ素通し）。
        is_enabled: 動的有効化（bool または `(context, agent) -> bool` callable）。
        options: 上記以外の生 `Handoff` kwarg 素通し用 dict。型付きフィールドと同義の
            予約キーは禁止（`_adapters` 側で `ValueError`）。
    """

    src: str
    tool_name: str
    candidates: list[str]
    resolver: Callable[..., Any]
    description: str | None = None
    on_handoff: Any = None
    input_type: Any = None
    input_filter: Any = None
    is_enabled: Any = True
    options: dict[str, Any] = field(default_factory=dict)


def _edge_to_dynamic_handoff(edge: DynamicHandoffEdge) -> DynamicHandoff:
    """`DynamicHandoffEdge` から `DynamicHandoff` を組み立てる（同名フィールドを自動コピー）。

    呼出毎に `dataclasses.fields(DynamicHandoff)` を引いて edge 側の同名属性を getattr で
    取得する。`dataclasses.fields()` は class 内部でキャッシュされているため呼出コストは
    小さく、static cache を持たないことで runtime での DynamicHandoff 拡張（テスト fixture
    等）にも追従する。ミュータブルなコンテナ（`candidates` / `options`）は apply 時点の
    snapshot として新規コピーし、Edge の後続変異が Handoff に伝播しないようにする。

    両 dataclass のフィールド集合が一致している前提で、片側拡張時の同期忘れを防ぐ
    （フィールド追加時に本ヘルパに変更は不要。パリティはユニットテスト
    `test_handoff_config_and_dynamic_handoff_share_field_names` で保証）。

    Args:
        edge: `DynamicHandoffEdge` インスタンス。

    Returns:
        対応する `DynamicHandoff` インスタンス（ミュータブルコンテナは新規コピー）。
    """
    kwargs: dict[str, Any] = {}
    for f in _dataclass_fields(DynamicHandoff):
        name = f.name
        value = getattr(edge, name)
        if name == "candidates":
            value = list(value)
        elif name == "options":
            value = dict(value)
        kwargs[name] = value
    return DynamicHandoff(**kwargs)


@dataclass
class HandoffGraph:
    """宣言的ハンドオフトポロジ。registry へ内部プリミティブ経由で反映する。

    使用例::

        graph = HandoffGraph(entry="triage")
        graph.edge("triage", "billing", description="請求")
        graph.edge("triage", "support", on_handoff=on_escalate, input_type=Escalation)
        graph.dynamic_edge("triage", ["billing", "support"], route, tool_name="route")
        graph.apply(registry)
    """

    entry: str | None = None
    edges: list[HandoffEdge] = field(default_factory=list)
    dynamic: list[DynamicHandoffEdge] = field(default_factory=list)
    # このグラフが直近の apply で反映した src 集合。次回 apply でエッジが無くなった src を
    # 空クリアするために保持する（replace セマンティクスの担保）。equality 比較からは除外。
    _applied_srcs: set[str] = field(default_factory=set, compare=False, repr=False)

    def edge(
        self,
        src: str,
        dst: str,
        description: str | None = None,
        *,
        tool_name: str | None = None,
        on_handoff: Any = None,
        input_type: Any = None,
        input_filter: Any = None,
        is_enabled: Any = True,
        options: dict[str, Any] | None = None,
    ) -> HandoffGraph:
        """静的エッジを追加する（自身を返す）。

        型付き引数は SDK `handoff()` の対応引数へマップされる。専用フィールドに無い
        handoff() 引数は `options` で素通しする。
        """
        config = HandoffConfig(
            description=description,
            tool_name=tool_name,
            on_handoff=on_handoff,
            input_type=input_type,
            input_filter=input_filter,
            is_enabled=is_enabled,
            options=dict(options or {}),
        )
        self.edges.append(HandoffEdge(src=src, dst=dst, config=config))
        return self

    def dynamic_edge(
        self,
        src: str,
        candidates: Iterable[str],
        resolver: Callable[..., Any],
        *,
        tool_name: str,
        description: str | None = None,
        on_handoff: Any = None,
        input_type: Any = None,
        input_filter: Any = None,
        is_enabled: Any = True,
        options: dict[str, Any] | None = None,
    ) -> HandoffGraph:
        """動的エッジを追加する（自身を返す）。

        resolver は `(context, input_json) -> 転送先名` を返す（候補内に限る）。
        `on_handoff` / `input_type` / `input_filter` / `is_enabled` / `options` は静的
        `edge` と同じ意味で、SDK `handoff()` の糖衣を `_adapters` 内で再現する。

        `options` は宣言時に `dict(options or {})` で snapshot し、`apply()` 時にも
        `_edge_to_dynamic_handoff` 内で再 snapshot する 2 段階の防御コピーを意図的に行う:
        - 宣言時 snapshot: 利用者が渡した dict の後続変異が Edge.options に波及しないようにする
        - apply 時 snapshot: Edge.options への直接アクセス変異が Handoff.options に波及しない
          ようにする（dataclass フィールドはミュータブル）
        dict は小さいためコスト無視可能で、層境界での所有権分離を優先する。
        """
        self.dynamic.append(
            DynamicHandoffEdge(
                src=src,
                tool_name=tool_name,
                candidates=list(candidates),
                resolver=resolver,
                description=description,
                on_handoff=on_handoff,
                input_type=input_type,
                input_filter=input_filter,
                is_enabled=is_enabled,
                options=dict(options or {}),
            )
        )
        return self

    def extend(self, edges: Iterable[tuple[str, str]]) -> HandoffGraph:
        """(src, dst) タプル列をまとめて追加する。"""
        for src, dst in edges:
            self.edge(src, dst)
        return self

    def outgoing(self, src: str) -> list[str]:
        """src の静的出辺の dst 名リスト。"""
        return [e.dst for e in self.edges if e.src == src]

    def apply(self, registry: AgentRegistry) -> None:
        """各 src のエッジを registry の内部プリミティブ（_update_handoffs）で反映する。

        mode="replace" で、グラフを当該 src のトポロジの真実源として上書きする
        （グラフを編集して再 apply すれば実行時に再構成できる）。前回 apply で反映したが
        今回エッジが無くなった src は空に上書きし、削除済みのルートが残らないようにする。

        Raises:
            KeyError: 現行エッジの src が spec ベースで未登録（factory 起点を含む）の場合。
        """
        spec_names = set(registry.names()) - set(_factory_names(registry))
        current_srcs = {e.src for e in self.edges} | {d.src for d in self.dynamic}
        for src in current_srcs:
            if src not in spec_names:
                raise KeyError(
                    f"handoff source {src!r} は spec ベースの登録ではありません"
                    "（factory 起点は手動で結線してください）"
                )
        # 現行 src + 前回反映したが今回消えた src（後者は空クリア対象）。
        for src in current_srcs | self._applied_srcs:
            if src not in spec_names:
                # 前回反映済みだが今は未登録（unregister 済み等）→ クリア不要。
                continue
            dsts: list[str] = []
            options_map: dict[str, HandoffConfig] = {}
            for e in self.edges:
                if e.src != src:
                    continue
                if e.dst not in dsts:
                    dsts.append(e.dst)
                if e.config != HandoffConfig():
                    options_map[e.dst] = e.config
            dynamic = [_edge_to_dynamic_handoff(d) for d in self.dynamic if d.src == src]
            registry._update_handoffs(  # noqa: SLF001 - apply は registry の委譲先
                src,
                dsts,
                mode="replace",
                handoff_options=options_map,
                dynamic_handoffs=dynamic,
            )
        self._applied_srcs = set(current_srcs)

    def entry_agent(self, registry: AgentRegistry) -> Any:
        """entry エージェントを取得する。

        Raises:
            ValueError: entry が未設定の場合。
        """
        if self.entry is None:
            raise ValueError("no entry agent set on HandoffGraph")
        return registry.get(self.entry)

    def mermaid(self) -> str:
        """グラフを Mermaid flowchart 文字列として返す。"""
        lines = ["flowchart TD"]
        if self.entry:
            lines.append(f"    start([start]) --> {self.entry}")
        for e in self.edges:
            label = f"|{e.config.description}|" if e.config.description else ""
            lines.append(f"    {e.src} -->{label} {e.dst}")
        for d in self.dynamic:
            for cand in d.candidates:
                lines.append(f"    {d.src} -.->|{d.tool_name}| {cand}")
        return "\n".join(lines)


def from_specs(specs: Iterable[Any], entry: str | None = None) -> HandoffGraph:
    """AgentSpec 群の `handoffs` 宣言から HandoffGraph を構築する。"""
    graph = HandoffGraph(entry=entry)
    for spec in specs:
        for dst in spec.handoffs:
            graph.edge(spec.name, dst)
    return graph


def _factory_names(registry: AgentRegistry) -> Iterable[str]:
    return getattr(registry, "_factories", {}).keys()  # noqa: SLF001 - apply 判定のみ


__all__ = ["HandoffEdge", "DynamicHandoffEdge", "HandoffGraph", "from_specs"]
