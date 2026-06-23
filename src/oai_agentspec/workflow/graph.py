"""宣言的ワークフロー本体 `WorkflowGraph`（node/edge 方式・build-don't-run）。

ノード（AGENT/FUNCTION）とエッジ（通常 / 条件 / fan-in）+ `START`/`END` 番兵を `add_*` で明示
宣言し、`validate`（build-time 検証）/ `mermaid`（可視化）/ `as_agent_spec` / `as_facade_spec` /
`connect_as_facade`（Agent / Tool 化）を提供する。内部インタプリタの実体は `_interpreter`、
ファサード化の実体は `_facade` に分離し、本クラスのインタプリタ系 / ファサード系メソッドはそれら
フリー関数へ薄く委譲する（公開シグネチャ・挙動は不変）。

SDK 隔離（NFR-1）: 本モジュールは `agents` をランタイム import しない。SDK 型は `TYPE_CHECKING` /
`Protocol` のみで参照し、SDK 実体への結合は `_adapters` に閉じる。依存方向は
`workflow -> _adapters -> agents` の一方向（循環回避）。
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from ..constants import WORKFLOW_DEFAULT_RECURSION_LIMIT
from ..spec import AgentSpec
from . import _facade, _interpreter
from ._declarations import ConditionalEdge, FanInEdge, WorkflowNode, WorkflowResult
from ._types import (
    _UNSET,
    END,
    START,
    FacadeMode,
    NodeFn,
    NodeHook,
    NodeKind,
    Router,
    _as_targets,
    _check_reserved_run_keys,
)

if TYPE_CHECKING:
    from ..handoffs import HandoffGraph
    from ..registry import AgentRegistry
    from ._declarations import RunnerSeam


class WorkflowFrozenError(RuntimeError):
    """凍結後の ``WorkflowGraph`` に対する変更操作で raise される例外。

    ``IntegrityError`` 系統とは別で ``RuntimeError`` を継承するため、利用者の
    ``except IntegrityError`` で握り潰されない。例外メッセージは違反操作名を含む。
    """


__all__ = [
    "WorkflowFrozenError",
    "WorkflowGraph",
]


@dataclass
class WorkflowGraph:
    """宣言的ワークフロートポロジ（node/edge 明示宣言・run 非依存の dataclass）。

    `add_agent_node` / `add_function_node` でノードを、`add_edge` /
    `add_conditional_edges` / `add_fan_in_edge`（+ `START`/`END` 番兵）でエッジを明示
    宣言する。`validate` で build-time 検証、`mermaid` で可視化、`as_agent_spec` /
    `as_facade_spec` で Agent / Tool 化する。各宣言メソッドは self を返しチェーンできる。

    使用例::

        wf = WorkflowGraph("pipeline")
        wf.add_agent_node("plan", agent="planner")
        wf.add_function_node("format", fn=format_result)
        wf.add_edge(START, "plan")
        wf.add_edge("plan", "format")
        wf.add_edge("format", END)
        spec = wf.as_agent_spec("pipeline_agent")

    Attributes:
        name: ワークフロー名。
        nodes: ノード名 -> WorkflowNode。
        edges: 通常エッジ（src ノード名 -> 下流ノード名 | END のリスト。fan-out 可）。
        conditional_edges: src ノード名 -> ConditionalEdge。
        fan_in_edges: dst ノード名 -> FanInEdge（合流先で引く）。
        entry: START の唯一の下流ノード名（add_edge(START, ...) で設定・FR-2）。
        recursion_limit: 1 run の総ノード実行数の上限（無限ループ防止を兼ねる。超過で実行時
            エラー。ループの無い大きなグラフでは引き上げる・C-5）。
        run_defaults: 全 AGENT ノードの `Runner.run` へ素通しするグラフ既定 kwarg（session /
            max_turns / run_config / hooks 等）。ノード `run_options` で個別上書きする。
            `input` / `context` は予約キーで指定不可（lib 管理・FR-15）。
    """

    name: str
    nodes: dict[str, WorkflowNode] = field(default_factory=dict)
    edges: dict[str, list[Any]] = field(default_factory=dict)
    conditional_edges: dict[str, ConditionalEdge] = field(default_factory=dict)
    fan_in_edges: dict[str, FanInEdge] = field(default_factory=dict)
    entry: str | None = None
    recursion_limit: int = WORKFLOW_DEFAULT_RECURSION_LIMIT
    run_defaults: dict[str, Any] | None = None
    # freeze 後はノード／エッジ追加を遮断する。等価性 / repr / pickle / diff に影響しない
    # よう init=False / compare=False / repr=False とする
    # （既存 HandoffEdge._applied_srcs パターン踏襲）。
    _frozen: bool = field(default=False, init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        # グラフ既定の Runner kwargs は input/context を予約キーとして禁止する（lib 管理）。
        _check_reserved_run_keys(self.run_defaults, allow_session=True, where="run_defaults")

    @property
    def session(self) -> Any:
        """グラフ既定の SDK Session（`run_defaults['session']`。並列ガード判定に使う）。"""
        return (self.run_defaults or {}).get("session")

    # ------------------------------------------------------------------
    # 宣言（ノード）
    # ------------------------------------------------------------------
    def add_agent_node(
        self, name: str, *, agent: str, run_options: dict[str, Any] | None = None
    ) -> WorkflowGraph:
        """AGENT ノードを登録する（自身を返す）。

        上流から流れた出力（メッセージ）を入力に当該 Agent を実行する。AGENT ノードの
        入力は string / SDK input-list を期待する。上流が非文字列を返す場合は手前の
        FUNCTION ノードで string へ整形してから渡す（暗黙の str 化はしない・FR-1）。

        Args:
            name: ノード名（一意）。
            agent: registry 上のエージェント名（名前参照）。
            run_options: このノードの `Runner.run` へ素通しする kwarg（max_turns / run_config /
                hooks / conversation_id 等）。グラフ既定 `run_defaults` を dict マージで上書きする。
                `input` / `context` / `session` は予約キーで指定不可。

        Returns:
            自身（メソッドチェーン用）。

        Raises:
            ValueError: ノード名が重複する、または run_options に予約キーが含まれる場合。
            WorkflowFrozenError: ``freeze()`` 後の呼び出し。
        """
        self._ensure_unfrozen(f"add_agent_node (name={name!r})")
        _check_reserved_run_keys(
            run_options, allow_session=False, where=f"node {name!r} の run_options"
        )
        self._add_node(
            WorkflowNode(name=name, kind=NodeKind.AGENT, agent=agent, run_options=run_options)
        )
        return self

    def add_function_node(self, name: str, *, fn: NodeFn) -> WorkflowGraph:
        """FUNCTION ノードを登録する（自身を返す）。

        Args:
            name: ノード名（一意）。
            fn: `(msg, ctx) -> 出力` の callable（sync / async 両対応）。callable は
                ノードが直接保持する（registry 参照ではない）。fan-in の合流先では
                msg は `{source名: 出力}` の dict を受ける（C-4）。`ctx` は経路A/D では
                `RunContextWrapper`（`ctx.context` で渡した値を取り出す）、経路C では `None`
                になる（C-11）。両経路で使い回す関数は `getattr(ctx, "context", None)` で
                防御的に取り出すこと。

        Returns:
            自身（メソッドチェーン用）。

        Raises:
            ValueError: ノード名が重複する場合。
            WorkflowFrozenError: ``freeze()`` 後の呼び出し。
        """
        self._ensure_unfrozen(f"add_function_node (name={name!r})")
        self._add_node(WorkflowNode(name=name, kind=NodeKind.FUNCTION, fn=fn))
        return self

    def _add_node(self, node: WorkflowNode) -> None:
        if node.name in self.nodes:
            raise ValueError(f"workflow {self.name!r}: node が重複しています: {node.name!r}")
        self.nodes[node.name] = node

    # ------------------------------------------------------------------
    # 宣言（エッジ・全て self 返し）
    # ------------------------------------------------------------------
    def add_edge(self, src: Any, dst: Any) -> WorkflowGraph:
        """src -> dst の有向エッジを張る（自身を返す・FR-2）。

        `START` / `END` 番兵をエッジ端点に使える（`add_edge(START, "plan")` /
        `add_edge("write", END)`）。1 ノードから複数の `add_edge` を張ると fan-out
        （並列）になり、下流ノードが並行実行される（asyncio.gather）。`START` からの
        エッジは 1 本のみ（単一エントリ・validate で検査）。

        Args:
            src: 始点ノード名 または `START`。
            dst: 終点ノード名 または `END`。

        Returns:
            自身（メソッドチェーン用）。

        Raises:
            ValueError: `START` から複数のエントリを張ろうとした場合。
            WorkflowFrozenError: ``freeze()`` 後の呼び出し。
        """
        self._ensure_unfrozen(f"add_edge (src={src!r}, dst={dst!r})")
        if src is START:
            if self.entry is not None and self.entry != dst:
                raise ValueError(
                    f"workflow {self.name!r}: START からのエッジは 1 本のみです"
                    f"（既存エントリ {self.entry!r} と {dst!r} が競合）"
                )
            self.entry = dst
            return self
        self.edges.setdefault(src, []).append(dst)
        return self

    def add_conditional_edges(
        self,
        src: str,
        router: Router,
        mapping: Mapping[Hashable, Any] | None = None,
        *,
        default: Any = None,
        candidates: list[Any] | None = None,
    ) -> WorkflowGraph:
        """条件エッジを張る（router の戻り値で 1 経路を選ぶ・自身を返す・FR-2）。

        router の戻り値は次の 2 通りで解決する:
        - `mapping=None`: 戻り値を次ノード名 | `END` として直接使う（LangGraph の path_map
          無しモード相当）。
        - `mapping` あり: 戻り値を判定キーとして mapping で引く。bool/int/Enum 等の任意の
          hashable をキーにできる（文字列限定ではない）。未一致時は `default`（あれば）、
          無ければ実行時に例外。

        条件 fan-out: 戻り値（または mapping の値）に **ノード名のリスト**を返すと、その複数
        ノードを並行起動する。下流の `add_fan_in_edge` 合流先は、実際に起動された枝だけを
        待ち、`{走った source 名: 出力}` の dict を受ける（走らない枝はキー omit）。空リストは
        どこへも進まず当該ノードの出力が最終出力候補になる。

        ループは「ノードへ戻るエッジ + 条件エッジで `END` へ抜ける」形で表す。無限ループ
        防止は `recursion_limit` で行う（C-5）。

        Args:
            src: 分岐元ノード名。
            router: `(msg, ctx) -> ノード名 | END | 判定キー`。ctx は `RunContextWrapper | None`。
            mapping: 判定キー -> 次ノード名 | `END`。None で戻り値を直接ノード名として使う。
            default: mapping 未一致時の既定の行き先（ノード名 | `END`）。
            candidates: `mapping=None`（動的にノード名/リストを返す）時に、可能な行き先を宣言
                するリスト。validate の到達性・可視化に使う（LangGraph の path_map 相当）。

        Returns:
            自身（メソッドチェーン用）。

        Raises:
            ValueError: 同一 src に条件エッジを 2 度宣言した場合（黙示上書きを防ぐ）、または
                `mapping=None` かつ `default` を指定した場合（mapping=None では router の戻り値を
                直接行き先にするため default は参照されない・黙殺を fail-fast で弾く）。
            WorkflowFrozenError: ``freeze()`` 後の呼び出し。
        """
        self._ensure_unfrozen(f"add_conditional_edges (src={src!r})")
        if src in self.conditional_edges:
            raise ValueError(
                f"workflow {self.name!r}: 条件エッジが重複しています: src={src!r}"
                "（同一ノードからの条件エッジは 1 本のみ・黙示上書きはしません）"
            )
        if mapping is None and default is not None:
            raise ValueError(
                f"workflow {self.name!r}: 条件エッジ {src!r} は mapping=None では default を"
                "指定できません（router の戻り値を直接行き先にするため default は参照されません）"
            )
        self.conditional_edges[src] = ConditionalEdge(
            src=src,
            router=router,
            mapping=None if mapping is None else dict(mapping),
            default=default,
            candidates=list(candidates) if candidates is not None else None,
        )
        return self

    def add_fan_in_edge(self, sources: list[str], dst: str) -> WorkflowGraph:
        """fan-in（合流）エッジを張る（合流先は FUNCTION 必須・自身を返す・C-4/FR-1）。

        dst は全ソース完了後に `{source名: 出力}` の dict を msg として受ける（位置依存
        list は廃止・名前で読む）。合流先は FUNCTION ノードでなければならない
        （AGENT を合流先にはできない・validate で検査）。

        Args:
            sources: 合流するソースノード名のリスト。
            dst: 合流先 FUNCTION ノード名。

        Returns:
            自身（メソッドチェーン用）。

        Raises:
            ValueError: 同一 dst に fan-in を 2 度宣言した場合（黙示上書きを防ぐ）。
            WorkflowFrozenError: ``freeze()`` 後の呼び出し。
        """
        self._ensure_unfrozen(f"add_fan_in_edge (dst={dst!r})")
        if dst in self.fan_in_edges:
            raise ValueError(
                f"workflow {self.name!r}: fan-in が重複しています: dst={dst!r}"
                "（同一合流先への fan-in は 1 本のみ・黙示上書きはしません）"
            )
        self.fan_in_edges[dst] = FanInEdge(sources=list(sources), dst=dst)
        return self

    # ------------------------------------------------------------------
    # 凍結
    # ------------------------------------------------------------------
    def freeze(self) -> None:
        """以降のノード／エッジ追加を禁止する。

        ``add_agent_node`` / ``add_function_node`` / ``add_edge`` /
        ``add_conditional_edges`` / ``add_fan_in_edge`` で ``WorkflowFrozenError`` を
        raise するようになる。``validate`` / ``mermaid`` / ``_interpret`` /
        ``as_agent_spec`` / ``as_facade_spec`` / ``connect_as_facade`` 等の read-only
        API は影響を受けない。

        freeze 時点で ``nodes`` / ``edges`` / ``conditional_edges`` / ``fan_in_edges``
        を ``MappingProxyType`` で read-only view に置換し、外部参照経由の dict mutation
        （``wf.nodes['evil'] = ...`` / ``wf.edges.clear()`` 等）も遮断する。``edges``
        の値（``list[Any]``）は ``tuple`` に変換し ``.append`` / ``.clear`` 等のリスト
        mutation も遮断する（interpreter / validate / mermaid は iterate するだけのため
        互換）。冪等で、複数回呼んでも 2 回目以降は no-op として成功する（snapshot は
        1 回目のみ。``ConditionalEdge`` / ``FanInEdge`` の内部 dataclass field mutation
        までは scope 外で follow-up とする）。
        """
        if self._frozen:
            return
        self.nodes = MappingProxyType(dict(self.nodes))  # type: ignore[assignment]
        self.edges = MappingProxyType(  # type: ignore[assignment]
            {k: tuple(v) for k, v in self.edges.items()}
        )
        self.conditional_edges = MappingProxyType(dict(self.conditional_edges))  # type: ignore[assignment]
        self.fan_in_edges = MappingProxyType(dict(self.fan_in_edges))  # type: ignore[assignment]
        self._frozen = True

    def _ensure_unfrozen(self, operation: str) -> None:
        """frozen workflow に対する変更操作なら ``WorkflowFrozenError`` を raise する。

        Args:
            operation: 違反した変更操作名（``add_edge (src=..., dst=...)`` 等。引数情報を
                含めるものは呼び出し側で整形して渡す）。

        Raises:
            WorkflowFrozenError: ``freeze()`` 後の場合。
        """
        if self._frozen:
            raise WorkflowFrozenError(f"frozen workflow に対する変更操作: {operation}")

    # ------------------------------------------------------------------
    # 検査・可視化
    # ------------------------------------------------------------------
    def validate(self, registry: AgentRegistry) -> None:
        """build-time 検証を行い、誤りを集約報告する（FR-6/NFR-5）。

        検査項目: (a) 全エッジ端点と AGENT ノードの参照解決（未登録ノード名 /
        未登録 agent 名）、(b) `START` からのエッジが 1 本（単一エントリ）・`START` から
        各ノードへの到達性および `END` への到達可能性、(c) 条件エッジ mapping 全分岐先の
        解決、(d) fan-in 全ソース解決および合流先が FUNCTION ノードであること、(e)
        `recursion_limit` の存在、(f) 通常エッジ fan-out の合流先二重実行（fan-in 未宣言）と
        複数枝の独立 `END` 到達。型整合は初版対象外（C-6）。validate が通っても実値の
        型保証はしない。

        Args:
            registry: AGENT ノード名を解決する AgentRegistry。

        Raises:
            ValueError: 検証エラーが 1 つ以上ある場合（全件を列挙）。
        """
        problems: list[str] = []
        known_agents = set(registry.names())

        # (b-1) 単一エントリ。
        if self.entry is None:
            problems.append("START からのエッジ（エントリ）が未設定です")
        elif self.entry is not END and self.entry not in self.nodes:
            problems.append(f"エントリノード {self.entry!r} が未登録です")

        # (a) AGENT ノードの registry 参照解決。
        for node in self.nodes.values():
            if node.kind is NodeKind.AGENT and node.agent not in known_agents:
                problems.append(
                    f"AGENT node {node.name!r} の agent 参照 {node.agent!r} が registry 未登録"
                )

        # (a/c/d) エッジ端点の参照解決と fan-in 合流先の FUNCTION 制約。
        problems.extend(self._validate_edge_refs())

        # (f) fan-out 枝の通常エッジ合流（fan-in 未宣言の二重実行）/ 複数枝の独立 END 到達。
        problems.extend(self._validate_convergence())

        # (e) recursion_limit の存在。
        if self.recursion_limit < 1:
            problems.append(f"recursion_limit は 1 以上が必須です（現在 {self.recursion_limit}）")

        # (b-2) START からの到達性 / END 到達可能性。
        if self.entry is not None:
            reachable = self._reachable()
            unreached = set(self.nodes) - reachable
            for name in sorted(unreached):
                problems.append(f"node {name!r} が START から到達不能です")
            if END not in self._reachable(include_end=True):
                problems.append("START から END へ到達できません")

        if problems:
            raise ValueError(f"workflow {self.name!r} の検証エラー: " + "; ".join(problems))

    def _validate_edge_refs(self) -> list[str]:
        """エッジ端点が参照するノード名の解決と fan-in 合流先制約を検査する。"""
        problems: list[str] = []
        for src, dsts in self.edges.items():
            if src not in self.nodes:
                problems.append(f"エッジの始点 node {src!r} が未登録です")
            for dst in dsts:
                if dst is not END and dst not in self.nodes:
                    problems.append(f"エッジの終点 node {dst!r} が未登録です")
        for cond in self.conditional_edges.values():
            if cond.src not in self.nodes:
                problems.append(f"条件エッジの始点 node {cond.src!r} が未登録です")
            # mapping=None（router がノード名を直接返す）は静的に分岐先を検査できない。
            for key, dst in (cond.mapping or {}).items():
                for target in _as_targets(dst):  # 値はノード名・END・それらのリスト
                    if target is not END and target not in self.nodes:
                        problems.append(
                            f"条件エッジ {cond.src!r} の分岐先 {target!r}"
                            f"（key={key!r}）が未登録です"
                        )
            default = cond.default
            if default is not None and default is not END and default not in self.nodes:
                problems.append(
                    f"条件エッジ {cond.src!r} の default 分岐先 {default!r} が未登録です"
                )
            for cand in cond.candidates or []:
                if cand is not END and cand not in self.nodes:
                    problems.append(
                        f"条件エッジ {cond.src!r} の candidates の {cand!r} が未登録です"
                    )
        normal_targets = {dst for dsts in self.edges.values() for dst in dsts}
        for fan_in in self.fan_in_edges.values():
            for source in fan_in.sources:
                if source not in self.nodes:
                    problems.append(f"fan-in のソース node {source!r} が未登録です")
            dst_node = self.nodes.get(fan_in.dst)
            if dst_node is None:
                problems.append(f"fan-in の合流先 node {fan_in.dst!r} が未登録です")
            elif dst_node.kind is not NodeKind.FUNCTION:
                problems.append(
                    f"fan-in の合流先 node {fan_in.dst!r} は FUNCTION ノードが必須です（C-4）"
                )
            if fan_in.dst in normal_targets:
                problems.append(
                    f"fan-in の合流先 node {fan_in.dst!r} が通常エッジの終点も兼ねています"
                    "（fan-in では msg が dict、通常流入では単一出力となり曖昧化します・C-4）"
                )
        return problems

    def _validate_convergence(self) -> list[str]:
        """通常エッジ fan-out が起こす「合流先の二重実行」「複数枝の独立 END 到達」を検出する。

        通常エッジは無条件に発火するため、fan-out（同一ノードから 2 本以上の通常エッジ）の異なる
        枝が同一ノードへ通常エッジで合流すると、その合流先が並行に二重実行される（fan-in 宣言で
        合流すべき・C-4 違反）。また複数枝が fan-in を介さず独立に `END` へ到達すると、最終出力は
        宣言順で先頭枝のみ採用され他枝出力が黙って破棄される。いずれも build-time で弾く。

        条件エッジ（排他的分岐）やループ（条件エッジで戻る）は通常エッジの fan-out ではないため
        対象外（誤検出しない）。fan-in 合流先・起点ノードで探索を打ち切り、ループ越えの誤検出も
        避ける。

        Returns:
            検出した問題メッセージ（重複排除済み・ソート済み）。
        """
        problems: set[str] = set()
        fan_in_dsts = set(self.fan_in_edges)
        for src, dsts in self.edges.items():
            node_branches = {d for d in dsts if d is not END and d in self.nodes}
            end_branches = sum(1 for d in dsts if d is END)
            if len(node_branches) + end_branches < 2:
                continue  # 出口が 1 本以下なら fan-out ではない
            blockers = fan_in_dsts | {src}
            counts: Counter[str] = Counter()
            for branch in node_branches:
                reach, hit_end = self._normal_reach(branch, blockers)
                counts.update(reach)
                if hit_end:
                    end_branches += 1
            for name, c in counts.items():
                if c >= 2 and name not in fan_in_dsts:
                    problems.add(
                        f"node {name!r} が fan-out {src!r} の複数枝から通常エッジで合流しますが "
                        "fan-in 宣言がありません（合流先が並行に二重実行されます。add_fan_in_edge "
                        "で合流してください・C-4）"
                    )
            if end_branches >= 2:
                problems.add(
                    f"fan-out {src!r} の複数枝が fan-in を介さず END へ到達します（最終出力は宣言"
                    "順で先頭枝のみ採用され他枝出力は破棄されます。END 手前で add_fan_in_edge で"
                    "合流してください）"
                )
        return sorted(problems)

    def _normal_reach(self, start: Any, blockers: set[Any]) -> tuple[set[str], bool]:
        """start から通常エッジ（self.edges）のみで到達するノード集合と END 到達有無を返す。

        `blockers`（fan-in 合流先 + fan-out 起点）に達したら打ち切る（合流の解決点とループ越えを
        探索しない）。条件エッジは辿らない（排他的分岐を並行合流と誤判定しないため）。

        Args:
            start: 起点ノード名。
            blockers: 探索を打ち切るノード名集合。

        Returns:
            (到達ノード名集合, START から END へ到達したか) のタプル。
        """
        seen: set[str] = set()
        hit_end = False
        stack: list[Any] = [start]
        while stack:
            cur = stack.pop()
            if cur is END:
                hit_end = True
                continue
            if cur in seen or cur not in self.nodes or cur in blockers:
                continue
            seen.add(cur)
            stack.extend(self.edges.get(cur, []))
        return seen, hit_end

    def _reachable(self, *, include_end: bool = False) -> set[Any]:
        """START（entry）から全エッジを辿って到達可能なノード名集合を返す。"""
        visited: set[Any] = set()
        if self.entry is None:
            return visited
        stack: list[Any] = [self.entry]
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            if current is END or current not in self.nodes:
                continue
            stack.extend(self._successors(current))
        if not include_end:
            visited.discard(END)
        return visited

    def _successors(self, name: str) -> list[Any]:
        """name のエッジ後続ノード名を列挙する（到達性解析用・END を含みうる）。"""
        succ: list[Any] = list(self.edges.get(name, []))
        if name in self.conditional_edges:
            cond = self.conditional_edges[name]
            for dst in (cond.mapping or {}).values():
                succ.extend(_as_targets(dst))  # 値はリストでも単一でも展開
            # mapping=None の動的返しは candidates で宣言された行き先を到達性に含める。
            if cond.candidates is not None:
                succ.extend(cond.candidates)
            if cond.default is not None:
                succ.append(cond.default)
        for fan_in in self.fan_in_edges.values():
            if name in fan_in.sources:
                succ.append(fan_in.dst)
        return succ

    def mermaid(self) -> str:
        """ノードとエッジ（通常 / 条件 / fan-in）+ START/END を Mermaid 文字列で返す（FR-7）。

        条件エッジは判定キーをラベルに、fan-in は破線で合流を示す。

        Returns:
            Mermaid flowchart 形式の文字列。
        """
        lines = ["flowchart TD", "    START([START])", "    END([END])"]
        for node in self.nodes.values():
            if node.kind is NodeKind.FUNCTION:
                lines.append(f"    {node.name}({node.name})")
            else:
                lines.append(f"    {node.name}[{node.name}]")
        if self.entry is not None:
            lines.append(f"    START --> {self._label(self.entry)}")
        for src, dsts in self.edges.items():
            for dst in dsts:
                lines.append(f"    {src} --> {self._label(dst)}")
        for cond in self.conditional_edges.values():
            if cond.mapping is None:
                # 動的（戻り値直接）。candidates があれば各候補へ破線で描く。
                if cond.candidates:
                    for cand in cond.candidates:
                        lines.append(f"    {cond.src} -.-> {self._label(cand)}")
                else:
                    lines.append(f"    {cond.src} -.->|?| {self._label(END)}")
            else:
                for key, dst in cond.mapping.items():
                    for target in _as_targets(dst):  # 条件 fan-out は複数本のエッジで描く
                        lines.append(f"    {cond.src} -->|{key}| {self._label(target)}")
            if cond.default is not None:
                lines.append(f"    {cond.src} -->|default| {self._label(cond.default)}")
        for fan_in in self.fan_in_edges.values():
            for source in fan_in.sources:
                lines.append(f"    {source} -.-> {fan_in.dst}")
        return "\n".join(lines)

    @staticmethod
    def _label(node: Any) -> str:
        """ノード名 または END 番兵を Mermaid ノード ID 文字列へ変換する。"""
        return "END" if node is END else str(node)

    # ------------------------------------------------------------------
    # 内部インタプリタ（agents 非依存・runner シームへ委譲）
    # 実体は `_interpreter` のフリー関数。本クラスのメソッドは薄く委譲する。
    # ------------------------------------------------------------------
    async def _interpret(
        self,
        runner: RunnerSeam,
        input: Any,
        *,
        context: Any = None,
        on_node_start: NodeHook | None = None,
        on_node_end: NodeHook | None = None,
    ) -> WorkflowResult:
        """グラフを解釈してワークフローを 1 回実行する（`_interpreter.interpret` へ委譲）。"""
        return await _interpreter.interpret(
            self,
            runner,
            input,
            context=context,
            on_node_start=on_node_start,
            on_node_end=on_node_end,
        )

    def _activation_closure(self, start: Any) -> set[str]:
        """start から決定的エッジ（通常エッジ + fan-in 合流先）だけで到達するノード集合を返す。

        fan-out 起動時に各枝の前方閉包を activated へ先行登録するために使う。多段 fan-in で
        浅い枝が深い fan-in 合流先へ先着しても、深い枝側の fan-in ソースが未起動と誤判定され
        required が過小評価されて早期発火するレースを防ぐ。条件エッジは実行時にどの枝へ進むか
        不定なため展開しない（その分岐先は従来どおり遅延 activated に委ねる）。

        Args:
            start: 閉包の起点ノード名。

        Returns:
            起点から決定的エッジで到達できるノード名集合（END は含めない）。
        """
        closure: set[str] = set()
        stack: list[Any] = [start]
        while stack:
            cur = stack.pop()
            if cur is END or cur in closure or cur not in self.nodes:
                continue
            closure.add(cur)
            stack.extend(self.edges.get(cur, []))
            for fan_in in self.fan_in_edges.values():
                if cur in fan_in.sources:
                    stack.append(fan_in.dst)
        return closure

    def _has_fan_out(self) -> bool:
        """通常エッジに fan-out（1 ノードから複数下流）が存在するかを返す。"""
        return any(len(dsts) > 1 for dsts in self.edges.values())

    # ------------------------------------------------------------------
    # Agent / Tool 化（実体は `_facade` のフリー関数。本クラスのメソッドは薄く委譲する）
    # ------------------------------------------------------------------
    def as_agent_spec(
        self,
        name: str,
        *,
        registry: AgentRegistry | None = None,
        output_extractor: Callable[[Any], str] | None = None,
        on_node_start: NodeHook | None = None,
        on_node_end: NodeHook | None = None,
    ) -> AgentSpec:
        """WorkflowModel を据えた AgentSpec を返す（経路C・主軸・FR-8）。

        registry.register すると WorkflowGraph が「本物の Agent」として保持され、handoff の
        直接ターゲットになれる。WorkflowModel が LLM を呼ばずエンジンを回すため決定論的に
        起動する。外側 run の共有 context はワークフロー内ステップへ伝播しない（C-11）。
        加えて先頭ノードへ渡るのは入力中の **末尾 user テキスト 1 件のみ**で、会話履歴や
        system_instructions は伝播しない（マルチターンで過去履歴を内部参照したい用途には
        不向き）。context や履歴が必要な場合は `as_facade_spec`（経路A/D）を使う。決定論を
        保ったまま context 透過したい場合は `as_facade_spec(mode=FacadeMode.DETERMINISTIC)`
        （経路D・実 LLM 0 回）。経路C 固有の利点は「外から 1 Agent・handoff 直接ターゲット・
        tool 往復を挟まない」点。

        Args:
            name: 生成する AgentSpec の名前。
            registry: AGENT ノード名を SDK Agent へ解決する AgentRegistry。None の場合は
                runner シームへ名前がそのまま渡る（fake runner で解決するテスト等）。
            output_extractor: 最終出力を単一メッセージ文字列へ変換する関数。None で既定
                （str 化）。型整合は後続（OQ-1）。
            on_node_start: ノード実行前フック（任意）。
            on_node_end: ノード実行後フック（任意）。

        Returns:
            model に WorkflowModel を据えた AgentSpec（tools / handoffs なし）。
        """
        return _facade.build_agent_spec(
            self,
            name,
            registry=registry,
            output_extractor=output_extractor,
            on_node_start=on_node_start,
            on_node_end=on_node_end,
        )

    def as_facade_spec(
        self,
        name: str,
        *,
        registry: AgentRegistry | None = None,
        mode: FacadeMode = FacadeMode.LLM_INPUT,
        model: Any = None,
        tool_name: str | None = None,
        tool_description: str | None = None,
        output_extractor: Callable[[Any], str] | None = None,
        on_node_start: NodeHook | None = None,
        on_node_end: NodeHook | None = None,
    ) -> AgentSpec:
        """ワークフロー tool だけを持つファサード AgentSpec を返す（経路A/D・FR-9）。

        外側の共有 context をワークフロー内ステップへ透過する（context を渡せない経路C との
        差別化点）。入口モデルは `mode` で切り替える（`FacadeMode` 参照）:

        - `LLM_INPUT`（既定・従来の経路A）: 実 LLM が tool 入力を整形して 1 回呼び、結果を
          素通しする（`stop_on_first_tool`）。
        - `LLM_INPUT_OUTPUT`: 実 LLM が入力整形に加え tool 結果も要約する（実 LLM 2 回）。
          `stop_on_first_tool` を付けず、SDK の `reset_tool_choice` 既定で 2 ターン目の無限
          ツール呼び出しを防ぐ。
        - `DETERMINISTIC`（経路D）: 決定論ステートレスモデルを入口に据え、実 LLM を呼ばずに
          毎回ワークフロー tool を強制発火する（決定論・実 LLM 0 回・入力は素通し）。

        `tool_choice='required'` は全 mode で model_settings に設定する（extra 不可・FR-9）。
        各 AGENT ノード内側 run の暴走上限（max_turns）等の Runner kwarg は、グラフ既定の
        `run_defaults` またはノード `run_options` で設定する（passthrough・FR-15）。

        本メソッドはファサード AgentSpec のみを返す。handoff 流入時の既定 input_filter
        （直近 1 件・C-10）は handoff エッジに載せる必要があるため、registry 登録 + エッジ
        結線まで行う `connect_as_facade` を使うこと（input_filter を facade 自身の
        handoff_options に載せても registry は読まないため）。

        Args:
            name: 生成する AgentSpec の名前。
            registry: AGENT ノード名を SDK Agent へ解決する AgentRegistry（任意）。
            mode: 入口モデル種別（`FacadeMode`。既定 `LLM_INPUT`）。
            model: 入口に据える Model の明示指定（LLM 系 mode で使用）。None だと LLM 系 mode は
                SDK 既定モデル（実 API 接続が必要）へフォールバックする。実 LLM を呼ばず offline で
                動かしたい場合は `mode=DETERMINISTIC`（経路D）を使う。`DETERMINISTIC` では決定論
                モデルを内部注入するため非 None 指定は ValueError。
            tool_name: ワークフロー tool 名（None で name 由来）。
            tool_description: ワークフロー tool の説明（任意）。
            output_extractor: 最終出力を文字列化する関数（任意）。
            on_node_start: ノード実行前フック（任意）。
            on_node_end: ノード実行後フック（任意）。

        Returns:
            ワークフロー tool・tool_choice='required' を持つ AgentSpec（mode により入口モデルと
            stop_on_first_tool の有無が変わる）。

        Raises:
            ValueError: `mode=DETERMINISTIC` かつ `model` を指定した場合（model は無視されるため
                誤用を fail-fast で弾く）。または `mode` が未知の値の場合。
        """
        return _facade.build_facade_spec(
            self,
            name,
            registry=registry,
            mode=mode,
            model=model,
            tool_name=tool_name,
            tool_description=tool_description,
            output_extractor=output_extractor,
            on_node_start=on_node_start,
            on_node_end=on_node_end,
        )

    def connect_as_facade(
        self,
        registry: AgentRegistry,
        graph: HandoffGraph,
        name: str,
        src: str,
        *,
        input_filter: Any = _UNSET,
        description: str | None = None,
        mode: FacadeMode = FacadeMode.LLM_INPUT,
        model: Any = None,
        tool_name: str | None = None,
        tool_description: str | None = None,
        output_extractor: Callable[[Any], str] | None = None,
        on_node_start: NodeHook | None = None,
        on_node_end: NodeHook | None = None,
    ) -> AgentSpec:
        """経路A/D の結線一式: ファサードを registry 登録し `src -> facade` の handoff を張る。

        既定 input_filter（直近 1 件）は handoff エッジ（registry が実際に読む場所）に載せ、
        流入履歴をコード既定で有界化する（C-10/FR-11）。明示 `input_filter=None` で全履歴
        流入（opt-in）。`graph.apply(registry)` は呼び出し側で行う。入口モデルは `mode` で
        切り替える（`FacadeMode` 参照。`DETERMINISTIC` で実 LLM 0 回の決定論流入になる）。

        Args:
            registry: ファサードを登録する AgentRegistry。
            graph: handoff エッジを張る HandoffGraph。
            name: ファサード AgentSpec 名（handoff 流入先）。
            src: 流入元エージェント名（`src -> name` のエッジを張る）。
            input_filter: handoff エッジの input_filter。未指定で既定（直近 1 件）、明示
                None で全履歴流入。
            description: handoff tool の説明（任意）。
            mode / model / tool_name / tool_description / output_extractor / on_node_start /
                on_node_end: `as_facade_spec` へ素通し。

        Returns:
            登録したファサード AgentSpec。

        Raises:
            RegistryFrozenError: ``registry`` が ``freeze()`` 済みの場合（内部で
                ``registry.register(spec)`` を呼ぶため）。lockdown 後の registry に
                ``connect_as_facade`` を適用したい場合は freeze 前に呼び出すか、別
                registry でファサードを構築する。
        """
        return _facade.connect_facade(
            self,
            registry,
            graph,
            name,
            src,
            input_filter=input_filter,
            description=description,
            mode=mode,
            model=model,
            tool_name=tool_name,
            tool_description=tool_description,
            output_extractor=output_extractor,
            on_node_start=on_node_start,
            on_node_end=on_node_end,
        )
