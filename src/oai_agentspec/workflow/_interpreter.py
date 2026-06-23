"""ワークフローの内部インタプリタ（agents 非依存・runner シームへ委譲）。

メッセージ / エッジ駆動でグラフを 1 回実行する内部エンジン。`_RunState`（実行スコープの可変
状態）と、`WorkflowGraph` のインタプリタ系メソッドが委譲するフリー関数群（`interpret` /
`drive` / `next_nodes` / `exec_node` / fan-in / fan-out 等）を提供する。`WorkflowGraph` の各
インタプリタ系メソッドは本モジュールのフリー関数へ薄く委譲する（実体をここに集約）。SDK には
依存しない（NFR-1）。

tracing: 任意の `WorkflowTracer`（`_adapters/tracing.py` 由来・workflow 層は Protocol だけ
を参照）を `interpret(..., tracer=...)` で受け取り、workflow span / node span /
condition span / fan-out span / fan-in span を with でラップする。tracer 未指定時は
モジュール内 `_NULL_TRACER`（span を発行しないシングルトン）に正規化するため、span 発行と
agents 非依存性を両立する。
"""

from __future__ import annotations

import asyncio
import inspect
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ._declarations import NodeResults, RunnerSeam, WorkflowResult
from ._types import _PENDING, END, NodeHook, NodeKind

if TYPE_CHECKING:
    from collections.abc import Iterator

    from .._adapters import WorkflowTracer
    from .graph import WorkflowGraph

__all__: list[str] = []


class _NullWorkflowTracer:
    """span を発行しない workflow 層内蔵の null tracer（tracer 未指定時の正規化先）。

    `_adapters.tracing._NoopWorkflowTracer` と同じ挙動だが、workflow 層の agents 非依存
    （NFR-1）を完全に保つため本ファイル内に閉じて定義する（`_adapters` を遅延 import しない）。
    全メソッドが `@contextmanager` で yield のみを返し span オブジェクトを生成しない。
    """

    @contextmanager
    def workflow_span(self, graph_name: str) -> Iterator[None]:
        """no-op（span を発行しない）。"""
        yield None

    @contextmanager
    def node_span(self, node_name: str, kind: str) -> Iterator[None]:
        """no-op（span を発行しない）。"""
        yield None

    @contextmanager
    def condition_span(self, src: str) -> Iterator[None]:
        """no-op（span を発行しない）。"""
        yield None

    @contextmanager
    def fan_out_span(self, src: str) -> Iterator[None]:
        """no-op（span を発行しない）。"""
        yield None

    @contextmanager
    def fan_in_span(self, dst: str) -> Iterator[None]:
        """no-op（span を発行しない）。"""
        yield None


# tracer 未指定時の正規化先（プロセス共有・状態を持たない）。
_NULL_TRACER: Any = _NullWorkflowTracer()


@dataclass
class _RunState:
    """1 回の実行スコープのみ保持する可変状態（run 終了で破棄・NFR-3）。

    Attributes:
        runner: AGENT ノード実行を委譲する runner シーム。
        context: 各ノードへ素通しする共有 context。
        on_node_start: ノード実行前フック。
        on_node_end: ノード実行後フック。
        tracer: span 発行を行う `WorkflowTracer`（未指定時は `_NULL_TRACER`）。
        activated: 実行時に起動（スケジュール）されたノード名の集合。条件 fan-out で実際に
            走った枝だけを動的 fan-in の待ち対象にするために使う。
        frontier: いま `exec_node` 実行中（await 中）のノード名の多重集合。fan-in 充足判定で
            「まだ実行中の枝から到達しうるソースは未到達でも待つ」ために使う（条件エッジ経由の
            深い fan-in ソースを浅い枝が先着しても取りこぼさない・WF-CONC-01 対策）。
        results: ノード名 -> 出力 の薄い記録。
        fan_in_arrivals: fan-in 合流先名 -> これまでの到達数。
        steps: 遷移回数（recursion_limit の判定に使う）。
    """

    runner: RunnerSeam
    context: Any
    on_node_start: NodeHook | None
    on_node_end: NodeHook | None
    tracer: Any = field(default_factory=lambda: _NULL_TRACER)
    activated: set[str] = field(default_factory=set)
    frontier: Counter[str] = field(default_factory=Counter)
    results: NodeResults = field(default_factory=NodeResults)
    fan_in_arrivals: dict[str, int] = field(default_factory=dict)
    steps: int = 0

    def tick(self) -> None:
        """遷移カウンタを 1 進める（recursion_limit 判定用）。"""
        self.steps += 1


async def interpret(
    graph: WorkflowGraph,
    runner: RunnerSeam,
    input: Any,
    *,
    context: Any = None,
    on_node_start: NodeHook | None = None,
    on_node_end: NodeHook | None = None,
    tracer: WorkflowTracer | None = None,
) -> WorkflowResult:
    """グラフを解釈してワークフローを 1 回実行する（run ごとに状態生成・破棄）。

    メッセージ / エッジ駆動。START から開始し、各ノードの出力を出辺に沿って下流へ
    流す。fan-out（複数通常エッジ）は asyncio.gather で並行実行し、fan-in の合流先は
    全ソース完了まで待ってから `{source名: 出力}` の dict で進む。条件エッジは router の
    判定キーで 1 経路を選ぶ。ループ防止に遷移回数を数え `recursion_limit` 超過で例外
    （C-5）。AGENT ノードは runner シームへ委譲、FUNCTION ノードは sync / async 両対応で
    呼ぶ。並列 + session は fail-fast で拒否する（FR-15）。

    tracing: `tracer` が渡されると workflow / node / condition / fan-out / fan-in の各 span
    を with でラップして発行する。未指定時はモジュール内 `_NULL_TRACER`（span を発行しない
    シングルトン）に正規化するため、tracer 引数を渡さない既存呼び出しは挙動が変わらない。

    Args:
        graph: 解釈対象の WorkflowGraph。
        runner: AGENT ノード実行を委譲する runner シーム。
        input: ワークフローへの初期入力（START 直後のノードの msg）。
        context: 各ノードへ素通しする共有 context（経路A 時のみ非 None・C-11）。
        on_node_start: ノード実行前フック（任意）。
        on_node_end: ノード実行後フック（任意）。
        tracer: 任意の `WorkflowTracer`（None で span 発行を完全にスキップする no-op に倒す）。

    Returns:
        WorkflowResult（END へ到達したノードの出力 + NodeResults）。

    Raises:
        ValueError: entry 未設定 / 並列+session 競合 / 条件分岐キー欠落 /
            recursion_limit 超過 / 未登録ノード実行 の場合。
    """
    if graph.entry is None:
        raise ValueError(f"workflow {graph.name!r}: START からのエントリが未設定です")
    if graph.session is not None and graph._has_fan_out():
        raise ValueError(
            f"workflow {graph.name!r}: fan-out（並列）と session は併用できません"
            "（SDK Session の add_items は並行安全を保証しません・FR-15）"
        )

    state = _RunState(
        runner=runner,
        context=context,
        on_node_start=on_node_start,
        on_node_end=on_node_end,
        tracer=tracer if tracer is not None else _NULL_TRACER,
    )
    if graph.entry is not END:
        state.activated.add(graph.entry)
    with state.tracer.workflow_span(graph.name):
        final_output = await drive(graph, graph.entry, input, state)
    # 内部 sentinel が最終出力へ到達したら（fan-in 未充足のまま全枝が離脱した等）、利用者や
    # LLM へ漏らさず fail-fast する（sentinel 漏出の恒久遮断・WF-CONC-02）。
    if final_output is _PENDING:
        raise ValueError(
            f"workflow {graph.name!r}: fan-in が未充足のまま実行が終了しました"
            "（条件エッジ経由の fan-in ソース取りこぼし等。トポロジを見直してください）"
        )
    return WorkflowResult(final_output=final_output, results=state.results)


async def drive(graph: WorkflowGraph, node: Any, msg: Any, state: _RunState) -> Any:
    """node を起点にエッジを辿りワークフローを駆動し END 到達時の出力を返す。

    メッセージ駆動の再帰下降。`node` が fan-in 合流先のときは到達を 1 件記録し、
    全ソースが揃った最後の到達のみが合流先 FUNCTION を実行して下流へ進む（他の到達は
    `_PENDING` を返す）。fan-out（複数通常エッジ）は下流枝を `asyncio.gather` で並行に
    END まで辿り、`_PENDING` でない最初の出力を返す。条件エッジは router の判定キーで
    1 経路を選ぶ。各ノード実行ごとに遷移回数を数え `recursion_limit` 超過で例外（C-5）。
    """
    while True:
        if node is END:
            return msg

        # fan-in 合流先: 自分の到達を記録し、未充足なら離脱する。充足したら合流入力を
        # 集めた後、当該ソースを activated から外してループ反復間の累積を断つ。span は
        # 「合流が成立した側のみ」を fan_in_span で包む（早期離脱は span を発行しない）。
        if node in graph.fan_in_edges:
            if not _fan_in_ready(graph, node, state):
                return _PENDING
            with state.tracer.fan_in_span(node):
                msg = _collect_fan_in(graph, node, state)
                for s in graph.fan_in_edges[node].sources:
                    state.activated.discard(s)

        state.tick()
        if state.steps > graph.recursion_limit:
            raise ValueError(
                f"workflow {graph.name!r}: recursion_limit={graph.recursion_limit} を"
                "超過しました（1 run の総ノード実行数の上限。無限ループ防止を兼ねるため、"
                "ループの無い大きなグラフでは recursion_limit を引き上げてください・C-5）"
            )

        # frontier に積んでから実行する。await 中に別枝の fan-in 充足判定が走ったとき、
        # 本ノードから（条件エッジ含め）到達しうる未到達ソースを「まだ来る」と数えさせる。
        state.frontier[node] += 1
        try:
            output = await exec_node(graph, node, msg, state)
        finally:
            state.frontier[node] -= 1
            if state.frontier[node] <= 0:
                del state.frontier[node]

        next_targets = next_nodes(graph, node, output, state)
        if not next_targets:
            # 出辺が無いノード（END へ繋がない宣言）の出力は最終出力候補にしない。
            return output
        # 起動する枝を「駆動前」にまとめて activated へ登録する（同期ノードでも動的
        # fan-in のカウントが安定するように・条件 fan-out の部分集合に対応）。
        for nxt in next_targets:
            if nxt is not END:
                state.activated.add(nxt)
        if len(next_targets) == 1:
            node, msg = next_targets[0], output
            continue
        # fan-out（条件 fan-out で実行時に複数枝になる場合を含む）+ session は不可。
        # run-entry の静的ガード（_has_fan_out）は通常エッジ由来の fan-out しか見ないため、
        # 条件 fan-out をここで実行時に fail-fast する（Session.add_items は並行非安全）。
        if graph.session is not None:
            raise ValueError(
                f"workflow {graph.name!r}: fan-out（並列）と session は併用できません"
                "（条件 fan-out 含む・Session.add_items は並行安全を保証しません・FR-15）"
            )
        # 多段 fan-in の取りこぼし防止: 各枝の前方閉包（決定的エッジのみ）を駆動前に
        # 先行 activated 化し、浅い枝が深い fan-in に先着しても required が過小評価され
        # ないようにする（activated 登録レース対策）。
        for dst in next_targets:
            if dst is not END:
                state.activated |= graph._activation_closure(dst)
        # fan-out: 各下流枝を並行に END まで辿る。span は asyncio.gather を囲む
        # （並列起動の親 span として、各子枝の node span を兄弟に並べる）。
        with state.tracer.fan_out_span(node):
            branch_outputs = await asyncio.gather(
                *(drive(graph, dst, output, state) for dst in next_targets)
            )
        for out in branch_outputs:
            if out is not _PENDING:
                return out
        return _PENDING


def next_nodes(graph: WorkflowGraph, node: str, output: Any, state: _RunState) -> list[Any]:
    """node 実行後の次ノード列を決める（条件エッジ優先・通常 fan-out・fan-in 合流先）。

    node が fan-in のソースなら合流先 dst を後続に含める（fan-in エッジは通常エッジと
    独立に宣言されるため・C-4）。

    条件エッジの場合は `_select_conditional` 呼び出しのみを `condition_span` で包む
    （分岐評価の所要時間と選択結果のみが span に乗り、後続枝の実行は別 span として現れる）。
    """
    if node in graph.conditional_edges:
        with state.tracer.condition_span(node):
            result = _select_conditional(graph, node, output, state.context)
        # 条件 fan-out: router/mapping がノード名のリストを返したら複数を並行起動する。
        return list(result) if isinstance(result, list) else [result]
    succ: list[Any] = list(graph.edges.get(node, []))
    for fan_in in graph.fan_in_edges.values():
        if node in fan_in.sources:
            succ.append(fan_in.dst)
    return succ


def _fan_in_ready(graph: WorkflowGraph, node: str, state: _RunState) -> bool:
    """fan-in 合流先への到達を 1 件記録し、まだ来るソースが無くなったら True を返す。

    待ち対象（required）は「来ることが分かっているソース」= 実行時に起動された
    （activated な）ソース、または いま実行中（frontier）のノードから静的に到達しうる
    ソース。後者により、条件エッジ経由でしか到達しない深い fan-in ソースを浅い枝が先着
    しても早期発火しない（WF-CONC-01）。解決済みの条件分岐は frontier を抜けるため、
    走らなかった枝のソースは required に数えず動的 fan-in のデッドロックも避ける。

    全ソース到達で発火する際は到達数をリセットする（合流ソースの activated 除去は
    `_collect_fan_in` 後に `drive` 側で行う。ループ反復をまたいだ累積を防ぐ・WF-CONC-02）。
    """
    sources = graph.fan_in_edges[node].sources
    reachable = _frontier_reach(graph, state)
    required = sum(1 for s in sources if s in state.activated or s in reachable)
    arrived = state.fan_in_arrivals.get(node, 0) + 1
    if arrived >= max(required, 1):
        state.fan_in_arrivals[node] = 0
        return True
    state.fan_in_arrivals[node] = arrived
    return False


def _frontier_reach(graph: WorkflowGraph, state: _RunState) -> set[str]:
    """いま実行中（frontier）のノードから静的に到達可能なノード名集合を返す。

    条件エッジ・通常エッジ・fan-in 合流先をすべて辿る過大近似（`_successors`）。fan-in の
    「まだ来るソース」判定に使う。frontier 自身（実行中ノード）も到達集合に含める。
    """
    reach: set[str] = set()
    stack: list[Any] = list(state.frontier)
    while stack:
        cur = stack.pop()
        if cur is END or cur in reach or cur not in graph.nodes:
            continue
        reach.add(cur)
        stack.extend(graph._successors(cur))
    return reach


def _select_conditional(graph: WorkflowGraph, node: str, output: Any, context: Any) -> Any:
    """条件エッジの router で次ノード名 | END を選ぶ。

    mapping=None なら戻り値を直接、ありならキーとして引く。未一致は default、無ければ例外。
    """
    cond = graph.conditional_edges[node]
    result = cond.router(output, context)
    if cond.mapping is None:
        return result  # 戻り値を次ノード名 | END として直接使う
    if result in cond.mapping:
        return cond.mapping[result]
    if cond.default is not None:
        return cond.default
    raise ValueError(
        f"workflow {graph.name!r}: 条件エッジ {node!r} のキー {result!r} が "
        f"mapping に解決せず default もありません（候補: {sorted(map(str, cond.mapping))}）"
    )


def _collect_fan_in(graph: WorkflowGraph, node: str, state: _RunState) -> dict[str, Any]:
    """fan-in 合流先に渡す `{source名: 出力}` の dict を組み立てる（C-4）。

    実際に起動された（activated な）ソースのみを含める。条件 fan-out で走らなかった枝は
    キーごと omit する（`None` を入れず「走った/走らない」を明確にする）。
    """
    fan_in = graph.fan_in_edges[node]
    return {s: state.results.outputs.get(s) for s in fan_in.sources if s in state.activated}


async def exec_node(graph: WorkflowGraph, name: str, msg: Any, state: _RunState) -> Any:
    """単一ノード（AGENT/FUNCTION）を実行し出力を記録して返す。

    tracing: ノード種別に応じた `node_span(name, "agent" | "function")` で実行と前後フックを
    包む（span 開始 → on_node_start → 実行 → on_node_end → span 終了）。フック発火は span
    スコープの内側で行う（フック契約は不変・観測の補完として並走する）。
    """
    node = graph.nodes.get(name)
    if node is None:
        hint = ""
        if name in ("END", "START"):
            # router が番兵 END を文字列 'END' として返した可能性（identity 比較を外す）。
            hint = (
                f"（番兵 {name} を文字列で返していませんか。"
                f"`from oai_agentspec import {name}` の {name} を返してください）"
            )
        raise ValueError(
            f"workflow {graph.name!r}: 未登録の node を実行しようとしました: {name!r}{hint}"
        )

    kind = "agent" if node.kind is NodeKind.AGENT else "function"
    with state.tracer.node_span(name, kind):
        if state.on_node_start is not None:
            await _maybe_await(state.on_node_start(name, state.results, state.context))

        if node.kind is NodeKind.AGENT:
            # グラフ既定 run_defaults をノード run_options で上書きマージし Runner.run へ素通し。
            # input / context は lib 管理のため merged には含まれない（予約キーで弾き済み）。
            merged = {**(graph.run_defaults or {}), **(node.run_options or {})}
            run_result = await state.runner.run(
                node.agent,
                msg,
                context=state.context,
                **merged,
            )
            output = run_result.final_output
        else:
            if node.fn is None:  # pragma: no cover - dataclass 不変条件
                raise ValueError(f"FUNCTION node {name!r} に fn がありません")
            output = await _maybe_await(node.fn(msg, state.context))

        state.results.record(name, output)

        if state.on_node_end is not None:
            await _maybe_await(state.on_node_end(name, state.results, state.context))

    return output


async def _maybe_await(value: Any) -> Any:
    """値が awaitable なら await し、そうでなければそのまま返す（sync/async 両対応）。

    Args:
        value: 同期戻り値または awaitable。

    Returns:
        await 後の値（同期値はそのまま）。
    """
    if inspect.isawaitable(value):
        return await value
    return value
