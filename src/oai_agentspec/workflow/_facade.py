"""ワークフローの Agent / Tool ファサード化ロジック（`_adapters` を関数内遅延 import・NFR-1）。

`build_agent_spec`（経路C: WorkflowModel を据えた AgentSpec）/ `build_facade_spec`（経路A/D:
ワークフロー tool を持つファサード AgentSpec）/ `connect_facade`（registry 登録 + handoff 結線）/
`default_input_filter`（流入履歴の既定有界化）を提供する。`WorkflowGraph` の `as_agent_spec` /
`as_facade_spec` / `connect_as_facade` メソッドは本モジュールのフリー関数へ薄く委譲する。

SDK 隔離（NFR-1）: `_adapters`（SDK 実体への結合点）はトップレベルではなく各関数内で遅延
import する（`workflow -> _adapters` の循環回避・遅延 import 境界(a)）。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ..constants import WORKFLOW_DEFAULT_INPUT_HISTORY_LIMIT
from ..spec import AgentSpec
from ._declarations import WorkflowResult
from ._interpreter import interpret
from ._types import _UNSET, FacadeMode, NodeHook

if TYPE_CHECKING:
    from ..handoffs import HandoffGraph
    from ..registry import AgentRegistry
    from .graph import WorkflowGraph

__all__ = [
    "default_input_filter",
]


def build_agent_spec(
    graph: WorkflowGraph,
    name: str,
    *,
    registry: AgentRegistry | None = None,
    output_extractor: Callable[[Any], str] | None = None,
    on_node_start: NodeHook | None = None,
    on_node_end: NodeHook | None = None,
) -> AgentSpec:
    """WorkflowModel を据えた AgentSpec を返す（経路C・主軸・FR-8）。

    `WorkflowGraph.as_agent_spec` の実体。引数・戻り値・挙動は同メソッドと同一。

    Args:
        graph: Agent 化する WorkflowGraph。
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
    from .. import _adapters

    async def interpret_input(input: Any, *, context: Any = None) -> WorkflowResult:
        # tracer は run ごとに新規生成するステートレス factory。SDK tracing 無効時 / 親 trace
        # 不在時は no-op tracer が返り span は発行されない（オーバーヘッドなし）。
        tracer = _adapters.make_workflow_tracer(graph.name)
        return await interpret(
            graph,
            _adapters.DefaultRunnerAdapter(registry),
            input,
            context=context,
            on_node_start=on_node_start,
            on_node_end=on_node_end,
            tracer=tracer,
        )

    model = _adapters.WorkflowModel(interpret_input, output_extractor=output_extractor)
    return AgentSpec(name=name, model=model)


def build_facade_spec(
    graph: WorkflowGraph,
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

    `WorkflowGraph.as_facade_spec` の実体。引数・戻り値・挙動は同メソッドと同一。

    Args:
        graph: ファサード化する WorkflowGraph。
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
    from .. import _adapters

    # 生文字列（"deterministic" 等）でも受けられるよう enum へ正規化する。FacadeMode は
    # str, Enum のため == は文字列とも等価だが is 比較は不一致になる。以降の分岐を is で
    # 統一するため冒頭で coerce する（未知値は ValueError で fail-fast）。
    mode = FacadeMode(mode)

    if mode is FacadeMode.DETERMINISTIC and model is not None:
        raise ValueError(
            f"as_facade_spec {name!r}: mode=DETERMINISTIC では model を指定できません"
            "（決定論モデルを内部注入します。LLM 系 mode で model を使ってください）"
        )

    async def interpret_input(input: Any, *, context: Any = None) -> WorkflowResult:
        # tracer は run ごとに新規生成するステートレス factory。SDK tracing 無効時 / 親 trace
        # 不在時は no-op tracer が返り span は発行されない（オーバーヘッドなし）。
        tracer = _adapters.make_workflow_tracer(graph.name)
        return await interpret(
            graph,
            _adapters.DefaultRunnerAdapter(registry),
            input,
            context=context,
            on_node_start=on_node_start,
            on_node_end=on_node_end,
            tracer=tracer,
        )

    resolved_tool_name = tool_name or f"{name}_workflow"
    workflow_tool = _adapters.workflow_as_tool(
        interpret_input,
        tool_name=resolved_tool_name,
        tool_description=tool_description,
        output_extractor=output_extractor,
    )
    facade_model = (
        _adapters.DeterministicToolCallModel(resolved_tool_name)
        if mode is FacadeMode.DETERMINISTIC
        else model
    )
    # 出口の LLM 要約を行わない mode では最初の tool 結果で停止する。
    # DETERMINISTIC では停止しないと決定論モデルが tool を吐き続け無限ループになる。
    stop_after_tool = mode in (FacadeMode.DETERMINISTIC, FacadeMode.LLM_INPUT)
    extra = _adapters.make_facade_extra() if stop_after_tool else {}
    return AgentSpec(
        name=name,
        model=facade_model,
        tools=[workflow_tool],
        model_settings=_adapters.make_required_tool_choice_settings(),
        extra=extra,
    )


def connect_facade(
    graph: WorkflowGraph,
    registry: AgentRegistry,
    handoff_graph: HandoffGraph,
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

    `WorkflowGraph.connect_as_facade` の実体。ファサード生成は `graph.as_facade_spec(...)`
    メソッド経由で呼ぶ（メソッド層の委譲連鎖を保つ）。引数・戻り値・挙動は同メソッドと同一。

    Args:
        graph: ファサード化する WorkflowGraph。
        registry: ファサードを登録する AgentRegistry。
        handoff_graph: handoff エッジを張る HandoffGraph。
        name: ファサード AgentSpec 名（handoff 流入先）。
        src: 流入元エージェント名（`src -> name` のエッジを張る）。
        input_filter: handoff エッジの input_filter。未指定で既定（直近 1 件）、明示
            None で全履歴流入。
        description: handoff tool の説明（任意）。
        mode / model / tool_name / tool_description / output_extractor / on_node_start /
            on_node_end: `as_facade_spec` へ素通し。

    Returns:
        登録したファサード AgentSpec。
    """
    spec = graph.as_facade_spec(
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
    registry.register(spec)
    resolved = default_input_filter() if input_filter is _UNSET else input_filter
    handoff_graph.edge(src, name, description=description, input_filter=resolved)
    return spec


def default_input_filter(limit: int = WORKFLOW_DEFAULT_INPUT_HISTORY_LIMIT) -> Callable[[Any], Any]:
    """流入履歴を直近 N 件に有界化する既定 input_filter を返す（C-10）。

    SDK の input_filter は `HandoffInputData -> HandoffInputData`。`input_history` が
    タプル/リストのときのみ末尾 N 件へ切り詰める（文字列等はそのまま透過）。

    Args:
        limit: 残す直近件数（既定は 1）。

    Returns:
        HandoffInputData を受け取り直近 N 件へ切り詰める callable。
    """

    def _filter(data: Any) -> Any:
        history = getattr(data, "input_history", None)
        if isinstance(history, (list, tuple)) and len(history) > limit:
            trimmed = tuple(history[-limit:])
            if hasattr(data, "clone"):
                return data.clone(input_history=trimmed)
        return data

    return _filter
