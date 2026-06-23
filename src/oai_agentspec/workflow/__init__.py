"""宣言的ワークフロー（`WorkflowGraph`）と非公開の内部インタプリタ（node/edge 方式）。

LangGraph（`StateGraph`）/ Microsoft Agent Framework（`WorkflowBuilder`）に倣い、
ノード（AGENT/FUNCTION）とエッジ（通常 / 条件 / fan-in）+ `START`/`END` 番兵を
`add_*` で明示宣言する。チェーン糖衣（sequence/parallel/branch/loop）や位置依存の
暗黙ルールは採らない。データはノード出力がエッジに沿って下流へ流れるメッセージ受け渡しで
運び、独立 state(reducer) 機構は新設しない（fan-in か共有 context で明示・C-1/C-4）。

実行口は SDK の `Runner.run` 一本に寄せ、本モジュールは公開の実行 API を持たない
（build-don't-run の純化、C-3）。ワークフローは Agent（経路C: `as_agent_spec`）または
Tool ファサード（経路A: `as_facade_spec`）として消費する。

SDK 隔離（NFR-1）: 本パッケージは `agents` をランタイム import しない。SDK 型は
`TYPE_CHECKING` / `Protocol` のみで参照し、SDK 実体（Runner / Model / ModelResponse /
ModelSettings / FunctionTool / ToolContext）への結合は `_adapters` に閉じる。依存方向は
`workflow -> _adapters -> agents` の一方向（循環回避）。

実装は責務別サブモジュール（`_types` / `_declarations` / `graph` / `_interpreter` /
`_facade`）に分割し、本 `__init__` は公開シンボルを再エクスポートする薄い集約に徹する
（import 互換の単一窓口を維持）。
"""

from __future__ import annotations

from ._declarations import (
    ConditionalEdge,
    FanInEdge,
    NodeResults,
    WorkflowNode,
    WorkflowResult,
)
from ._facade import default_input_filter
from ._types import (
    END,
    START,
    FacadeMode,
    NodeFn,
    NodeHook,
    NodeKind,
    Router,
)
from .graph import WorkflowFrozenError, WorkflowGraph

__all__ = [
    "END",
    "START",
    "ConditionalEdge",
    "FacadeMode",
    "FanInEdge",
    "NodeFn",
    "NodeHook",
    "NodeKind",
    "NodeResults",
    "Router",
    "WorkflowFrozenError",
    "WorkflowGraph",
    "WorkflowNode",
    "WorkflowResult",
    "default_input_filter",
]
