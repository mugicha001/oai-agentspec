"""LLMOps 評価の公開窓口（`oai-agentspec[llmops]` extra・agents 非依存・公開 API）。

`evaluate`（評価エントリ）・観点オブジェクト（`Criterion` + 組込みファクトリ）・結果型 / 状態
Enum・設定型・入力型・Langfuse dataset 連携ヘルパ（`register_dataset` / `load_dataset`）を再
エクスポートする。重い依存（`deepeval` / `langfuse`）はトップ import せず `evaluator` /
`dataset` / `_adapters/{judge,langfuse}` の関数内遅延 import に閉じる。よって
`from oai_agentspec.runtime.llmops import evaluate` は extra 未導入でも壊れず、実際の採点 / 送信
時に初めて必要 extra を案内する（FR-7 / NFR-2）。

観点は `Criterion` オブジェクト（組込みファクトリ `Relevance` / `Safety` / `Conciseness` /
`Faithfulness` / `GEval` / `ToolUse` / `HandoffRoute` / `ApprovalGate`）で宣言する。抽象
メトリクス識別子
（`MetricId`）・観点文字列定数は内部実装（公開しない）。コア `__init__` の `__all__` には載せない
（公開 API は本窓口に集約・FR-7）。
"""

from __future__ import annotations

from .config import EvaluationConfig, JudgeConfig, LangfuseConfig
from .criteria import (
    ApprovalGate,
    Conciseness,
    Criterion,
    Faithfulness,
    GEval,
    HandoffRoute,
    Relevance,
    Safety,
    ToolUse,
)
from .dataset import EvalCase, load_dataset, register_dataset
from .evaluator import evaluate
from .types import (
    CaseResult,
    CriterionResult,
    CriterionStatus,
    EvaluationResult,
    ObservedApproval,
    ObservedRoute,
    ObservedRun,
    ObservedToolCall,
    RouteStep,
    Verdict,
)

__all__ = [
    # エントリ
    "evaluate",
    # 観点オブジェクト + 組込みファクトリ
    "Criterion",
    "Relevance",
    "Safety",
    "Conciseness",
    "Faithfulness",
    "GEval",
    "ToolUse",
    "HandoffRoute",
    "ApprovalGate",
    # 結果型 / 状態 Enum
    "CaseResult",
    "CriterionResult",
    "CriterionStatus",
    "EvaluationResult",
    "Verdict",
    # 実行トレース plain 型
    "ObservedApproval",
    "ObservedRoute",
    "ObservedRun",
    "ObservedToolCall",
    "RouteStep",
    # 設定型
    "EvaluationConfig",
    "JudgeConfig",
    "LangfuseConfig",
    # 入力型
    "EvalCase",
    # Langfuse dataset 連携（register → fetch → use）
    "register_dataset",
    "load_dataset",
]
