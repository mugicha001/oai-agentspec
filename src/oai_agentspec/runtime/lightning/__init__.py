"""Agent Lightning 最適化の公開窓口（`oai-agentspec[lightning]` extra・agents 非依存・公開 API）。

`optimize`（最適化エントリ・APO）・結果型 / スロット型（`OptimizeResult` / `Slot`）・設定型
（`OptimizeConfig`）・reward ファクトリ（`contains` / `exact` / `tool_match` / `judge`）・
`prompt_slot` / `prompt_slots`・`train_val_split` を再エクスポートする。重い依存
（`agentlightning`）はトップ import せず `optimizer` / `rewards` / `_adapters/lightning` の関数内
遅延 import に閉じる。よって `from oai_agentspec.runtime.lightning import optimize` は extra
未導入でも壊れず、実際の最適化実行 / judge 採点時に初めて必要 extra を案内する（FR-7 / NFR-2）。

APO（プロンプト最適化）のみを提供する。RL（モデル更新）は別 extra `oai-agentspec[lightning-rl]` で
提供される（`optimize(algorithm="rl", ...)` は明確なエラーで案内する）。公開 API は本窓口に集約し、
コア `__init__` の `__all__` には載せない（FR-7）。
"""

from __future__ import annotations

from .config import OptimizeConfig
from .dataset import OptimizeCase, train_val_split
from .optimizer import optimize
from .rewards import (
    approval_match,
    contains,
    exact,
    judge,
    last_agent_match,
    route_match,
    tool_match,
)
from .slots import prompt_slot, prompt_slots
from .types import (
    FailureKind,
    OptimizeError,
    OptimizeResult,
    RolloutResult,
    Slot,
)

__all__ = [
    # エントリ
    "optimize",
    # 結果型 / スロット型 / ケース型
    "OptimizeResult",
    "OptimizeCase",
    "Slot",
    "RolloutResult",
    # 失敗種別 / 構造化エラー（FR-8）
    "FailureKind",
    "OptimizeError",
    # 設定型
    "OptimizeConfig",
    # reward ファクトリ
    "contains",
    "exact",
    "tool_match",
    "approval_match",
    "route_match",
    "last_agent_match",
    "judge",
    # スロットヘルパ
    "prompt_slot",
    "prompt_slots",
    # データ分割
    "train_val_split",
]
