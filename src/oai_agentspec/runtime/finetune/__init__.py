"""Fine-Tuning 支援の公開窓口（`oai-agentspec[finetune]` extra・agents 非依存・公開 API）。

データセット変換（`to_sft_dataset` / `to_dpo_dataset`）・検証（`validate_dataset`）と、
ケース型 / 結果型 / 検証レポート型 / 構造化エラーを再エクスポートする。本窓口の実装は純データ
操作とローカルファイル I/O のみで、`agents` / `openai` を import せずネットワークにも触れない。
よって `from oai_agentspec.runtime.finetune import to_sft_dataset` は extra 未導入でも壊れない。

公開 API は本窓口に集約し、コア `__init__` の `__all__` には載せない。
"""

from __future__ import annotations

from .dataset import to_dpo_dataset, to_sft_dataset, validate_dataset
from .types import (
    DatasetBuildResult,
    DatasetValidationReport,
    DatasetViolation,
    DpoCase,
    FineTuneError,
    FineTuneFailureKind,
)

__all__ = [
    # 変換 / 検証エントリ
    "to_sft_dataset",
    "to_dpo_dataset",
    "validate_dataset",
    # ケース型 / 結果型 / 検証レポート型
    "DpoCase",
    "DatasetBuildResult",
    "DatasetValidationReport",
    "DatasetViolation",
    # 失敗種別 / 構造化エラー
    "FineTuneFailureKind",
    "FineTuneError",
]
