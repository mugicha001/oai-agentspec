"""Fine-Tuning 支援の公開窓口（`oai-agentspec[finetune]` extra・agents 非依存・公開 API）。

データセット変換（`to_sft_dataset` / `to_dpo_dataset` / `dataset_from_session` /
`dpo_dataset_from_session`）・検証（`validate_dataset`）・DPO 雛形の記入ワークフロー
（`save_dpo_draft` / `finalize_dpo_draft`）・学習ジョブ管理（`submit_job` / `get_job` /
`wait_job`）と、ケース型 / 結果型 / 検証レポート型 / ジョブ参照型 / 構造化エラーを
再エクスポートする。本窓口の実装は
`agents` / `openai` を import せず（SDK
接触は呼び出し時の遅延 import で `_adapters` へ閉じる）、変換・検証（`to_sft_dataset` /
`to_dpo_dataset` / `validate_dataset`）と雛形ヘルパ（`save_dpo_draft` /
`finalize_dpo_draft`）は純データ操作とローカルファイル I/O のみで完結する。
`dataset_from_session` / `dpo_dataset_from_session` は利用者供給の `Session` 越しに履歴を
読むため、Session の実装がネットワーク背当て（OpenAI Conversations 等）なら読み取りに
通信を伴う。
`from oai_agentspec.runtime.finetune import to_sft_dataset` は extra 未導入でも壊れない
（実際にジョブ管理関数を呼んだ時点で openai が必要になる）。

公開 API は本窓口に集約し、コア `__init__` の `__all__` には載せない。
"""

from __future__ import annotations

from .dataset import to_dpo_dataset, to_sft_dataset, validate_dataset
from .dpo_draft import finalize_dpo_draft, save_dpo_draft
from .jobs import get_job, submit_job, wait_job
from .session_dataset import dataset_from_session, dpo_dataset_from_session
from .types import (
    DatasetBuildResult,
    DatasetValidationReport,
    DatasetViolation,
    DpoCase,
    FineTuneError,
    FineTuneFailureKind,
    JobRef,
    JobResult,
    JobStatus,
)

__all__ = [
    # 変換 / 検証エントリ
    "to_sft_dataset",
    "to_dpo_dataset",
    "dataset_from_session",
    "dpo_dataset_from_session",
    "validate_dataset",
    # DPO 雛形の記入ワークフロー
    "save_dpo_draft",
    "finalize_dpo_draft",
    # ジョブ管理エントリ
    "submit_job",
    "get_job",
    "wait_job",
    # ケース型 / 結果型 / 検証レポート型
    "DpoCase",
    "DatasetBuildResult",
    "DatasetValidationReport",
    "DatasetViolation",
    # ジョブ参照 / 結果型
    "JobRef",
    "JobResult",
    "JobStatus",
    # 失敗種別 / 構造化エラー
    "FineTuneFailureKind",
    "FineTuneError",
]
