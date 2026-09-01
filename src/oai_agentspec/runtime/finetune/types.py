"""Fine-Tuning データ層の plain 型（外部 SDK 非依存）。

本モジュールは `agents` / `openai` を一切 import しない純データ層で、変換・検証ヘルパ
（`dataset`）・学習ジョブ管理（`jobs`）と公開窓口が扱う型のみを定義する（NFR-1）。
すべて `@dataclass(frozen=True)`（lightning / llmops の結果型と一致・Pydantic 非導入）。

`DatasetBuildResult.save` は利用者指定パスへの opt-in 書込のみで、明示呼び出しが唯一の
書込経路である（`OptimizeResult.save` と同一契約）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

from ..._validation import validate_bool


class FineTuneFailureKind(StrEnum):
    """Fine-Tuning 支援の失敗種別（構造化エラーで判別可能にする）。

    Attributes:
        VALIDATION_FAILED: データ不備（欠落 / 不正な形式 / system 競合 / `tools=` の不正要素）
            またはデータセット検証の不合格。
        EXTRA_MISSING: `finetune` extra が未導入で必要な依存を import できない。
        CONFIG_MISSING: 必須設定の不在および設定の不整合（重複指定による衝突を含む）。
        API_ERROR: プラットフォーム API がエラーを返した。
        TIMEOUT: 待機がタイムアウトした。
    """

    VALIDATION_FAILED = "validation_failed"
    EXTRA_MISSING = "extra_missing"
    CONFIG_MISSING = "config_missing"
    API_ERROR = "api_error"
    TIMEOUT = "timeout"


@dataclass(frozen=True)
class DatasetViolation:
    """データセット検証で検出した違反 1 件。

    Attributes:
        line: 違反箇所の位置（1 始まり）。source がファイルパスの場合は行番号、dict 列の
            場合は要素位置を表す。
        reason: 人間可読の違反理由。
    """

    line: int
    reason: str


@dataclass(frozen=True)
class DatasetValidationReport:
    """データセット検証の結果レポート（fail-closed）。

    Attributes:
        ok: 違反ゼロのときのみ True。
        checked: 検証したレコード件数。
        violations: 検出した違反（検出順）。
    """

    ok: bool
    checked: int
    violations: tuple[DatasetViolation, ...]

    def __post_init__(self) -> None:
        """`ok` が bool であることを構築時に検証する（ADR 0021）。

        Raises:
            ValueError: `ok` が bool でない場合。
        """
        validate_bool(self.ok, "ok")


class FineTuneError(Exception):
    """Fine-Tuning 支援が送出する構造化エラー（`OptimizeError` の型枠を踏襲）。

    Attributes:
        kind: 失敗種別（`FineTuneFailureKind`）。
        message: 人間可読のエラーメッセージ。
        report: 検証レポート（`validate_dataset(raise_on_invalid=True)` 経路でのみ非 None）。
    """

    def __init__(
        self,
        kind: FineTuneFailureKind,
        message: str,
        *,
        report: DatasetValidationReport | None = None,
    ) -> None:
        """Fine-Tuning エラーを生成する。

        Args:
            kind: 失敗種別。
            message: 人間可読メッセージ。
            report: 検証レポート（keyword-only・検証経路のみ）。

        Note:
            `report` は keyword-only。位置引数で渡すと `TypeError`。
        """
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.report = report


@dataclass(frozen=True)
class DpoCase:
    """DPO（preference）学習用のケース。

    Attributes:
        input: 入力。文字列（単一ターン）または messages 形式のリスト（複数ターン）。
        preferred_output: 望ましい出力。文字列または assistant メッセージ配列。
        non_preferred_output: 望ましくない出力。文字列または assistant メッセージ配列。
        id: 利用者側の識別子（任意）。変換結果のレコードには載せない。
        metadata: 利用者側の任意メタデータ。変換結果のレコードには載せない。
    """

    input: Any
    preferred_output: Any
    non_preferred_output: Any
    id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetBuildResult:
    """データセット変換の結果（既定は返却のみ・書込は `save` の明示呼び出しのみ）。

    Attributes:
        records: 変換済みレコード（1 要素 = JSONL 1 行）。
        skipped: 除外したケース件数。`to_sft_dataset` / `to_dpo_dataset` では
            `skip_missing=True` による除外、`dataset_from_session` では空 input ケース・
            空応答ケース（吸収後の content が空になる assistant ターン）・`case_filter` に
            よる除外がここへ計上される。
    """

    records: tuple[dict[str, Any], ...]
    skipped: int

    def save(self, path: str | Path) -> None:
        """レコードを JSONL（1 行 1 JSON）として利用者指定パスへ書き出す（opt-in）。

        非 ASCII はエスケープせず utf-8 のまま書く（`ensure_ascii=False`）。

        Args:
            path: 書き出し先パス（str / Path のいずれも可）。

        Raises:
            OSError: 書込先が書込不能 / 不正な場合（fail-closed・呼び出し側へ伝播）。
            TypeError: レコードに JSON へ変換できない値が含まれる場合（`json.dumps` から伝播）。

        Note:
            既存ファイルは上書きする（追記しない）。シンボリックリンクはリンク先へ追従し、
            作成されるファイルのパーミッションはプロセスの umask に依存する。
        """
        body = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in self.records)
        Path(path).write_text(body, encoding="utf-8")


class JobStatus(StrEnum):
    """学習ジョブの状態（プラットフォームの生の状態文字列を写像した plain 型）。

    終端は SUCCEEDED / FAILED / CANCELLED の 3 種で、QUEUED / RUNNING は非終端である。
    未知の状態値（プラットフォーム固有の中間状態を含む）は例外にせず RUNNING（非終端）へ
    倒すため、状態一覧の追加により待機が失敗することはない（FR-6）。

    Attributes:
        QUEUED: 実行待ち（非終端）。
        RUNNING: 実行中。未知の状態値のフォールバック先でもある（非終端）。
        SUCCEEDED: 正常終了（終端）。
        FAILED: 失敗して終了（終端）。
        CANCELLED: 取り消されて終了（終端）。
    """

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


_TERMINAL_STATUSES: Final[frozenset[JobStatus]] = frozenset(
    {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED}
)


def _map_status(raw_status: str) -> JobStatus:
    """プラットフォームの生の状態文字列を `JobStatus` へ写像する（FR-6）。

    終端 3 種と `queued` のみを判定し、それ以外（`running` / 未知の中間状態 / 空文字を
    含む）はすべて RUNNING へ倒す。例外は送出しない。

    Args:
        raw_status: プラットフォームが返す生の状態文字列。

    Returns:
        写像後の `JobStatus`。判定できない値は `JobStatus.RUNNING`。
    """
    for terminal in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED):
        if raw_status == terminal.value:
            return terminal
    if raw_status == JobStatus.QUEUED.value:
        return JobStatus.QUEUED
    return JobStatus.RUNNING


@dataclass(frozen=True)
class JobRef:
    """投入した学習ジョブの参照（投入時に確定する識別子のみを持つ）。

    Attributes:
        job_id: プラットフォームが払い出したジョブ ID。
        training_file_id: 学習データのファイル ID（アップロード済み / 利用者指定）。
        validation_file_id: 検証データのファイル ID。指定していない場合は None。
    """

    job_id: str
    training_file_id: str
    validation_file_id: str | None


@dataclass(frozen=True)
class JobResult:
    """学習ジョブの照会結果（生の状態文字列を保全する）。

    Attributes:
        job_id: 照会したジョブ ID。
        status: 写像後の状態。
        raw_status: プラットフォームが返した生の状態文字列（写像で失わない）。
        model_ref: 学習済みモデルの参照。未確定の場合は None。
        error_message: 失敗理由の文言。無い場合は None。
    """

    job_id: str
    status: JobStatus
    raw_status: str
    model_ref: str | None
    error_message: str | None

    @property
    def is_terminal(self) -> bool:
        """状態が終端 3 種（succeeded / failed / cancelled）のいずれかかを返す。

        Returns:
            終端なら True、非終端（queued / running）なら False。
        """
        return self.status in _TERMINAL_STATUSES
