"""Fine-Tuning データ層の plain 型（外部 SDK 非依存）。

本モジュールは `agents` / `openai` を一切 import しない純データ層で、変換・検証ヘルパ
（`dataset`）と公開窓口が扱う型のみを定義する（NFR-1）。すべて `@dataclass(frozen=True)`
（lightning / llmops の結果型と一致・Pydantic 非導入）。

`DatasetBuildResult.save` は利用者指定パスへの opt-in 書込のみで、明示呼び出しが唯一の
書込経路である（`OptimizeResult.save` と同一契約）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from ..._validation import validate_bool


class FineTuneFailureKind(StrEnum):
    """Fine-Tuning 支援の失敗種別（構造化エラーで判別可能にする）。

    Attributes:
        VALIDATION_FAILED: データ不備（欠落 / 不正な形式 / system 競合 / `tools=` の不正要素）
            またはデータセット検証の不合格。
    """

    VALIDATION_FAILED = "validation_failed"


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
        skipped: `skip_missing=True` により除外したケース件数。
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
