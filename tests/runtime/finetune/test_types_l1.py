"""L1: Fine-Tuning データ層の plain 型（外部 SDK 非依存）を検証する。

`FineTuneFailureKind`（5 種の失敗種別・StrEnum 値）・`JobStatus` / `JobRef` / `JobResult`
（ジョブ管理の plain 型・状態写像・終端判定）・`FineTuneError`
（kind / message / keyword-only `report`）・`DpoCase`（frozen・既定値の独立性）・
`DatasetBuildResult`（frozen・`save(path)` の str / Path 両対応・JSONL 書式・非 ASCII 非
エスケープ・既定では書き出さない opt-in 契約）・`DatasetViolation` / `DatasetValidationReport`
（frozen・fail-closed な `ok`）を網羅する。すべて純データ操作で外部依存なし
（`@pytest.mark.unit`）。
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from oai_agentspec.runtime.finetune import (
    DatasetBuildResult,
    DatasetValidationReport,
    DatasetViolation,
    DpoCase,
    FineTuneError,
    FineTuneFailureKind,
)
from oai_agentspec.runtime.finetune.types import (
    JobRef,
    JobResult,
    JobStatus,
    _map_status,
)

pytestmark = pytest.mark.unit


# ----------------------------------------------------------------------
# FineTuneFailureKind（StrEnum）
# ----------------------------------------------------------------------


def test_failure_kind_validation_failed_string_value() -> None:
    """VALIDATION_FAILED は "validation_failed" の文字列値を持ち str として扱える。"""
    assert FineTuneFailureKind.VALIDATION_FAILED == "validation_failed"
    assert isinstance(FineTuneFailureKind.VALIDATION_FAILED, str)


def test_failure_kind_member_set_is_pinned() -> None:
    """失敗種別のメンバ集合は 5 種で固定する（未使用メンバを持ち込まない）。

    `==` でメンバ名集合を照合するため、余分メンバの混入（過大側）とメンバ欠落
    （過小側）の双方を検知できる。
    """
    assert {member.name for member in FineTuneFailureKind} == {
        "VALIDATION_FAILED",
        "EXTRA_MISSING",
        "CONFIG_MISSING",
        "API_ERROR",
        "TIMEOUT",
    }


def test_failure_kind_values_are_pinned_snake_case() -> None:
    """各メンバの文字列値は snake_case で固定する（既存 validation_failed の値は不変）。

    値は構造化エラーの判別キーとして外部へ露出するため、名前と値の対応を `==` で固定する。
    """
    assert {member.name: member.value for member in FineTuneFailureKind} == {
        "VALIDATION_FAILED": "validation_failed",
        "EXTRA_MISSING": "extra_missing",
        "CONFIG_MISSING": "config_missing",
        "API_ERROR": "api_error",
        "TIMEOUT": "timeout",
    }


@pytest.mark.parametrize(
    "kind",
    [
        FineTuneFailureKind.EXTRA_MISSING,
        FineTuneFailureKind.CONFIG_MISSING,
        FineTuneFailureKind.API_ERROR,
        FineTuneFailureKind.TIMEOUT,
    ],
)
def test_finetune_error_carries_each_new_kind(kind: FineTuneFailureKind) -> None:
    """新しい失敗種別を FineTuneError へ渡すと kind 属性で判別できる。"""
    with pytest.raises(FineTuneError) as exc_info:
        raise FineTuneError(kind, "boom")
    assert exc_info.value.kind == kind
    assert exc_info.value.report is None


# ----------------------------------------------------------------------
# FineTuneError
# ----------------------------------------------------------------------


def test_finetune_error_carries_kind_and_message() -> None:
    """FineTuneError は kind / message を保持し Exception メッセージにも反映する。"""
    err = FineTuneError(FineTuneFailureKind.VALIDATION_FAILED, "検証に失敗しました")
    assert err.kind == FineTuneFailureKind.VALIDATION_FAILED
    assert err.message == "検証に失敗しました"
    assert str(err) == "検証に失敗しました"


def test_finetune_error_is_exception() -> None:
    """FineTuneError は Exception サブクラスで raise / except できる。"""
    with pytest.raises(FineTuneError) as exc_info:
        raise FineTuneError(FineTuneFailureKind.VALIDATION_FAILED, "boom")
    assert exc_info.value.kind == FineTuneFailureKind.VALIDATION_FAILED


def test_finetune_error_report_defaults_none() -> None:
    """`report` は既定 None（データ不備エラーは検証レポートを伴わない経路がある）。"""
    err = FineTuneError(FineTuneFailureKind.VALIDATION_FAILED, "msg")
    assert err.report is None


def test_finetune_error_stores_report() -> None:
    """`report` に DatasetValidationReport を保持できる（raise_on_invalid 経路の情報保全）。"""
    report = DatasetValidationReport(ok=False, checked=1, violations=())
    err = FineTuneError(FineTuneFailureKind.VALIDATION_FAILED, "msg", report=report)
    assert err.report is report


def test_finetune_error_report_is_keyword_only() -> None:
    """`report` は keyword-only（位置引数として渡すと TypeError）。"""
    report = DatasetValidationReport(ok=False, checked=1, violations=())
    with pytest.raises(TypeError):
        FineTuneError(FineTuneFailureKind.VALIDATION_FAILED, "msg", report)  # type: ignore[misc]


# ----------------------------------------------------------------------
# DpoCase
# ----------------------------------------------------------------------


def test_dpo_case_holds_required_fields() -> None:
    """DpoCase は input / preferred_output / non_preferred_output を保持する。"""
    case = DpoCase(
        input="配送状況を確認したい",
        preferred_output="注文番号をお知らせください。",
        non_preferred_output="自分で調べてください。",
    )
    assert case.input == "配送状況を確認したい"
    assert case.preferred_output == "注文番号をお知らせください。"
    assert case.non_preferred_output == "自分で調べてください。"


def test_dpo_case_optional_defaults() -> None:
    """id 既定は None・metadata 既定は空 dict。"""
    case = DpoCase(input="x", preferred_output="a", non_preferred_output="b")
    assert case.id is None
    assert case.metadata == {}


def test_dpo_case_is_frozen() -> None:
    """DpoCase は frozen dataclass で属性再代入できない。"""
    case = DpoCase(input="x", preferred_output="a", non_preferred_output="b")
    with pytest.raises((AttributeError, TypeError)):
        case.input = "y"  # type: ignore[misc]


def test_dpo_case_independent_default_metadata() -> None:
    """既定 metadata は default_factory でインスタンスごとに独立（同一参照を共有しない）。"""
    a = DpoCase(input="a", preferred_output="p", non_preferred_output="n")
    b = DpoCase(input="b", preferred_output="p", non_preferred_output="n")
    assert a.metadata is not b.metadata


def test_dpo_case_accepts_messages_list_input_and_array_outputs() -> None:
    """input は messages リスト・出力側は assistant 配列も保持できる（複数ターン受理）。"""
    messages = [{"role": "user", "content": "解約したい"}]
    preferred = [{"role": "assistant", "content": "承知しました。"}]
    non_preferred = [{"role": "assistant", "content": "できません。"}]
    case = DpoCase(input=messages, preferred_output=preferred, non_preferred_output=non_preferred)
    assert case.input == messages
    assert case.preferred_output == preferred
    assert case.non_preferred_output == non_preferred


# ----------------------------------------------------------------------
# DatasetBuildResult
# ----------------------------------------------------------------------


def test_dataset_build_result_holds_records_and_skipped() -> None:
    """DatasetBuildResult は records と skipped 件数を保持する。"""
    result = DatasetBuildResult(records=({"messages": []},), skipped=2)
    assert list(result.records) == [{"messages": []}]
    assert result.skipped == 2


def test_dataset_build_result_is_frozen() -> None:
    """DatasetBuildResult は frozen dataclass で属性再代入できない。"""
    result = DatasetBuildResult(records=(), skipped=0)
    with pytest.raises((AttributeError, TypeError)):
        result.skipped = 1  # type: ignore[misc]


def test_save_writes_jsonl_one_record_per_line(tmp_path: Path) -> None:
    """save は 1 行 1 JSON の JSONL として書き出す（レコード順を保つ）。"""
    target = tmp_path / "train.jsonl"
    records = (
        {"messages": [{"role": "user", "content": "a"}]},
        {"messages": [{"role": "user", "content": "b"}]},
    )
    DatasetBuildResult(records=records, skipped=0).save(target)

    lines = target.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert [json.loads(line) for line in lines] == list(records)


def test_save_keeps_non_ascii_unescaped(tmp_path: Path) -> None:
    """save は非 ASCII をエスケープせず utf-8 のまま書き出す（ensure_ascii=False）。"""
    target = tmp_path / "train.jsonl"
    DatasetBuildResult(
        records=({"messages": [{"role": "user", "content": "返品ポリシー"}]},), skipped=0
    ).save(target)

    text = target.read_text(encoding="utf-8")
    assert "返品ポリシー" in text
    assert "\\u" not in text


def test_save_accepts_str_path(tmp_path: Path) -> None:
    """save は str パスも受け取れる（Path へ正規化される）。"""
    target = tmp_path / "train.jsonl"
    DatasetBuildResult(records=({"messages": []},), skipped=0).save(str(target))
    assert json.loads(target.read_text(encoding="utf-8").strip()) == {"messages": []}


def test_build_result_does_not_write_without_save(tmp_path: Path) -> None:
    """既定は返却のみで書き込まない（save の明示呼び出しが唯一の書込経路 = opt-in 契約）。"""
    target = tmp_path / "train.jsonl"
    DatasetBuildResult(records=({"messages": []},), skipped=0)
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_save_overwrites_existing_file_instead_of_appending(tmp_path: Path) -> None:
    """save は既存ファイルを上書きする（追記しない・2 回目の records だけが残る）。"""
    target = tmp_path / "train.jsonl"
    first = (
        {"messages": [{"role": "user", "content": "1 回目 a"}]},
        {"messages": [{"role": "user", "content": "1 回目 b"}]},
    )
    second = ({"messages": [{"role": "user", "content": "2 回目"}]},)
    DatasetBuildResult(records=first, skipped=0).save(target)
    DatasetBuildResult(records=second, skipped=0).save(target)

    lines = target.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert [json.loads(line) for line in lines] == list(second)
    assert "1 回目" not in target.read_text(encoding="utf-8")


# ----------------------------------------------------------------------
# DatasetViolation / DatasetValidationReport
# ----------------------------------------------------------------------


def test_dataset_violation_holds_line_and_reason() -> None:
    """DatasetViolation は line（1 始まり）と reason を保持し frozen である。"""
    violation = DatasetViolation(line=7, reason="JSON として解析できない")
    assert violation.line == 7
    assert violation.reason == "JSON として解析できない"
    with pytest.raises((AttributeError, TypeError)):
        violation.line = 8  # type: ignore[misc]


def test_dataset_validation_report_holds_fields_and_is_frozen() -> None:
    """DatasetValidationReport は ok / checked / violations を保持し frozen である。"""
    violation = DatasetViolation(line=1, reason="必須キー 'messages' が存在しない")
    report = DatasetValidationReport(ok=False, checked=3, violations=(violation,))
    assert report.ok is False
    assert report.checked == 3
    assert report.violations == (violation,)
    with pytest.raises((AttributeError, TypeError)):
        report.ok = True  # type: ignore[misc]


# ----------------------------------------------------------------------
# JobStatus（StrEnum）
# ----------------------------------------------------------------------


def test_job_status_member_set_is_pinned() -> None:
    """JobStatus のメンバ集合は 5 種で固定する（過大側・過小側の双方を検知する）。"""
    assert {member.name for member in JobStatus} == {
        "QUEUED",
        "RUNNING",
        "SUCCEEDED",
        "FAILED",
        "CANCELLED",
    }


def test_job_status_values_are_pinned_and_str_like() -> None:
    """各メンバの文字列値を固定し、StrEnum として str と比較できる。"""
    assert {member.name: member.value for member in JobStatus} == {
        "QUEUED": "queued",
        "RUNNING": "running",
        "SUCCEEDED": "succeeded",
        "FAILED": "failed",
        "CANCELLED": "cancelled",
    }
    assert isinstance(JobStatus.RUNNING, str)
    assert JobStatus.SUCCEEDED == "succeeded"


# ----------------------------------------------------------------------
# 状態写像（_map_status）: FR-6 の未知状態フォールバック規則
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("succeeded", JobStatus.SUCCEEDED),
        ("failed", JobStatus.FAILED),
        ("cancelled", JobStatus.CANCELLED),
        ("queued", JobStatus.QUEUED),
        ("running", JobStatus.RUNNING),
    ],
)
def test_map_status_maps_known_states(raw: str, expected: JobStatus) -> None:
    """既知の状態文字列は対応する JobStatus へ写像される（終端 3 種 + queued + running）。"""
    assert _map_status(raw) == expected


@pytest.mark.parametrize(
    "raw",
    ["validating_files", "pausing", "paused", "", "SUCCEEDED_TYPO", "unknown-state"],
)
def test_map_status_falls_back_to_running_for_unknown_states(raw: str) -> None:
    """本要件の列挙に無い状態値は例外にせず RUNNING（非終端）へ倒す（FR-6）。

    状態一覧をハードコードせず終端のみを判定する設計のため、未知値は待機継続側へ倒す。
    """
    assert _map_status(raw) == JobStatus.RUNNING


def test_map_status_unknown_state_is_not_terminal() -> None:
    """未知状態から作った JobResult は非終端（wait_job が待機を継続できる）。"""
    result = JobResult(
        job_id="ftjob-1",
        status=_map_status("validating_files"),
        raw_status="validating_files",
        model_ref=None,
        error_message=None,
    )
    assert result.is_terminal is False


# ----------------------------------------------------------------------
# JobRef
# ----------------------------------------------------------------------


def test_job_ref_holds_fields() -> None:
    """JobRef は job_id / training_file_id / validation_file_id を保持する。"""
    ref = JobRef(job_id="ftjob-1", training_file_id="file-tr", validation_file_id="file-val")
    assert ref.job_id == "ftjob-1"
    assert ref.training_file_id == "file-tr"
    assert ref.validation_file_id == "file-val"


def test_job_ref_allows_none_validation_file_id() -> None:
    """validation_file_id は None を取りうる（val 省略時）。"""
    ref = JobRef(job_id="ftjob-1", training_file_id="file-tr", validation_file_id=None)
    assert ref.validation_file_id is None


def test_job_ref_is_frozen() -> None:
    """JobRef は frozen dataclass で属性再代入できない。"""
    ref = JobRef(job_id="ftjob-1", training_file_id="file-tr", validation_file_id=None)
    with pytest.raises((AttributeError, TypeError)):
        ref.job_id = "ftjob-2"  # type: ignore[misc]


# ----------------------------------------------------------------------
# JobResult
# ----------------------------------------------------------------------


def _result(status: JobStatus, raw_status: str) -> JobResult:
    """テスト用の JobResult を組み立てる（model_ref / error_message は未設定）。"""
    return JobResult(
        job_id="ftjob-1",
        status=status,
        raw_status=raw_status,
        model_ref=None,
        error_message=None,
    )


def test_job_result_holds_fields() -> None:
    """JobResult は job_id / status / raw_status / model_ref / error_message を保持する。"""
    result = JobResult(
        job_id="ftjob-1",
        status=JobStatus.SUCCEEDED,
        raw_status="succeeded",
        model_ref="ft:gpt-4o-mini:acme::abc123",
        error_message=None,
    )
    assert result.job_id == "ftjob-1"
    assert result.status == JobStatus.SUCCEEDED
    assert result.raw_status == "succeeded"
    assert result.model_ref == "ft:gpt-4o-mini:acme::abc123"
    assert result.error_message is None


def test_job_result_keeps_raw_status_for_unknown_state() -> None:
    """未知状態でも raw_status にプラットフォームの生文字列を保全する（写像で失わない）。"""
    result = _result(_map_status("validating_files"), "validating_files")
    assert result.status == JobStatus.RUNNING
    assert result.raw_status == "validating_files"


def test_job_result_keeps_error_message_on_failure() -> None:
    """失敗時は error_message に理由文言を保全する。"""
    result = JobResult(
        job_id="ftjob-1",
        status=JobStatus.FAILED,
        raw_status="failed",
        model_ref=None,
        error_message="training file is invalid",
    )
    assert result.error_message == "training file is invalid"


def test_job_result_is_frozen() -> None:
    """JobResult は frozen dataclass で属性再代入できない。"""
    result = _result(JobStatus.RUNNING, "running")
    with pytest.raises((AttributeError, TypeError)):
        result.status = JobStatus.SUCCEEDED  # type: ignore[misc]


@pytest.mark.parametrize(
    "status",
    [JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED],
)
def test_job_result_is_terminal_true_for_terminal_states(status: JobStatus) -> None:
    """終端 3 種（succeeded / failed / cancelled）では is_terminal が True。"""
    assert _result(status, status.value).is_terminal is True


@pytest.mark.parametrize("status", [JobStatus.QUEUED, JobStatus.RUNNING])
def test_job_result_is_terminal_false_for_non_terminal_states(status: JobStatus) -> None:
    """非終端（queued / running）では is_terminal が False。"""
    assert _result(status, status.value).is_terminal is False


def test_job_result_is_terminal_is_a_property_not_a_field() -> None:
    """is_terminal は property であり dataclass フィールドではない。

    bool の dataclass フィールドにすると ADR-0021 の網羅性メタテスト
    （`tests/test_bool_fields_l1.py`）の走査対象となり構築時検証を要求されるため、
    導出値であることをクラス属性の型で固定する。
    """
    assert isinstance(vars(JobResult).get("is_terminal"), property)
    assert "is_terminal" not in {f.name for f in dataclasses.fields(JobResult)}
