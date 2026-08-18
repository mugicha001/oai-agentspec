"""L1: Fine-Tuning データ層の plain 型（外部 SDK 非依存）を検証する。

`FineTuneFailureKind`（段階 1 は VALIDATION_FAILED のみ・StrEnum 値）・`FineTuneError`
（kind / message / keyword-only `report`）・`DpoCase`（frozen・既定値の独立性）・
`DatasetBuildResult`（frozen・`save(path)` の str / Path 両対応・JSONL 書式・非 ASCII 非
エスケープ・既定では書き出さない opt-in 契約）・`DatasetViolation` / `DatasetValidationReport`
（frozen・fail-closed な `ok`）を網羅する。すべて純データ操作で外部依存なし
（`@pytest.mark.unit`）。
"""

from __future__ import annotations

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

pytestmark = pytest.mark.unit


# ----------------------------------------------------------------------
# FineTuneFailureKind（StrEnum）
# ----------------------------------------------------------------------


def test_failure_kind_validation_failed_string_value() -> None:
    """VALIDATION_FAILED は "validation_failed" の文字列値を持ち str として扱える。"""
    assert FineTuneFailureKind.VALIDATION_FAILED == "validation_failed"
    assert isinstance(FineTuneFailureKind.VALIDATION_FAILED, str)


def test_failure_kind_member_set_is_pinned_to_validation_failed_only() -> None:
    """段階 1 のメンバ集合は VALIDATION_FAILED のみ（未使用メンバを持ち込まない）。

    `==` でメンバ名集合を照合し、余分メンバの混入（過大側）も検知できる形で固定する。
    """
    assert {member.name for member in FineTuneFailureKind} == {"VALIDATION_FAILED"}


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
