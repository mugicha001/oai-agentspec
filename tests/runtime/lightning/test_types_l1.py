"""L1: 最適化の plain 結果型・スロット型（外部 SDK 非依存）を検証する。

`FailureKind`（StrEnum 値）・`OptimizeError`（kind / message 保持）・`Slot`（frozen・vars 既定）・
`OptimizeResult`（`to_dict` / `save`・単一 str / mapping 分岐・`${var}` 保持・PromptStore 非書込）を
網羅する。すべて純データ操作で外部依存なし（`@pytest.mark.unit`）。
"""

from __future__ import annotations

import dataclasses
import json
import re

import pytest

from oai_agentspec.runtime.lightning import (
    FailureKind,
    OptimizeError,
    OptimizeResult,
    Slot,
)

pytestmark = pytest.mark.unit


# ----------------------------------------------------------------------
# FailureKind（StrEnum）
# ----------------------------------------------------------------------


def test_failure_kind_string_values() -> None:
    """各 FailureKind は仕様どおりの文字列値を持ち str として扱える（StrEnum）。"""
    assert FailureKind.EXTRA_MISSING == "extra_missing"
    assert FailureKind.CONFIG_MISSING == "config_missing"
    assert FailureKind.TRAINER_FAILED == "trainer_failed"
    # StrEnum なので str サブクラスとして比較できる。
    assert isinstance(FailureKind.EXTRA_MISSING, str)


def test_failure_kind_members_distinct() -> None:
    """3 種別はすべて異なる値を持つ（種別判別の前提）。"""
    values = {FailureKind.EXTRA_MISSING, FailureKind.CONFIG_MISSING, FailureKind.TRAINER_FAILED}
    assert len(values) == 3


# ----------------------------------------------------------------------
# OptimizeError
# ----------------------------------------------------------------------


def test_optimize_error_carries_kind_and_message() -> None:
    """OptimizeError は kind / message を保持し、Exception メッセージにも反映する。"""
    err = OptimizeError(FailureKind.CONFIG_MISSING, "設定が足りません")
    assert err.kind == FailureKind.CONFIG_MISSING
    assert err.message == "設定が足りません"
    assert str(err) == "設定が足りません"


def test_optimize_error_is_exception() -> None:
    """OptimizeError は Exception サブクラスで raise / except できる。"""
    with pytest.raises(OptimizeError) as exc_info:
        raise OptimizeError(FailureKind.TRAINER_FAILED, "boom")
    assert exc_info.value.kind == FailureKind.TRAINER_FAILED


# ----------------------------------------------------------------------
# Slot
# ----------------------------------------------------------------------


def test_slot_holds_name_seed_build_and_default_vars() -> None:
    """Slot は name / seed / build を保持し、vars 既定は空 dict。"""
    build = lambda c: c  # noqa: E731 - テスト用簡易 build
    slot = Slot(name="bot", seed="seed ${x}", build=build)
    assert slot.name == "bot"
    assert slot.seed == "seed ${x}"
    assert slot.build is build
    assert slot.vars == {}


def test_slot_is_frozen() -> None:
    """Slot は frozen dataclass で属性再代入できない。"""
    slot = Slot(name="bot", seed="s", build=lambda c: c)
    with pytest.raises((AttributeError, TypeError)):
        slot.name = "other"  # type: ignore[misc]


def test_slot_keeps_explicit_vars() -> None:
    """vars を明示すると保持される（最適化対象外の置換値）。"""
    slot = Slot(name="bot", seed="s", build=lambda c: c, vars={"x": "1"})
    assert slot.vars == {"x": "1"}


# ----------------------------------------------------------------------
# OptimizeResult.to_dict
# ----------------------------------------------------------------------


def test_optimize_result_to_dict_with_str_prompt() -> None:
    """単一スロット（str prompt）の to_dict は全フィールドを plain dict で返す。"""
    result = OptimizeResult(
        prompt="optimized ${var}",
        seed="seed ${var}",
        diff="--- before\n+++ after\n@@ -1 +1 @@\n-seed ${var}\n+optimized ${var}",
        train_score=0.75,
        val_score=0.5,
        history=[{"round": 0, "train_score": 0.75}],
    )
    d = result.to_dict()
    assert d == {
        "prompt": "optimized ${var}",
        "seed": "seed ${var}",
        "diff": "--- before\n+++ after\n@@ -1 +1 @@\n-seed ${var}\n+optimized ${var}",
        "train_score": 0.75,
        "val_score": 0.5,
        "history": [{"round": 0, "train_score": 0.75}],
    }


def test_optimize_result_to_dict_with_mapping_prompt_copies() -> None:
    """複数スロット（mapping prompt）の to_dict は prompt/seed/diff/history を新コピーで返す。"""
    prompt = {"a": "pa", "b": "pb"}
    seed = {"a": "sa", "b": "sb"}
    diff = {"a": "diff-a", "b": "diff-b"}
    history = [{"round": 0}]
    result = OptimizeResult(prompt=prompt, seed=seed, diff=diff, train_score=1.0, history=history)
    d = result.to_dict()
    assert d["prompt"] == {"a": "pa", "b": "pb"}
    assert d["seed"] == {"a": "sa", "b": "sb"}
    assert d["diff"] == {"a": "diff-a", "b": "diff-b"}
    # コピーであり元 dict / list と identity が異なる（外部改変から守る）。
    assert d["prompt"] is not prompt
    assert d["seed"] is not seed
    assert d["diff"] is not diff
    assert d["history"] is not history
    assert d["val_score"] is None


def test_optimize_result_defaults() -> None:
    """val_score 既定は None・history 既定は空 list・seed/diff 既定は空文字。"""
    result = OptimizeResult(prompt="p", train_score=0.0)
    assert result.val_score is None
    assert result.history == []
    assert result.seed == ""
    assert result.diff == ""


# ----------------------------------------------------------------------
# OptimizeResult.save
# ----------------------------------------------------------------------


def test_save_str_prompt_writes_text_verbatim(tmp_path: object) -> None:
    """str prompt は `${var}` 展開せずテキストをそのまま書き出す。"""
    from pathlib import Path

    target = Path(tmp_path) / "out.txt"  # type: ignore[arg-type]
    result = OptimizeResult(prompt="hello ${name}", train_score=1.0)
    result.save(target)
    assert target.read_text(encoding="utf-8") == "hello ${name}"


def test_save_mapping_prompt_writes_json(tmp_path: object) -> None:
    """mapping prompt は JSON として書き出す（`${var}` 保持・ensure_ascii=False）。"""
    from pathlib import Path

    target = Path(tmp_path) / "out.json"  # type: ignore[arg-type]
    result = OptimizeResult(prompt={"a": "プロンプト ${v}", "b": "pb"}, train_score=1.0)
    result.save(target)
    loaded = json.loads(target.read_text(encoding="utf-8"))
    assert loaded == {"a": "プロンプト ${v}", "b": "pb"}


def test_save_accepts_str_path(tmp_path: object) -> None:
    """save は str パスも受け取れる（Path へ正規化される）。"""
    from pathlib import Path

    target = Path(tmp_path) / "out.txt"  # type: ignore[arg-type]
    result = OptimizeResult(prompt="x", train_score=0.0)
    result.save(str(target))
    assert target.read_text(encoding="utf-8") == "x"


# ----------------------------------------------------------------------
# CoverageReport（Issue #47 Phase 1: graph routing 未到達 slot 検知）
# ----------------------------------------------------------------------


def test_coverage_report_is_frozen_with_expected_fields() -> None:
    """CoverageReport は frozen dataclass で covered/missing/per_case/interrupted_cases を持つ。"""
    from oai_agentspec.runtime.lightning import CoverageReport

    r = CoverageReport(
        covered=frozenset({"a"}),
        missing=frozenset({"b"}),
        per_case=(("case_x", ("a",)),),
        interrupted_cases=0,
    )
    assert r.covered == frozenset({"a"})
    assert r.missing == frozenset({"b"})
    assert r.per_case == (("case_x", ("a",)),)
    assert r.interrupted_cases == 0
    with pytest.raises((AttributeError, TypeError)):
        r.covered = frozenset()  # type: ignore[misc]  # frozen なので代入不可


def test_coverage_report_repr_omits_per_case() -> None:
    """CoverageReport の repr は per_case を展開しない（case 本文の意図せぬ露出を防ぐ）。

    per_case は利用者任意型の case 本体（PII / 機密を含みうる）をそのまま保持する診断フィールド
    で、train 件数ぶん並ぶため repr が肥大化する。ログ / トレースバックへ自動で流れる repr から
    除外し、明示アクセス（`report.per_case`）でのみ取得できるようにする。
    """
    from oai_agentspec.runtime.lightning import CoverageReport

    r = CoverageReport(
        covered=frozenset({"a"}),
        missing=frozenset({"b"}),
        per_case=(("secret-case-marker", ("a",)),),
        interrupted_cases=0,
    )
    text = repr(r)
    assert "secret-case-marker" not in text
    # 他フィールドは repr に残る（診断性の維持）。
    assert "CoverageReport" in text
    assert "interrupted_cases=0" in text
    # 明示アクセスでは中身が取れる。
    assert r.per_case == (("secret-case-marker", ("a",)),)


def test_optimize_error_accepts_and_stores_coverage() -> None:
    """OptimizeError は optional keyword-only `coverage` を受け取り self.coverage に保持する。"""
    from oai_agentspec.runtime.lightning import CoverageReport

    r = CoverageReport(
        covered=frozenset(), missing=frozenset({"x"}), per_case=(), interrupted_cases=0
    )
    err = OptimizeError(FailureKind.CONFIG_MISSING, "msg", coverage=r)
    assert err.coverage is r
    assert err.kind == FailureKind.CONFIG_MISSING
    assert err.message == "msg"


def test_optimize_error_coverage_defaults_none() -> None:
    """既存の `OptimizeError(kind, message)` 呼び出しは非破壊で通り coverage=None。"""
    err = OptimizeError(FailureKind.CONFIG_MISSING, "msg")
    assert err.coverage is None


def test_optimize_error_coverage_is_keyword_only() -> None:
    """coverage は keyword-only（位置引数として渡すと TypeError）。"""
    from oai_agentspec.runtime.lightning import CoverageReport

    r = CoverageReport(frozenset(), frozenset({"x"}), (), 0)
    with pytest.raises(TypeError):
        OptimizeError(FailureKind.CONFIG_MISSING, "msg", r)  # type: ignore[misc]


def test_coverage_report_complete_defaults_true() -> None:
    """`complete` は既定 True（train 全件を観測しきった確定判定を表す）。

    既定値付きの末尾追加であるため、既存の 4 引数構築（位置・キーワードとも）は無改修で通る。
    """
    from oai_agentspec.runtime.lightning import CoverageReport

    positional = CoverageReport(frozenset({"a"}), frozenset({"b"}), (), 0)
    keyword = CoverageReport(
        covered=frozenset({"a"}), missing=frozenset({"b"}), per_case=(), interrupted_cases=0
    )
    assert positional.complete is True
    assert keyword.complete is True


def test_coverage_report_complete_false_marks_partial_observation() -> None:
    """`complete=False` は「観測が途中で終わり missing が未確定」であることを表す。

    観測失敗経路（`TRAINER_FAILED`）で添付される部分レポートの識別子。`kind` との対応表を
    利用者に記憶させず、レポート単体で完了度が読めるようにするための自己記述フィールド。
    """
    from oai_agentspec.runtime.lightning import CoverageReport

    r = CoverageReport(
        covered=frozenset({"triage"}),
        missing=frozenset({"billing"}),
        per_case=(),
        interrupted_cases=0,
        complete=False,
    )
    assert r.complete is False


def test_coverage_report_complete_appears_in_repr() -> None:
    """`complete` は repr に出る（例外から切り離してログ・保存しても完了度が残る）。

    `per_case` の repr 抑止（PII 方針）は維持されたままであることも同時に固定する。
    """
    from oai_agentspec.runtime.lightning import CoverageReport

    r = CoverageReport(
        covered=frozenset({"a"}),
        missing=frozenset({"b"}),
        per_case=(("secret-case-marker", ("a",)),),
        interrupted_cases=0,
        complete=False,
    )
    text = repr(r)
    assert "complete=False" in text
    assert "secret-case-marker" not in text


def test_optimize_partial_shape_and_repr_suppression() -> None:
    """`OptimizePartial` は frozen で、`completed_slots` は repr に出さない。

    prompt テキストは利用者資産だが、例外 repr がログへ乗る経路に本文を出さない方針は
    `CoverageReport.per_case` と同一。明示アクセス（`partial.completed_slots`）は可。
    """
    from oai_agentspec.runtime.lightning import OptimizePartial

    p = OptimizePartial(
        completed_slots={"triage": "PROMPT-BODY-MARKER"},
        history=[
            {
                "slot": "triage",
                "best_score": 0.8,
                "best_version": 3,
                "placeholder_fallback": False,
            }
        ],
        failed_slot="billing",
    )
    text = repr(p)
    assert "PROMPT-BODY-MARKER" not in text
    assert "failed_slot='billing'" in text
    assert p.completed_slots == {"triage": "PROMPT-BODY-MARKER"}
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.failed_slot = "x"  # type: ignore[misc]


def test_optimize_partial_failed_slot_none_means_all_slots_done() -> None:
    """`failed_slot=None` は「全 slot 完了・スコア再計算段の失敗」を表す。"""
    from oai_agentspec.runtime.lightning import OptimizePartial

    p = OptimizePartial(completed_slots={"a": "t"}, history=[], failed_slot=None)
    assert p.failed_slot is None


def test_optimize_error_partial_defaults_none_and_keyword_only() -> None:
    """`OptimizeError.partial` は keyword-only・既定 None（既存呼び出しは非破壊）。"""
    from oai_agentspec.runtime.lightning import OptimizePartial

    err = OptimizeError(FailureKind.TRAINER_FAILED, "msg")
    assert err.partial is None

    p = OptimizePartial(completed_slots={}, history=[], failed_slot=None)
    err2 = OptimizeError(FailureKind.TRAINER_FAILED, "msg", partial=p)
    assert err2.partial is p


def test_format_exception_message_includes_type_and_nonempty_body_only() -> None:
    """`_format_exception_message` は型名を常に出し、本文は非空のときだけ連結する。

    `str(TimeoutError())` は空文字で、無条件連結だと `'TimeoutError: '` とコロン終わりの
    情報ゼロ文字列になる（pre-flight で修正した穴と同型。共有ヘルパで drift を防ぐ）。
    """
    from oai_agentspec.runtime.lightning.types import _format_exception_message

    assert _format_exception_message(RuntimeError("boom")) == "RuntimeError: boom"
    assert _format_exception_message(TimeoutError()) == "TimeoutError"


def test_coverage_report_invalid_cases_defaults_zero_and_appears_in_repr() -> None:
    """`invalid_cases` は既定 0 で repr に出る（無効化の存在が report 単体で読める）。

    候補無効化（観測なし）と「実行済み・観測空」を区別する診断カウンタ。既定値付き
    末尾追加のため既存の構築（位置・キーワードとも）は無改修で通る。
    """
    from oai_agentspec.runtime.lightning import CoverageReport

    positional = CoverageReport(frozenset(), frozenset({"x"}), (), 0)
    assert positional.invalid_cases == 0

    r = CoverageReport(
        covered=frozenset(),
        missing=frozenset({"x"}),
        per_case=(),
        interrupted_cases=0,
        complete=True,
        invalid_cases=2,
    )
    assert r.invalid_cases == 2
    assert "invalid_cases=2" in repr(r)


def test_coverage_report_per_case_accepts_none_as_invalidated_marker() -> None:
    """`per_case` の値は 3 値: None = 無効化 / () = 実行済み観測空 / 非空 = 到達観測。

    None と () を同一視すると「rollout が実行されたか」が report から読めず、
    無効化の誤診断（未到達確定と主張）を再生産する。
    """
    from oai_agentspec.runtime.lightning import CoverageReport

    r = CoverageReport(
        covered=frozenset({"triage"}),
        missing=frozenset({"billing"}),
        per_case=(("case-a", None), ("case-b", ()), ("case-c", ("triage",))),
        interrupted_cases=0,
        invalid_cases=1,
    )
    values = [steps for _, steps in r.per_case]
    assert values == [None, (), ("triage",)]


# ----------------------------------------------------------------------
# CoverageReport.complete の構築時 bool 型検証
# ----------------------------------------------------------------------


def _coverage_report_kwargs(**overrides: object) -> dict[str, object]:
    """CoverageReport の最小構築 kwargs（bool 検証テストの共通引数）。"""
    base: dict[str, object] = {
        "covered": frozenset({"a"}),
        "missing": frozenset({"b"}),
        "per_case": (),
        "interrupted_cases": 0,
    }
    base.update(overrides)
    return base


def test_coverage_report_complete_none_raises() -> None:
    """complete=None は bool でないため構築時 ValueError（メッセージ全文を pin）。

    観測完了度フラグが黙って falsy になると「部分レポートを完走扱いする」誤診断が起きるため、
    構築時に fail-fast する。
    """
    from oai_agentspec.runtime.lightning import CoverageReport

    with pytest.raises(ValueError, match=re.escape("complete must be a bool, got 'NoneType'")):
        CoverageReport(**_coverage_report_kwargs(complete=None))  # type: ignore[arg-type]


def test_coverage_report_complete_str_raises() -> None:
    """complete="no" は truthy な文字列だが ValueError で弾く（silent 受理しない）。"""
    from oai_agentspec.runtime.lightning import CoverageReport

    with pytest.raises(ValueError, match=re.escape("complete must be a bool, got 'str'")):
        CoverageReport(**_coverage_report_kwargs(complete="no"))  # type: ignore[arg-type]


def test_coverage_report_complete_int_zero_raises() -> None:
    """complete=0（int）は bool でないため ValueError（bool は int のサブクラスだが逆は不可）。"""
    from oai_agentspec.runtime.lightning import CoverageReport

    with pytest.raises(ValueError, match="complete"):
        CoverageReport(**_coverage_report_kwargs(complete=0))  # type: ignore[arg-type]


def test_coverage_report_complete_bool_constructs() -> None:
    """complete へ True / False を渡した構築は成功する（正常系の維持）。"""
    from oai_agentspec.runtime.lightning import CoverageReport

    assert CoverageReport(**_coverage_report_kwargs(complete=True)).complete is True  # type: ignore[arg-type]
    assert CoverageReport(**_coverage_report_kwargs(complete=False)).complete is False  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# SlotSegment.tune の構築時 bool 型検証
# ----------------------------------------------------------------------


def test_slot_segment_tune_none_raises() -> None:
    """tune=None は bool でないため構築時 ValueError（メッセージ全文を pin）。

    tune は APO 最適化対象かの二値フラグで、falsy な誤値を silent 受理すると
    「最適化したいセグメントが固定扱いになる」silent failure になる。
    """
    from oai_agentspec.runtime.lightning.types import SlotSegment

    with pytest.raises(ValueError, match=re.escape("tune must be a bool, got 'NoneType'")):
        SlotSegment(ref="base:main", text="seed", tune=None)  # type: ignore[arg-type]


def test_slot_segment_tune_str_raises() -> None:
    """tune="no" は truthy な文字列だが ValueError で弾く（silent 受理しない）。"""
    from oai_agentspec.runtime.lightning.types import SlotSegment

    with pytest.raises(ValueError, match=re.escape("tune must be a bool, got 'str'")):
        SlotSegment(ref="base:main", text="seed", tune="no")  # type: ignore[arg-type]


def test_slot_segment_tune_int_zero_raises() -> None:
    """tune=0（int）は bool でないため ValueError。"""
    from oai_agentspec.runtime.lightning.types import SlotSegment

    with pytest.raises(ValueError, match="tune"):
        SlotSegment(ref="base:main", text="seed", tune=0)  # type: ignore[arg-type]


def test_slot_segment_tune_bool_constructs() -> None:
    """tune へ True / False を渡した構築は成功する（正常系の維持）。"""
    from oai_agentspec.runtime.lightning.types import SlotSegment

    assert SlotSegment(ref="agent:triage", text="seed", tune=True).tune is True
    assert SlotSegment(ref="part:style", text="fixed", tune=False).tune is False
