"""L1: 最適化の plain 結果型・スロット型（外部 SDK 非依存）を検証する。

`FailureKind`（StrEnum 値）・`OptimizeError`（kind / message 保持）・`Slot`（frozen・vars 既定）・
`OptimizeResult`（`to_dict` / `save`・単一 str / mapping 分岐・`${var}` 保持・PromptStore 非書込）を
網羅する。すべて純データ操作で外部依存なし（`@pytest.mark.unit`）。
"""

from __future__ import annotations

import json

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
