"""L1: `runtime/finetune` 公開窓口の contract と extra 未導入契約を検証する。

窓口 `__all__` のメンバ集合（FR-9・過大側 / 過小側の双方を `==` で固定）・全シンボルの取得
可能性・コア `__all__` への非混入に加えて、openai の import 失敗を注入した状態でも窓口の
import とシンボル取得が成功すること（遅延 import 境界）と、ジョブ管理関数を実際に呼んだ
ときに `FineTuneError(EXTRA_MISSING)` へ変換されること（NFR-6）を検証する。実 API 通信は
行わない（`@pytest.mark.unit`）。
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import pytest

import oai_agentspec
from oai_agentspec.runtime import finetune as window
from oai_agentspec.runtime.finetune import FineTuneError, FineTuneFailureKind

pytestmark = pytest.mark.unit

_WINDOW_SYMBOLS = {
    # 変換 / 検証エントリ
    "to_sft_dataset",
    "to_dpo_dataset",
    "validate_dataset",
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
}


# ----------------------------------------------------------------------
# 公開窓口の contract（FR-9）
# ----------------------------------------------------------------------


def test_window_all_member_set_is_pinned() -> None:
    """窓口 `__all__` は 15 件のシンボル集合で固定する（FR-9）。

    `==` の集合比較により、意図しない公開追加（過大側）と公開喪失（過小側）の双方を
    検知する。
    """
    assert set(window.__all__) == _WINDOW_SYMBOLS
    assert len(window.__all__) == len(_WINDOW_SYMBOLS)


def test_window_all_symbols_are_resolvable() -> None:
    """`__all__` の全要素が窓口から実際に取得できる（宣言と実体の乖離を防ぐ）。"""
    missing = [name for name in window.__all__ if not hasattr(window, name)]
    assert missing == []


def test_finetune_symbols_are_absent_from_core_all() -> None:
    """finetune のシンボルはコア `oai_agentspec.__all__` へ 1 件も混入しない（FR-9）。

    公開 API は `runtime.finetune` 窓口へ集約し、コアの公開契約を変更しない。
    """
    assert set(oai_agentspec.__all__).isdisjoint(_WINDOW_SYMBOLS)


def test_core_import_still_succeeds() -> None:
    """コア `import oai_agentspec` は finetune 追加後も成功し `__all__` を解決できる。"""
    module = importlib.import_module("oai_agentspec")
    assert all(hasattr(module, name) for name in module.__all__)


# ----------------------------------------------------------------------
# extra 未導入契約（FR-9 / NFR-2 / NFR-6）
# ----------------------------------------------------------------------


def _write_jsonl(tmp_path: Path) -> Path:
    """アップロード対象のローカル JSONL を tmp_path へ書き出す。"""
    target = tmp_path / "train.jsonl"
    target.write_text('{"messages": []}\n', encoding="utf-8")
    return target


def _block_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    """openai の import 失敗を注入する（`import openai` が ImportError になる）。"""
    monkeypatch.setitem(sys.modules, "openai", None)


def test_window_imports_without_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    """openai を import できない状態でも窓口の import とシンボル取得が成功する。

    窓口は import 時に openai へ到達しない（SDK 接触は呼び出し時の遅延 import）ため、
    extra 未導入環境でも変換 / 検証ヘルパは使える（NFR-2）。
    """
    _block_openai(monkeypatch)
    for name in [n for n in list(sys.modules) if n.startswith("oai_agentspec.runtime.finetune")]:
        monkeypatch.delitem(sys.modules, name)

    module = importlib.import_module("oai_agentspec.runtime.finetune")

    assert set(module.__all__) == _WINDOW_SYMBOLS
    assert all(hasattr(module, name) for name in module.__all__)


async def test_submit_job_raises_extra_missing_without_openai(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """openai 未導入の状態で submit_job を呼ぶと EXTRA_MISSING へ変換される（NFR-6）。"""
    _block_openai(monkeypatch)

    with pytest.raises(FineTuneError) as exc_info:
        await window.submit_job(
            object(), train="file-abc123", model="gpt-4.1-mini-2025-04-14", method="sft"
        )

    assert exc_info.value.kind == FineTuneFailureKind.EXTRA_MISSING
    assert "oai-agentspec[finetune]" in exc_info.value.message


@pytest.mark.parametrize(
    "train_factory",
    [
        pytest.param(lambda _tmp: [{"messages": [{"role": "user", "content": "a"}]}], id="records"),
        pytest.param(lambda tmp: _write_jsonl(tmp), id="path"),
    ],
)
async def test_submit_job_raises_extra_missing_on_upload_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, train_factory: Any
) -> None:
    """アップロードを要する受理形でも EXTRA_MISSING へ変換される（ファイル解決フェーズ）。

    `train` が str（ファイル id）の場合はアップロードを通らずジョブ作成側の変換で
    EXTRA_MISSING になるため、レコード列 / `Path` の経路を独立に pin する。
    """
    _block_openai(monkeypatch)

    with pytest.raises(FineTuneError) as exc_info:
        await window.submit_job(
            object(),
            train=train_factory(tmp_path),
            model="gpt-4.1-mini-2025-04-14",
            method="sft",
        )

    assert exc_info.value.kind == FineTuneFailureKind.EXTRA_MISSING
    assert "oai-agentspec[finetune]" in exc_info.value.message
    assert isinstance(exc_info.value.__cause__, ImportError)


async def test_submit_job_raises_extra_missing_when_only_val_needs_upload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """train がファイル id で val だけアップロードが要る場合も EXTRA_MISSING になる。"""
    _block_openai(monkeypatch)

    with pytest.raises(FineTuneError) as exc_info:
        await window.submit_job(
            object(),
            train="file-abc123",
            val=[{"messages": [{"role": "user", "content": "a"}]}],
            model="gpt-4.1-mini-2025-04-14",
            method="sft",
        )

    assert exc_info.value.kind == FineTuneFailureKind.EXTRA_MISSING
    assert "oai-agentspec[finetune]" in exc_info.value.message


async def test_get_job_raises_extra_missing_without_openai(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """openai 未導入の状態で get_job を呼ぶと EXTRA_MISSING へ変換される（NFR-6）。"""
    _block_openai(monkeypatch)

    with pytest.raises(FineTuneError) as exc_info:
        await window.get_job(object(), "ftjob-1")

    assert exc_info.value.kind == FineTuneFailureKind.EXTRA_MISSING
    assert "oai-agentspec[finetune]" in exc_info.value.message


async def test_wait_job_raises_extra_missing_without_openai(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """openai 未導入の状態で wait_job を呼ぶと EXTRA_MISSING へ変換される（NFR-6）。"""
    _block_openai(monkeypatch)

    with pytest.raises(FineTuneError) as exc_info:
        await window.wait_job(object(), "ftjob-1", timeout=10.0)

    assert exc_info.value.kind == FineTuneFailureKind.EXTRA_MISSING
    assert "oai-agentspec[finetune]" in exc_info.value.message


async def test_extra_missing_chains_original_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """EXTRA_MISSING は元の ImportError を `__cause__` として保持する（原因を失わない）。"""
    _block_openai(monkeypatch)

    with pytest.raises(FineTuneError) as exc_info:
        await window.get_job(object(), "ftjob-1")

    cause: Any = exc_info.value.__cause__
    assert isinstance(cause, ImportError)
