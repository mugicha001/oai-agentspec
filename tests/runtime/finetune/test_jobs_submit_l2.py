"""L2: `submit_job` を実 adapter 経由で通し、ファイル処理完了待機の契約を固定する。

`test_jobs_l1.py` は adapter 関数自体を fake へ差し替えるため、client レベルの制約
（アップロード直後のファイルは `status="pending"`・未処理 id でのジョブ作成は 400）が
`submit_job` の経路へ届かない。本モジュールは **adapter を差し替えず client のみ fake 化**し、
「`submit_job` -> `_adapters.finetune` -> client」の鎖を測る（実 API で壊れた経路）。

固定する契約:
    - データ経路（レコード列 / `Path`）は upload の直後に処理完了を待ってからジョブを作る
    - `str`（アップロード済みファイル id）経路では upload も待機も行わない
    - `file_wait_timeout` が `wait_for_processing` の `max_wait_seconds` まで届く
      （`poll_interval` は lib 定数 2.0 で固定）
    - `file_wait_timeout` の非正値は API 呼び出し前に `CONFIG_MISSING`

実 adapter は openai を import するため層は L2（`@pytest.mark.integration`）。ネットワーク
通信は行わない（client は `_helpers.fake_finetune_client.FakeFineTuneClient`）。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from oai_agentspec.runtime.finetune import FineTuneError, FineTuneFailureKind
from oai_agentspec.runtime.finetune.jobs import submit_job
from oai_agentspec.runtime.finetune.types import JobRef

from _helpers.fake_finetune_client import FakeFineTuneClient

pytestmark = pytest.mark.integration

_MODEL = "gpt-4.1-mini-2025-04-14"
_RECORDS = [{"messages": [{"role": "user", "content": "a"}]}]


def _val_path(tmp_path: Path) -> Path:
    """検証データの JSONL を tmp_path へ書き出す。"""
    source = tmp_path / "val.jsonl"
    source.write_text('{"messages": []}\n', encoding="utf-8")
    return source


async def test_data_path_submits_job_after_files_are_processed(tmp_path: Path) -> None:
    """レコード列 + Path のデータ経路で JobRef が返る（未処理 id で 400 にならない）。

    待機を持たない実装では、fake が実 API と同じ 400
    （`The specified file reference must point to a completed file import.`）を返すため
    `API_ERROR` で落ちる。
    """
    client = FakeFineTuneClient(job_id="ftjob-live")

    ref = await submit_job(
        client,
        train=_RECORDS,
        val=_val_path(tmp_path),
        model=_MODEL,
        method="sft",
    )

    assert isinstance(ref, JobRef)
    assert ref.job_id == "ftjob-live"
    assert client.statuses[ref.training_file_id] == "processed"
    assert ref.validation_file_id is not None
    assert client.statuses[ref.validation_file_id] == "processed"


async def test_upload_and_wait_are_paired_per_file_before_job_creation(tmp_path: Path) -> None:
    """SDK 呼び出し順は train / val それぞれ upload -> wait で、最後に jobs.create になる。

    待機を落とすと 2 要素が欠落し、かつ `jobs.create` が 400 になる二重で赤くなる。
    """
    client = FakeFineTuneClient()

    await submit_job(
        client,
        train=_RECORDS,
        val=_val_path(tmp_path),
        model=_MODEL,
        method="sft",
    )

    assert client.calls == [
        "files.create",
        "wait_for_processing",
        "files.create",
        "wait_for_processing",
        "jobs.create",
    ]


async def test_str_file_id_path_neither_uploads_nor_waits() -> None:
    """`str`（アップロード済み id）経路は upload も待機も行わない（利用者責任・ADR 0032）。"""
    client = FakeFineTuneClient()

    ref = await submit_job(client, train="file-abc", model=_MODEL, method="sft")

    assert client.calls == ["jobs.create"]
    assert client.wait_calls == []
    assert ref.training_file_id == "file-abc"
    assert ref.validation_file_id is None


async def test_file_wait_timeout_reaches_max_wait_seconds() -> None:
    """`file_wait_timeout` は `max_wait_seconds` へ届き、`poll_interval` は 2.0 に固定される。

    SDK 既定（`poll_interval=5.0` / `max_wait_seconds=1800`）へ暗黙に依存しないことの pin。
    """
    client = FakeFineTuneClient()

    await submit_job(
        client,
        train=_RECORDS,
        model=_MODEL,
        method="sft",
        file_wait_timeout=123.0,
    )

    assert len(client.wait_calls) == 1
    call = client.wait_calls[0]
    assert call["max_wait_seconds"] == 123.0
    assert call["poll_interval"] == 2.0


@pytest.mark.parametrize("timeout", [0, -1.0])
async def test_non_positive_file_wait_timeout_fails_before_any_api_call(timeout: float) -> None:
    """`file_wait_timeout` の非正値は API 呼び出し前に CONFIG_MISSING で失敗する。"""
    client = FakeFineTuneClient()

    with pytest.raises(FineTuneError) as exc_info:
        await submit_job(
            client,
            train=_RECORDS,
            model=_MODEL,
            method="sft",
            file_wait_timeout=timeout,
        )

    assert exc_info.value.kind == FineTuneFailureKind.CONFIG_MISSING
    assert client.calls == []
