"""L2: Fine-Tuning ジョブ管理の SDK 窓口（`_adapters/finetune.py`）を検証する。

`_require_openai`（遅延 import ガード・未導入時は生 ImportError）・`upload_file`
（`purpose="fine-tune"` 明示 + filename 付きタプル）・`create_job`（判断 6 の分配規則:
`model` / `training_file` 以外はすべて SDK の `extra_body` へ一括委譲）・`retrieve_job`
（状態 / fine_tuned_model / 失敗理由の plain 取り出し）・`wait_file_processed`
（`poll_interval` / `max_wait_seconds` の明示指定・終端 status の fail-closed 変換・
上限超過の TIMEOUT 変換・ADR 0032）と、openai 例外 →
`FineTuneError(API_ERROR)` 変換（理由文言の保全・`FineTuneError` の二重変換回避）を
網羅する。client はすべて fake（`tests/_adapters/test_lightning_adapters_l2.py` の
`_FakeGatewayClient` 様式）で、実 API 通信は行わない（NFR-4）。
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest

from oai_agentspec._adapters import finetune as _ft
from oai_agentspec._adapters.finetune import (
    _require_openai,
    create_job,
    retrieve_job,
    upload_file,
)
from oai_agentspec.runtime.finetune import FineTuneError, FineTuneFailureKind

from _helpers.fake_finetune_client import FakeFineTuneClient

pytestmark = pytest.mark.integration

_SENTINEL_REASON = "suffix must be at most 18 characters"


# ----------------------------------------------------------------------
# fake client
# ----------------------------------------------------------------------


class _FakeFineTuneClient:
    """`files.create` / `fine_tuning.jobs.create` / `.retrieve` を捕捉する疑似 client。

    各メソッドの呼び出し kwargs（retrieve は args）を list へ append し、`error` に例外を
    設定するとその呼び出しで送出する。
    """

    def __init__(
        self,
        *,
        file_id: str = "file-abc",
        job: Any = None,
        error: BaseException | None = None,
    ) -> None:
        self.file_calls: list[dict[str, Any]] = []
        self.create_calls: list[dict[str, Any]] = []
        self.retrieve_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        outer = self

        class _Files:
            async def create(self, **kwargs: Any) -> Any:
                outer.file_calls.append(kwargs)
                if error is not None:
                    raise error
                return SimpleNamespace(id=file_id)

        class _Jobs:
            async def create(self, **kwargs: Any) -> Any:
                outer.create_calls.append(kwargs)
                if error is not None:
                    raise error
                return job if job is not None else SimpleNamespace(id="ftjob-123")

            async def retrieve(self, *args: Any, **kwargs: Any) -> Any:
                outer.retrieve_calls.append((args, kwargs))
                if error is not None:
                    raise error
                return job

        class _FineTuning:
            jobs = _Jobs()

        self.files = _Files()
        self.fine_tuning = _FineTuning()


def _not_found_error() -> Exception:
    """openai.NotFoundError（404）を最小構成で作る（理由文言つき）。"""
    import httpx
    import openai

    request = httpx.Request("GET", "https://api.example.com/v1/fine_tuning/jobs/ftjob-x")
    response = httpx.Response(404, request=request, json={"detail": _SENTINEL_REASON})
    return openai.NotFoundError(
        _SENTINEL_REASON, response=response, body={"detail": _SENTINEL_REASON}
    )


def _connection_error() -> Exception:
    """openai.APIConnectionError（通信断）を最小構成で作る（理由文言つき）。"""
    import httpx
    import openai

    request = httpx.Request("POST", "https://api.example.com/v1/fine_tuning/jobs")
    return openai.APIConnectionError(message=_SENTINEL_REASON, request=request)


# ----------------------------------------------------------------------
# _require_openai（遅延 import ガード）
# ----------------------------------------------------------------------


def test_require_openai_returns_module() -> None:
    """openai 導入済み環境では openai モジュールを返す（遅延 import が通る）。"""
    module = _require_openai()
    assert module is sys.modules["openai"]


def test_require_openai_raises_plain_import_error_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """import 失敗を注入すると生 ImportError（FineTuneError ではない）を送出する。

    extra 未導入の変換（`EXTRA_MISSING`）は `jobs.py` 側の責務であり、adapter は
    導入ヒント付きの生 ImportError に留める（NFR-6: モックで import 失敗を再現可能）。
    """
    monkeypatch.setitem(sys.modules, "openai", None)
    with pytest.raises(ImportError) as exc_info:
        _require_openai()
    assert not isinstance(exc_info.value, FineTuneError)
    assert "finetune" in str(exc_info.value)


# ----------------------------------------------------------------------
# upload_file（purpose 明示 + filename 付きタプル）
# ----------------------------------------------------------------------


async def test_upload_file_passes_purpose_and_named_tuple_for_train() -> None:
    """学習ファイルは `purpose="fine-tune"` と `("train.jsonl", data)` タプルで送る。

    filename なしの生 bytes だとプラットフォーム側で拡張子を認識できないため、
    タプルの第 1 要素（filename）まで固定する。
    """
    client = _FakeFineTuneClient(file_id="file-train")
    data = b'{"messages": []}\n'

    file_id = await upload_file(client, data=data, filename="train.jsonl")

    assert file_id == "file-train"
    assert len(client.file_calls) == 1
    kwargs = client.file_calls[0]
    assert set(kwargs) == {"file", "purpose"}
    assert kwargs["purpose"] == "fine-tune"
    assert isinstance(kwargs["file"], tuple)
    assert kwargs["file"][0] == "train.jsonl"
    assert kwargs["file"][1] == data


async def test_upload_file_uses_validation_filename() -> None:
    """検証ファイルは `("validation.jsonl", data)` タプルで送る（filename を取り違えない）。"""
    client = _FakeFineTuneClient(file_id="file-val")
    data = b'{"messages": []}\n'

    file_id = await upload_file(client, data=data, filename="validation.jsonl")

    assert file_id == "file-val"
    assert client.file_calls[0]["file"][0] == "validation.jsonl"
    assert client.file_calls[0]["file"][1] == data


async def test_upload_file_calls_sdk_once() -> None:
    """アップロードは単発呼び出し（再試行・分割アップロードをしない）。"""
    client = _FakeFineTuneClient()
    await upload_file(client, data=b"x", filename="train.jsonl")
    assert len(client.file_calls) == 1


async def test_upload_file_converts_openai_error() -> None:
    """アップロードの openai 例外は API_ERROR へ変換し理由文言を保全する。"""
    client = _FakeFineTuneClient(error=_connection_error())
    with pytest.raises(FineTuneError) as exc_info:
        await upload_file(client, data=b"x", filename="train.jsonl")
    assert exc_info.value.kind == FineTuneFailureKind.API_ERROR
    assert _SENTINEL_REASON in exc_info.value.message


# ----------------------------------------------------------------------
# create_job（判断 6: model / training_file 以外は extra_body へ一括委譲）
# ----------------------------------------------------------------------


def _full_body() -> dict[str, Any]:
    """SDK ネイティブ引数（model / training_file）と非ネイティブ設定を含む body。"""
    return {
        "model": "gpt-4o-mini-2024-07-18",
        "training_file": "file-train",
        "method": {"type": "supervised"},
        "validation_file": "file-val",
        "suffix": "acme",
        "seed": 42,
        "trainingType": "Standard",
    }


async def test_create_job_returns_job_id() -> None:
    """ジョブ作成のレスポンスから job id を取り出して返す。"""
    client = _FakeFineTuneClient(job=SimpleNamespace(id="ftjob-xyz"))
    job_id = await create_job(client, _full_body())
    assert job_id == "ftjob-xyz"
    assert len(client.create_calls) == 1


async def test_create_job_sends_only_model_training_file_and_extra_body() -> None:
    """SDK へ渡す kwargs は model / training_file / extra_body の 3 つに限る（判断 6）。

    SDK ネイティブ引数のキー名リストを adapter に持たない設計のため、
    suffix / seed / trainingType 等を個別の SDK 引数へ分配する実装だと RED になる。
    """
    client = _FakeFineTuneClient()
    await create_job(client, _full_body())

    kwargs = client.create_calls[0]
    assert set(kwargs) == {"model", "training_file", "extra_body"}
    assert kwargs["model"] == "gpt-4o-mini-2024-07-18"
    assert kwargs["training_file"] == "file-train"


async def test_create_job_puts_remaining_keys_into_extra_body() -> None:
    """model / training_file 以外のキーはすべて extra_body の中へ入る（欠落も混入もしない）。"""
    client = _FakeFineTuneClient()
    await create_job(client, _full_body())

    assert client.create_calls[0]["extra_body"] == {
        "method": {"type": "supervised"},
        "validation_file": "file-val",
        "suffix": "acme",
        "seed": 42,
        "trainingType": "Standard",
    }


async def test_create_job_minimal_body_sends_empty_capable_extra_body() -> None:
    """最小 body（model / training_file / method）でも method は extra_body へ載る。"""
    client = _FakeFineTuneClient()
    await create_job(
        client,
        {"model": "m", "training_file": "file-train", "method": {"type": "dpo"}},
    )

    kwargs = client.create_calls[0]
    assert set(kwargs) == {"model", "training_file", "extra_body"}
    assert kwargs["extra_body"] == {"method": {"type": "dpo"}}


async def test_create_job_does_not_mutate_caller_body() -> None:
    """呼び出し側の body dict を破壊しない（pop で取り出さない）。"""
    client = _FakeFineTuneClient()
    body = _full_body()
    await create_job(client, body)
    assert body == _full_body()


async def test_create_job_converts_openai_error_preserving_reason() -> None:
    """ジョブ作成の openai 例外は API_ERROR へ変換し、理由文言を保全する。

    suffix 長・model と method の組み合わせ可否は lib で検証せずプラットフォームの
    エラー文言をそのまま利用者へ渡す（FR-5 / FR-10）。
    """
    client = _FakeFineTuneClient(error=_not_found_error())
    with pytest.raises(FineTuneError) as exc_info:
        await create_job(client, _full_body())
    assert exc_info.value.kind == FineTuneFailureKind.API_ERROR
    assert _SENTINEL_REASON in exc_info.value.message


async def test_create_job_reraises_finetune_error_without_rewrapping() -> None:
    """既に FineTuneError の失敗は kind / message を保ったまま re-raise する（二重変換しない）。"""
    original = FineTuneError(FineTuneFailureKind.CONFIG_MISSING, "衝突したキー: suffix")
    client = _FakeFineTuneClient(error=original)
    with pytest.raises(FineTuneError) as exc_info:
        await create_job(client, _full_body())
    assert exc_info.value is original
    assert exc_info.value.kind == FineTuneFailureKind.CONFIG_MISSING


# ----------------------------------------------------------------------
# retrieve_job（状態 / model_ref / 失敗理由の plain 取り出し）
# ----------------------------------------------------------------------


async def test_retrieve_job_passes_job_id_positionally() -> None:
    """`jobs.retrieve(job_id)` を単発で呼ぶ（job_id を位置引数で渡す）。"""
    job = SimpleNamespace(id="ftjob-1", status="running", fine_tuned_model=None, error=None)
    client = _FakeFineTuneClient(job=job)

    await retrieve_job(client, "ftjob-1")

    assert len(client.retrieve_calls) == 1
    args, _kwargs = client.retrieve_calls[0]
    assert args[0] == "ftjob-1"


async def test_retrieve_job_returns_plain_payload_on_success() -> None:
    """成功時は id / status / fine_tuned_model / error_message を plain dict で返す。"""
    job = SimpleNamespace(
        id="ftjob-1",
        status="succeeded",
        fine_tuned_model="ft:gpt-4o-mini:acme::abc123",
        error=None,
    )
    client = _FakeFineTuneClient(job=job)

    payload = await retrieve_job(client, "ftjob-1")

    assert payload == {
        "id": "ftjob-1",
        "status": "succeeded",
        "fine_tuned_model": "ft:gpt-4o-mini:acme::abc123",
        "error_message": None,
    }


async def test_retrieve_job_extracts_error_message_on_failure() -> None:
    """失敗時は error オブジェクトの message を error_message として取り出す。"""
    job = SimpleNamespace(
        id="ftjob-1",
        status="failed",
        fine_tuned_model=None,
        error=SimpleNamespace(code="invalid_training_file", message=_SENTINEL_REASON),
    )
    client = _FakeFineTuneClient(job=job)

    payload = await retrieve_job(client, "ftjob-1")

    assert payload["status"] == "failed"
    assert payload["error_message"] == _SENTINEL_REASON
    assert payload["fine_tuned_model"] is None


async def test_retrieve_job_keeps_unknown_status_string_as_is() -> None:
    """未知の状態文字列も加工せずそのまま返す（写像はロジック層の責務）。"""
    job = SimpleNamespace(
        id="ftjob-1", status="validating_files", fine_tuned_model=None, error=None
    )
    client = _FakeFineTuneClient(job=job)

    payload = await retrieve_job(client, "ftjob-1")

    assert payload["status"] == "validating_files"


async def test_retrieve_job_converts_openai_error_preserving_reason() -> None:
    """存在しない job_id 等の openai 例外は API_ERROR へ変換し理由文言を保全する（FR-6）。"""
    client = _FakeFineTuneClient(error=_not_found_error())
    with pytest.raises(FineTuneError) as exc_info:
        await retrieve_job(client, "ftjob-missing")
    assert exc_info.value.kind == FineTuneFailureKind.API_ERROR
    assert _SENTINEL_REASON in exc_info.value.message


async def test_retrieve_job_reraises_finetune_error_without_rewrapping() -> None:
    """FineTuneError はそのまま re-raise する（API_ERROR へ上書きしない）。"""
    original = FineTuneError(FineTuneFailureKind.TIMEOUT, "待機がタイムアウトしました")
    client = _FakeFineTuneClient(error=original)
    with pytest.raises(FineTuneError) as exc_info:
        await retrieve_job(client, "ftjob-1")
    assert exc_info.value is original
    assert exc_info.value.kind == FineTuneFailureKind.TIMEOUT


# ----------------------------------------------------------------------
# wait_file_processed（ファイル処理完了待機・ADR 0032）
#
# モジュール属性経由（`_ft.wait_file_processed`）で呼ぶ。from-import にすると未実装時に
# 収集エラーとなり、本ファイルの既存テストまで巻き添えで実行不能になるため。
# ----------------------------------------------------------------------


async def test_wait_file_processed_passes_both_sdk_params_explicitly() -> None:
    """`poll_interval=2.0` と `max_wait_seconds=<timeout>` を明示指定する。

    SDK 既定（`poll_interval=5.0` / `max_wait_seconds=1800`）へ暗黙依存すると、既定値の
    版差でポーリング頻度・上限が変わる。両値を lib 側で固定することの pin。
    """
    client = FakeFineTuneClient()
    file_id = await upload_file(client, data=b"x", filename="train.jsonl")

    await _ft.wait_file_processed(client, file_id, timeout=300.0)

    assert len(client.wait_calls) == 1
    assert client.wait_calls[0] == {
        "file_id": file_id,
        "poll_interval": 2.0,
        "max_wait_seconds": 300.0,
    }


async def test_wait_file_processed_returns_file_id_when_processed() -> None:
    """`processed` へ到達したら受け取った file id をそのまま返す。"""
    client = FakeFineTuneClient()
    file_id = await upload_file(client, data=b"x", filename="train.jsonl")

    assert await _ft.wait_file_processed(client, file_id, timeout=10.0) == file_id


async def test_wait_file_processed_error_status_becomes_api_error_with_details() -> None:
    """`status="error"` は API_ERROR へ変換し、status_details を逐語で含める。

    SDK は error 終端でも例外を投げずに返すため、lib 側で fail-closed に倒す必要がある。
    """
    client = FakeFineTuneClient(wait_status="error", status_details=_SENTINEL_REASON)
    file_id = await upload_file(client, data=b"x", filename="train.jsonl")

    with pytest.raises(FineTuneError) as exc_info:
        await _ft.wait_file_processed(client, file_id, timeout=10.0)

    assert exc_info.value.kind == FineTuneFailureKind.API_ERROR
    assert _SENTINEL_REASON in exc_info.value.message


async def test_wait_file_processed_deleted_status_becomes_api_error() -> None:
    """`status="deleted"` も API_ERROR（processed 以外は一律 fail-closed）。"""
    client = FakeFineTuneClient(wait_status="deleted")
    file_id = await upload_file(client, data=b"x", filename="train.jsonl")

    with pytest.raises(FineTuneError) as exc_info:
        await _ft.wait_file_processed(client, file_id, timeout=10.0)

    assert exc_info.value.kind == FineTuneFailureKind.API_ERROR


async def test_wait_file_processed_runtime_error_becomes_timeout() -> None:
    """SDK の上限超過 `RuntimeError` は TIMEOUT へ変換し、message に file id を含める。"""
    client = FakeFineTuneClient(wait_error=RuntimeError("Giving up on waiting"))
    file_id = await upload_file(client, data=b"x", filename="train.jsonl")

    with pytest.raises(FineTuneError) as exc_info:
        await _ft.wait_file_processed(client, file_id, timeout=10.0)

    assert exc_info.value.kind == FineTuneFailureKind.TIMEOUT
    assert file_id in exc_info.value.message


async def test_wait_file_processed_openai_error_becomes_api_error() -> None:
    """openai 例外は API_ERROR（`except RuntimeError` が総括 except より前にある裏取り）。"""
    client = FakeFineTuneClient(wait_error=_connection_error())
    file_id = await upload_file(client, data=b"x", filename="train.jsonl")

    with pytest.raises(FineTuneError) as exc_info:
        await _ft.wait_file_processed(client, file_id, timeout=10.0)

    assert exc_info.value.kind == FineTuneFailureKind.API_ERROR
    assert _SENTINEL_REASON in exc_info.value.message


async def test_upload_file_does_not_wait_for_processing() -> None:
    """`upload_file` は `files.create` だけを呼ぶ（1 関数 = 1 SDK 呼び出しの核）。

    待機を `upload_file` へ内包すると、jobs 層が待機の有無・上限を制御できなくなる
    （`wait_job` と同型に jobs 層が制御を持つ・判断 1）。
    """
    client = FakeFineTuneClient()

    await upload_file(client, data=b"x", filename="train.jsonl")

    assert client.calls == ["files.create"]
    assert client.wait_calls == []
