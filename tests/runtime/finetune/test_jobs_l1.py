"""L1: `submit_job` / `get_job` / `wait_job`（`runtime/finetune/jobs.py`）の純ロジックを検証する。

SDK 接触（`_adapters.finetune.upload_file` / `create_job`）を fake へ差し替え、`jobs.py` が
組み上げた body dict を測定点として、NFR-7 のキー集合完全一致・FR-5 の method エイリアス写像 /
hyperparameters 配置 / train・val の受理形（型分岐）/ トップレベル設定と `trainingType` 写像 /
`extra_body` 合成とキー衝突検出（判断 5）/ FR-10 の必須設定不在エラーと戻り値 `JobRef` を
網羅する。`get_job` / `wait_job` は `retrieve_job` と時計（`time.monotonic` / `asyncio.sleep`）を
差し替え、FR-6 の状態写像・FR-7 と ADR 0031 の待機契約（timeout 必須・TIMEOUT 送出・未知状態での
待機継続・ポーリングループの隔離）を実時間を待たずに検証する。openai には触れず、ネットワーク
通信も行わない（`@pytest.mark.unit`）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from oai_agentspec.runtime.finetune import (
    DatasetBuildResult,
    FineTuneError,
    FineTuneFailureKind,
)
from oai_agentspec.runtime.finetune.jobs import get_job, submit_job, wait_job
from oai_agentspec.runtime.finetune.types import JobRef, JobResult, JobStatus

pytestmark = pytest.mark.unit

_CLIENT = object()
_MODEL = "gpt-4.1-mini-2025-04-14"


# ----------------------------------------------------------------------
# fake adapter（SDK 接触の差し替え）
# ----------------------------------------------------------------------


class _FakeAdapter:
    """`_adapters.finetune` の upload / wait / create の 3 窓口を捕捉する fake。

    `upload_file` は filename から決まる file id（`train.jsonl` -> `file-train`）を返し、
    `wait_file_processed` は呼び出しを記録して受け取った file id をそのまま返し、
    `create_job` は受け取った body を保持して固定 job id を返す。`calls` に SDK 窓口の
    呼び出し順（`"upload_file"` / `"wait_file_processed"` / `"create_job"`）を記録する。
    """

    def __init__(self, job_id: str = "ftjob-1") -> None:
        self.job_id = job_id
        self.upload_calls: list[dict[str, Any]] = []
        self.wait_calls: list[dict[str, Any]] = []
        self.calls: list[str] = []
        self.body: dict[str, Any] | None = None

    async def upload_file(self, client: Any, *, data: bytes, filename: str) -> str:
        self.calls.append("upload_file")
        self.upload_calls.append({"client": client, "data": data, "filename": filename})
        return "file-" + filename.removesuffix(".jsonl")

    async def wait_file_processed(self, client: Any, file_id: str, *, timeout: float) -> str:
        self.calls.append("wait_file_processed")
        self.wait_calls.append({"client": client, "file_id": file_id, "timeout": timeout})
        return file_id

    async def create_job(self, client: Any, body: dict[str, Any]) -> str:
        self.calls.append("create_job")
        self.body = body
        return self.job_id

    @property
    def sent_body(self) -> dict[str, Any]:
        """`create_job` へ渡された body（未呼び出しなら失敗させる）。"""
        assert self.body is not None, "create_job が呼ばれていない"
        return self.body


@pytest.fixture
def adapter(monkeypatch: pytest.MonkeyPatch) -> _FakeAdapter:
    """`_adapters.finetune` の 3 関数を fake へ差し替える（関数内遅延 import 前提）。"""
    fake = _FakeAdapter()
    monkeypatch.setattr("oai_agentspec._adapters.finetune.upload_file", fake.upload_file)
    monkeypatch.setattr(
        "oai_agentspec._adapters.finetune.wait_file_processed",
        fake.wait_file_processed,
        raising=False,
    )
    monkeypatch.setattr("oai_agentspec._adapters.finetune.create_job", fake.create_job)
    return fake


async def _submit_minimal(**overrides: Any) -> Any:
    """最小引数（train はファイル id）で submit_job を呼ぶ。"""
    kwargs: dict[str, Any] = {"train": "file-abc123", "model": _MODEL, "method": "sft"}
    kwargs.update(overrides)
    return await submit_job(_CLIENT, **kwargs)


# ----------------------------------------------------------------------
# NFR-7: 最小引数での body キー集合の完全一致
# ----------------------------------------------------------------------


async def test_minimal_body_key_set_is_exactly_three_keys(adapter: _FakeAdapter) -> None:
    """最小引数の body キー集合は {model, training_file, method} と完全一致する（NFR-7）。

    利用者が指定していないフィールド（suffix / seed / metadata / integrations /
    trainingType / 自動デプロイ系）を lib が既定で付加しないことを、`==` の完全一致で
    固定する（過大側・過小側の双方で RED になる）。
    """
    await _submit_minimal()
    assert set(adapter.sent_body) == {"model", "training_file", "method"}


async def test_minimal_body_values(adapter: _FakeAdapter) -> None:
    """最小引数の body は model / training_file / method の値をそのまま持つ。"""
    await _submit_minimal()
    assert adapter.sent_body == {
        "model": _MODEL,
        "training_file": "file-abc123",
        "method": {"type": "supervised"},
    }


@pytest.mark.parametrize(
    ("kwargs", "added"),
    [
        ({"val": "file-def456"}, {"validation_file"}),
        ({"suffix": "support-bot"}, {"suffix"}),
        ({"seed": 42}, {"seed"}),
        ({"metadata": {"owner": "acme"}}, {"metadata"}),
        ({"integrations": [{"type": "wandb"}]}, {"integrations"}),
        ({"training_type": "GlobalStandard"}, {"trainingType"}),
        ({"extra_body": {"deploymentType": "x", "other": 1}}, {"deploymentType", "other"}),
        ({"hyperparameters": {"n_epochs": 3}}, set()),
    ],
)
async def test_optional_argument_adds_only_its_own_key(
    adapter: _FakeAdapter, kwargs: dict[str, Any], added: set[str]
) -> None:
    """任意引数を 1 つ指定すると、その引数が担当するキーだけが body へ増える。

    `hyperparameters` は method 内（別階層）へ入るため body 直下のキーは増えない。
    """
    await _submit_minimal(**kwargs)
    assert set(adapter.sent_body) == {"model", "training_file", "method"} | added


# ----------------------------------------------------------------------
# FR-5: method のエイリアス写像と hyperparameters の配置
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "expected_type"),
    [("sft", "supervised"), ("dpo", "dpo"), ("reinforcement", "reinforcement")],
)
async def test_method_alias_mapping(adapter: _FakeAdapter, method: str, expected_type: str) -> None:
    """sft / dpo はエイリアス写像し、未知の値は写像せず非解釈で type に載せる。"""
    await _submit_minimal(method=method)
    assert adapter.sent_body["method"] == {"type": expected_type}


async def test_method_mapping_is_passed_through_verbatim(adapter: _FakeAdapter) -> None:
    """method が Mapping のときは丸ごと非解釈で透過する（lib が構造を解釈しない）。"""
    method = {
        "type": "reinforcement",
        "reinforcement": {"hyperparameters": {"eval_interval": "auto"}},
    }
    await _submit_minimal(method=method)
    assert adapter.sent_body["method"] == method


async def test_hyperparameters_nested_under_mapped_type_for_sft(adapter: _FakeAdapter) -> None:
    """hyperparameters はエイリアス写像後の type 値の下へ入る（"sft" キーの下ではない）。"""
    await _submit_minimal(method="sft", hyperparameters={"n_epochs": 3, "batch_size": "auto"})
    assert adapter.sent_body["method"] == {
        "type": "supervised",
        "supervised": {"hyperparameters": {"n_epochs": 3, "batch_size": "auto"}},
    }
    assert "sft" not in adapter.sent_body["method"]


async def test_hyperparameters_nested_under_dpo(adapter: _FakeAdapter) -> None:
    """dpo の hyperparameters は method.dpo 下へ入る（beta 等も非解釈で透過）。"""
    await _submit_minimal(method="dpo", hyperparameters={"beta": 0.1})
    assert adapter.sent_body["method"] == {
        "type": "dpo",
        "dpo": {"hyperparameters": {"beta": 0.1}},
    }


async def test_hyperparameters_are_not_placed_at_top_level(adapter: _FakeAdapter) -> None:
    """hyperparameters は body 直下へ載せない（配置ミスの変異で RED）。"""
    await _submit_minimal(hyperparameters={"n_epochs": 3})
    assert "hyperparameters" not in adapter.sent_body


async def test_hyperparameters_with_mapping_method_raises_config_missing(
    adapter: _FakeAdapter,
) -> None:
    """method が Mapping のとき hyperparameters 併用は CONFIG_MISSING（配置先が一意でない）。"""
    with pytest.raises(FineTuneError) as exc_info:
        await _submit_minimal(method={"type": "supervised"}, hyperparameters={"n_epochs": 3})
    assert exc_info.value.kind == FineTuneFailureKind.CONFIG_MISSING
    assert "hyperparameters" in exc_info.value.message
    assert adapter.body is None


# ----------------------------------------------------------------------
# FR-5: train / val の受理形（型分岐）
# ----------------------------------------------------------------------


async def test_str_train_is_used_as_file_id_without_upload(adapter: _FakeAdapter) -> None:
    """train が str のときはファイル id とみなし、アップロードを一切呼ばない。"""
    await _submit_minimal(train="file-abc123")
    assert adapter.upload_calls == []
    assert adapter.sent_body["training_file"] == "file-abc123"


async def test_data_path_waits_for_file_processing_after_upload(
    adapter: _FakeAdapter, tmp_path: Path
) -> None:
    """データ経路は upload -> wait_file_processed の順で、file id と timeout を渡す（ADR 0032）。

    実 API はアップロード直後の `status="pending"` なファイルでのジョブ作成を 400 で拒む。
    jobs 層が待機の制御を出していること（adapter へ内包しないこと）を L1 で固定する。
    """
    source = tmp_path / "val.jsonl"
    source.write_text('{"messages": []}\n', encoding="utf-8")

    await _submit_minimal(train=[{"messages": []}], val=source, file_wait_timeout=300.0)

    assert adapter.calls == [
        "upload_file",
        "wait_file_processed",
        "upload_file",
        "wait_file_processed",
        "create_job",
    ]
    assert [call["file_id"] for call in adapter.wait_calls] == ["file-train", "file-validation"]
    assert [call["timeout"] for call in adapter.wait_calls] == [300.0, 300.0]


async def test_str_file_id_path_calls_neither_upload_nor_wait(adapter: _FakeAdapter) -> None:
    """`str`（アップロード済み id）経路は upload も待機も呼ばない（ADR 0032・利用者責任）。"""
    await _submit_minimal(train="file-abc123", val="file-def456")

    assert adapter.calls == ["create_job"]
    assert adapter.wait_calls == []


async def test_path_train_is_uploaded_with_train_filename(
    adapter: _FakeAdapter, tmp_path: Path
) -> None:
    """train が Path のときはローカル JSONL を読んで train.jsonl としてアップロードする。"""
    source = tmp_path / "my_dataset.jsonl"
    source.write_text('{"messages": []}\n', encoding="utf-8")

    await _submit_minimal(train=source)

    assert len(adapter.upload_calls) == 1
    call = adapter.upload_calls[0]
    assert call["filename"] == "train.jsonl"
    assert call["data"] == source.read_bytes()
    assert call["client"] is _CLIENT
    assert adapter.sent_body["training_file"] == "file-train"


async def test_build_result_train_is_serialized_to_jsonl_bytes(adapter: _FakeAdapter) -> None:
    """DatasetBuildResult は 1 行 1 JSON + 末尾改行の JSONL bytes へ直列化してアップロードする。"""
    records = (
        {"messages": [{"role": "user", "content": "a"}]},
        {"messages": [{"role": "user", "content": "b"}]},
    )
    await _submit_minimal(train=DatasetBuildResult(records=records, skipped=0))

    data = adapter.upload_calls[0]["data"]
    assert isinstance(data, bytes)
    lines = data.decode("utf-8").splitlines()
    assert [json.loads(line) for line in lines] == list(records)
    assert data.endswith(b"\n")


async def test_record_sequence_train_is_serialized(adapter: _FakeAdapter) -> None:
    """レコード列（Sequence[Mapping]）も JSONL bytes 化してアップロードする。"""
    records = [{"messages": [{"role": "user", "content": "a"}]}]
    await _submit_minimal(train=records)

    data = adapter.upload_calls[0]["data"]
    assert [json.loads(line) for line in data.decode("utf-8").splitlines()] == records


async def test_serialized_jsonl_keeps_non_ascii_unescaped(adapter: _FakeAdapter) -> None:
    """JSONL 直列化は utf-8・ensure_ascii=False（DatasetBuildResult.save と同一契約）。"""
    await _submit_minimal(train=[{"messages": [{"role": "user", "content": "返品ポリシー"}]}])

    text = adapter.upload_calls[0]["data"].decode("utf-8")
    assert "返品ポリシー" in text
    assert "\\u" not in text


async def test_val_is_uploaded_with_validation_filename(
    adapter: _FakeAdapter, tmp_path: Path
) -> None:
    """val のアップロードは validation.jsonl の filename で行う（train と取り違えない）。"""
    source = tmp_path / "val.jsonl"
    source.write_text('{"messages": []}\n', encoding="utf-8")

    await _submit_minimal(val=source)

    assert [call["filename"] for call in adapter.upload_calls] == ["validation.jsonl"]
    assert adapter.sent_body["validation_file"] == "file-validation"


async def test_train_and_val_are_judged_independently(
    adapter: _FakeAdapter, tmp_path: Path
) -> None:
    """train がファイル id・val がローカルデータという混在指定を受理する（独立判定）。"""
    source = tmp_path / "val.jsonl"
    source.write_text('{"messages": []}\n', encoding="utf-8")

    await _submit_minimal(train="file-abc123", val=source)

    assert [call["filename"] for call in adapter.upload_calls] == ["validation.jsonl"]
    assert adapter.sent_body["training_file"] == "file-abc123"
    assert adapter.sent_body["validation_file"] == "file-validation"


async def test_str_val_is_used_as_file_id_without_upload(adapter: _FakeAdapter) -> None:
    """val が str のときもファイル id とみなし再アップロードしない。"""
    await _submit_minimal(val="file-def456")
    assert adapter.upload_calls == []
    assert adapter.sent_body["validation_file"] == "file-def456"


async def test_validation_file_key_absent_when_val_omitted(adapter: _FakeAdapter) -> None:
    """val 省略時は validation_file キー自体を送らない（None を明示送信しない）。"""
    await _submit_minimal()
    assert "validation_file" not in adapter.sent_body


# ----------------------------------------------------------------------
# FR-5: トップレベル設定と training_type の wire key 写像
# ----------------------------------------------------------------------


async def test_top_level_settings_are_placed_at_body_root(adapter: _FakeAdapter) -> None:
    """suffix / seed / metadata / integrations は body 直下の同名キーへ載る。"""
    await _submit_minimal(
        suffix="support-bot",
        seed=42,
        metadata={"owner": "acme"},
        integrations=[{"type": "wandb"}],
    )
    body = adapter.sent_body
    assert body["suffix"] == "support-bot"
    assert body["seed"] == 42
    assert body["metadata"] == {"owner": "acme"}
    assert body["integrations"] == [{"type": "wandb"}]


async def test_seed_is_top_level_not_inside_method(adapter: _FakeAdapter) -> None:
    """seed は method 内の hyperparameters ではなく body トップレベルへ載せる。"""
    await _submit_minimal(seed=42, hyperparameters={"n_epochs": 3})
    assert adapter.sent_body["seed"] == 42
    assert "seed" not in adapter.sent_body["method"]["supervised"]["hyperparameters"]


async def test_training_type_is_mapped_to_wire_key(adapter: _FakeAdapter) -> None:
    """training_type は body 直下の "trainingType"（wire key）へ写像し、値は検証しない。"""
    await _submit_minimal(training_type="GlobalStandard")
    assert adapter.sent_body["trainingType"] == "GlobalStandard"
    assert "training_type" not in adapter.sent_body


@pytest.mark.parametrize(
    "absent_key", ["suffix", "seed", "metadata", "integrations", "trainingType"]
)
async def test_omitted_settings_are_not_sent(adapter: _FakeAdapter, absent_key: str) -> None:
    """省略された設定はキー自体を送らない（lib は既定値を発明しない）。"""
    await _submit_minimal()
    assert absent_key not in adapter.sent_body


# ----------------------------------------------------------------------
# FR-5 / 判断 5: extra_body の合成とキー衝突検出
# ----------------------------------------------------------------------


async def test_extra_body_is_merged_into_body_root(adapter: _FakeAdapter) -> None:
    """extra_body は非解釈で body 直下へ合成する（内容を解釈・正規化しない）。"""
    await _submit_minimal(extra_body={"deploymentType": "Standard", "nested": {"a": 1}})
    assert adapter.sent_body["deploymentType"] == "Standard"
    assert adapter.sent_body["nested"] == {"a": 1}


@pytest.mark.parametrize("key", ["model", "training_file", "method"])
async def test_extra_body_collision_with_always_occupied_keys(
    adapter: _FakeAdapter, key: str
) -> None:
    """常時占有キー（model / training_file / method）との衝突は CONFIG_MISSING。"""
    with pytest.raises(FineTuneError) as exc_info:
        await _submit_minimal(extra_body={key: "x"})
    assert exc_info.value.kind == FineTuneFailureKind.CONFIG_MISSING
    assert key in exc_info.value.message
    assert adapter.body is None


@pytest.mark.parametrize(
    ("kwargs", "key"),
    [
        ({"suffix": "s"}, "suffix"),
        ({"seed": 1}, "seed"),
        ({"metadata": {"a": 1}}, "metadata"),
        ({"integrations": [{"type": "wandb"}]}, "integrations"),
        ({"training_type": "Standard"}, "trainingType"),
        ({"val": "file-def456"}, "validation_file"),
    ],
)
async def test_extra_body_collision_with_specified_optional_keys(
    adapter: _FakeAdapter, kwargs: dict[str, Any], key: str
) -> None:
    """実際に指定した任意引数の担当キーと extra_body が交差したら CONFIG_MISSING。"""
    with pytest.raises(FineTuneError) as exc_info:
        await _submit_minimal(extra_body={key: "x"}, **kwargs)
    assert exc_info.value.kind == FineTuneFailureKind.CONFIG_MISSING
    assert key in exc_info.value.message


async def test_collision_message_lists_all_conflicting_keys(adapter: _FakeAdapter) -> None:
    """衝突が複数あるときはキー名を全件メッセージへ列挙する（1 件だけ報告しない）。"""
    with pytest.raises(FineTuneError) as exc_info:
        await _submit_minimal(suffix="s", extra_body={"model": "x", "suffix": "y"})
    message = exc_info.value.message
    assert "model" in message
    assert "suffix" in message


@pytest.mark.parametrize("key", ["suffix", "seed", "metadata", "integrations", "trainingType"])
async def test_omitted_optional_keys_are_not_occupied(adapter: _FakeAdapter, key: str) -> None:
    """省略した任意引数の担当キーは占有されず、extra_body から指定できる（衝突にしない）。"""
    await _submit_minimal(extra_body={key: "from-extra"})
    assert adapter.sent_body[key] == "from-extra"


async def test_validation_file_is_not_occupied_when_val_omitted(adapter: _FakeAdapter) -> None:
    """val 省略時は validation_file が占有されず extra_body から指定できる。"""
    await _submit_minimal(extra_body={"validation_file": "file-def456"})
    assert adapter.sent_body["validation_file"] == "file-def456"


async def test_method_inner_keys_do_not_collide_with_body_root(adapter: _FakeAdapter) -> None:
    """method 内は別階層のため body 直下の同名キー指定と衝突しない。"""
    await _submit_minimal(
        method={"type": "supervised", "supervised": {"hyperparameters": {"n_epochs": 3}}},
        extra_body={"supervised": "body-root-value"},
    )
    assert adapter.sent_body["supervised"] == "body-root-value"
    assert adapter.sent_body["method"]["supervised"] == {"hyperparameters": {"n_epochs": 3}}


# ----------------------------------------------------------------------
# FR-10: 必須設定の不在（CONFIG_MISSING）
# ----------------------------------------------------------------------


@pytest.mark.parametrize("value", [None, "", "   "])
async def test_missing_model_raises_config_missing(adapter: _FakeAdapter, value: Any) -> None:
    """model が None / 空文字 / 空白のみなら CONFIG_MISSING（引数名をメッセージに含む）。"""
    with pytest.raises(FineTuneError) as exc_info:
        await _submit_minimal(model=value)
    assert exc_info.value.kind == FineTuneFailureKind.CONFIG_MISSING
    assert "model" in exc_info.value.message
    assert adapter.body is None


@pytest.mark.parametrize("value", [None, "", "   ", {}])
async def test_missing_method_raises_config_missing(adapter: _FakeAdapter, value: Any) -> None:
    """method が None / 空文字 / 空 Mapping なら CONFIG_MISSING。"""
    with pytest.raises(FineTuneError) as exc_info:
        await _submit_minimal(method=value)
    assert exc_info.value.kind == FineTuneFailureKind.CONFIG_MISSING
    assert "method" in exc_info.value.message


@pytest.mark.parametrize("value", [None, "", "   ", [], ()])
async def test_missing_train_raises_config_missing(adapter: _FakeAdapter, value: Any) -> None:
    """train が None / 空文字 / 空レコード列なら CONFIG_MISSING（空データを送らない）。"""
    with pytest.raises(FineTuneError) as exc_info:
        await _submit_minimal(train=value)
    assert exc_info.value.kind == FineTuneFailureKind.CONFIG_MISSING
    assert "train" in exc_info.value.message
    assert adapter.upload_calls == []
    assert adapter.body is None


async def test_empty_build_result_train_raises_config_missing(adapter: _FakeAdapter) -> None:
    """レコード 0 件の DatasetBuildResult も CONFIG_MISSING（空 JSONL を送らない）。"""
    with pytest.raises(FineTuneError) as exc_info:
        await _submit_minimal(train=DatasetBuildResult(records=(), skipped=0))
    assert exc_info.value.kind == FineTuneFailureKind.CONFIG_MISSING


# ----------------------------------------------------------------------
# 戻り値（JobRef）
# ----------------------------------------------------------------------


async def test_returns_job_ref_with_file_ids(adapter: _FakeAdapter, tmp_path: Path) -> None:
    """戻り値は JobRef で job_id と学習 / 検証ファイル id を保持する。"""
    source = tmp_path / "val.jsonl"
    source.write_text('{"messages": []}\n', encoding="utf-8")

    ref = await _submit_minimal(train="file-abc123", val=source)

    assert isinstance(ref, JobRef)
    assert ref == JobRef(
        job_id="ftjob-1",
        training_file_id="file-abc123",
        validation_file_id="file-validation",
    )


async def test_returns_job_ref_with_none_validation_file_id(adapter: _FakeAdapter) -> None:
    """val 省略時の JobRef は validation_file_id が None。"""
    ref = await _submit_minimal()
    assert ref.validation_file_id is None
    assert ref.training_file_id == "file-abc123"
    assert ref.job_id == "ftjob-1"


# ----------------------------------------------------------------------
# get_job / wait_job 用の fake（retrieve_job の差し替え・fake clock）
# ----------------------------------------------------------------------


def _payload(
    status: str,
    *,
    job_id: str = "ftjob-1",
    model: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """`_adapters.finetune.retrieve_job` が返す plain payload を組み立てる。"""
    return {"id": job_id, "status": status, "fine_tuned_model": model, "error_message": error}


class _FakeRetriever:
    """`retrieve_job` を差し替え、payload を順番に返す fake（最後の payload を反復）。"""

    def __init__(self, payloads: list[dict[str, Any]], error: BaseException | None = None) -> None:
        self.payloads = payloads
        self.error = error
        self.calls: list[tuple[Any, str]] = []

    async def retrieve_job(self, client: Any, job_id: str) -> dict[str, Any]:
        self.calls.append((client, job_id))
        if self.error is not None:
            raise self.error
        index = min(len(self.calls) - 1, len(self.payloads) - 1)
        return self.payloads[index]

    @property
    def count(self) -> int:
        """照会回数。"""
        return len(self.calls)


class _FakeClock:
    """`time.monotonic` / `asyncio.sleep` を差し替える擬似時計（実時間を待たない）。

    `monotonic()` は現在時刻を返すだけで進めず、`sleep(seconds)` が待機秒数を記録して
    その分だけ時計を進める（ポーリングループの経過時間を決定的に再現する）。
    """

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def _install(
    monkeypatch: pytest.MonkeyPatch,
    payloads: list[dict[str, Any]],
    *,
    error: BaseException | None = None,
) -> tuple[_FakeRetriever, _FakeClock]:
    """retrieve_job と時計（`time.monotonic` / `asyncio.sleep`）を差し替える。"""
    import asyncio
    import time

    retriever = _FakeRetriever(payloads, error=error)
    clock = _FakeClock()
    monkeypatch.setattr("oai_agentspec._adapters.finetune.retrieve_job", retriever.retrieve_job)
    monkeypatch.setattr(time, "monotonic", clock.monotonic)
    monkeypatch.setattr(asyncio, "sleep", clock.sleep)
    return retriever, clock


# ----------------------------------------------------------------------
# get_job（FR-6）
# ----------------------------------------------------------------------


async def test_get_job_queries_adapter_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_job は単発照会（retrieve_job を 1 回だけ・client と job_id を渡す）。"""
    retriever, _clock = _install(monkeypatch, [_payload("running")])

    await get_job(_CLIENT, "ftjob-1")

    assert retriever.count == 1
    assert retriever.calls[0] == (_CLIENT, "ftjob-1")


async def test_get_job_maps_succeeded_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """succeeded の payload は JobResult（status / raw_status / model_ref）へ写像される。"""
    _install(
        monkeypatch,
        [_payload("succeeded", model="ft:gpt-4.1-mini:acme::abc123")],
    )

    result = await get_job(_CLIENT, "ftjob-1")

    assert result == JobResult(
        job_id="ftjob-1",
        status=JobStatus.SUCCEEDED,
        raw_status="succeeded",
        model_ref="ft:gpt-4.1-mini:acme::abc123",
        error_message=None,
    )
    assert result.is_terminal is True


async def test_get_job_maps_failed_payload_with_error_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """failed の payload は例外にせず error_message を保持した JobResult を返す。"""
    _install(monkeypatch, [_payload("failed", error="training file is invalid")])

    result = await get_job(_CLIENT, "ftjob-1")

    assert result.status == JobStatus.FAILED
    assert result.error_message == "training file is invalid"
    assert result.model_ref is None


async def test_get_job_maps_unknown_status_to_running_keeping_raw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """未知の状態値は例外にせず RUNNING へ倒し、raw_status に生文字列を保全する（FR-6）。"""
    _install(monkeypatch, [_payload("validating_files")])

    result = await get_job(_CLIENT, "ftjob-1")

    assert result.status == JobStatus.RUNNING
    assert result.raw_status == "validating_files"
    assert result.is_terminal is False


async def test_get_job_uses_payload_job_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """JobResult の job_id は payload の id を用いる。"""
    _install(monkeypatch, [_payload("queued", job_id="ftjob-zzz")])

    result = await get_job(_CLIENT, "ftjob-zzz")

    assert result.job_id == "ftjob-zzz"
    assert result.status == JobStatus.QUEUED


async def test_get_job_propagates_adapter_api_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """adapter の FineTuneError(API_ERROR) はそのまま伝播する（握り潰さない）。"""
    original = FineTuneError(FineTuneFailureKind.API_ERROR, "job not found: ftjob-missing")
    _install(monkeypatch, [], error=original)

    with pytest.raises(FineTuneError) as exc_info:
        await get_job(_CLIENT, "ftjob-missing")
    assert exc_info.value is original
    assert exc_info.value.kind == FineTuneFailureKind.API_ERROR


# ----------------------------------------------------------------------
# wait_job（FR-7・ADR 0031）
# ----------------------------------------------------------------------


async def test_wait_job_requires_timeout_keyword_adr_0031(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ADR 0031 Confirmation 1: timeout は既定値のない必須 keyword 引数である。

    省略した呼び出しが TypeError となることで、無限待機の経路が構造的に存在しないことを
    シグネチャレベルで固定する。
    """
    _install(monkeypatch, [_payload("succeeded")])

    with pytest.raises(TypeError):
        await wait_job(_CLIENT, "ftjob-1")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        await wait_job(_CLIENT, "ftjob-1", 10.0)  # type: ignore[misc]


def test_wait_job_poll_interval_default_is_30_seconds() -> None:
    """poll_interval の既定値は 30.0 秒（設計判断 7 の確定値・ADR 0031）。"""
    import inspect

    parameters = inspect.signature(wait_job).parameters
    assert parameters["poll_interval"].default == 30.0
    assert parameters["timeout"].default is inspect.Parameter.empty
    assert parameters["timeout"].kind is inspect.Parameter.KEYWORD_ONLY


async def test_wait_job_raises_timeout_error_adr_0031(monkeypatch: pytest.MonkeyPatch) -> None:
    """ADR 0031 Confirmation 2: 終端に達しないまま deadline を超えたら TIMEOUT を送出する。

    メッセージには job_id・timeout 値・最後に観測した raw_status を含め、ジョブを取り消して
    いない（同じ job_id で再実行できる）旨を明記する。
    """
    retriever, clock = _install(monkeypatch, [_payload("running", job_id="ftjob-42")])

    with pytest.raises(FineTuneError) as exc_info:
        await wait_job(_CLIENT, "ftjob-42", timeout=10.0, poll_interval=3.0)

    message = exc_info.value.message
    assert exc_info.value.kind == FineTuneFailureKind.TIMEOUT
    assert "ftjob-42" in message
    assert "10" in message
    assert "running" in message
    assert "取り消" in message
    assert "再" in message
    assert retriever.count >= 2


async def test_wait_job_does_not_sleep_past_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    """sleep は min(poll_interval, 残時間) で deadline を超過させない（ADR 0031）。"""
    _retriever, clock = _install(monkeypatch, [_payload("running")])

    with pytest.raises(FineTuneError):
        await wait_job(_CLIENT, "ftjob-1", timeout=10.0, poll_interval=3.0)

    assert clock.sleeps == [3.0, 3.0, 3.0, 1.0]
    assert sum(clock.sleeps) == 10.0


async def test_wait_job_continues_on_unknown_status_adr_0031(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ADR 0031 Confirmation 3: 未知の状態文字列は非終端として待機を継続する。

    validating_files -> pausing -> succeeded と遷移しても例外にならず、終端で結果を返す。
    """
    retriever, clock = _install(
        monkeypatch,
        [
            _payload("validating_files"),
            _payload("pausing"),
            _payload("succeeded", model="ft:model:abc"),
        ],
    )

    result = await wait_job(_CLIENT, "ftjob-1", timeout=100.0, poll_interval=5.0)

    assert result.status == JobStatus.SUCCEEDED
    assert result.model_ref == "ft:model:abc"
    assert retriever.count == 3
    assert clock.sleeps == [5.0, 5.0]


async def test_wait_job_queries_immediately_before_sleeping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """初回照会は sleep 前に即時実行する（既に終端なら 1 回の照会だけで返る）。"""
    retriever, clock = _install(monkeypatch, [_payload("succeeded", model="ft:model:abc")])

    result = await wait_job(_CLIENT, "ftjob-1", timeout=100.0)

    assert retriever.count == 1
    assert clock.sleeps == []
    assert result.is_terminal is True


@pytest.mark.parametrize(
    ("raw", "status"),
    [
        ("succeeded", JobStatus.SUCCEEDED),
        ("failed", JobStatus.FAILED),
        ("cancelled", JobStatus.CANCELLED),
    ],
)
async def test_wait_job_returns_on_each_terminal_status(
    monkeypatch: pytest.MonkeyPatch, raw: str, status: JobStatus
) -> None:
    """終端 3 種すべてで JobResult を返す（failed / cancelled も例外にしない）。"""
    _install(monkeypatch, [_payload("running"), _payload(raw, error="reason text")])

    result = await wait_job(_CLIENT, "ftjob-1", timeout=100.0, poll_interval=5.0)

    assert result.status == status
    assert result.raw_status == raw


async def test_wait_job_failed_result_carries_error_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """failed で返る JobResult は失敗理由を error_message に保持する。"""
    _install(monkeypatch, [_payload("failed", error="invalid training file")])

    result = await wait_job(_CLIENT, "ftjob-1", timeout=100.0)

    assert result.status == JobStatus.FAILED
    assert result.error_message == "invalid training file"


@pytest.mark.parametrize("timeout", [0.0, -1.0])
async def test_wait_job_rejects_non_positive_timeout(
    monkeypatch: pytest.MonkeyPatch, timeout: float
) -> None:
    """timeout の非正値は CONFIG_MISSING（照会を 1 回も行わない）。"""
    retriever, _clock = _install(monkeypatch, [_payload("succeeded")])

    with pytest.raises(FineTuneError) as exc_info:
        await wait_job(_CLIENT, "ftjob-1", timeout=timeout)

    assert exc_info.value.kind == FineTuneFailureKind.CONFIG_MISSING
    assert "timeout" in exc_info.value.message
    assert retriever.count == 0


@pytest.mark.parametrize("poll_interval", [0.0, -5.0])
async def test_wait_job_rejects_non_positive_poll_interval(
    monkeypatch: pytest.MonkeyPatch, poll_interval: float
) -> None:
    """poll_interval の非正値は CONFIG_MISSING（照会を 1 回も行わない）。"""
    retriever, _clock = _install(monkeypatch, [_payload("succeeded")])

    with pytest.raises(FineTuneError) as exc_info:
        await wait_job(_CLIENT, "ftjob-1", timeout=10.0, poll_interval=poll_interval)

    assert exc_info.value.kind == FineTuneFailureKind.CONFIG_MISSING
    assert "poll_interval" in exc_info.value.message
    assert retriever.count == 0


async def test_wait_job_propagates_adapter_api_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """照会中の FineTuneError(API_ERROR) は待機を打ち切ってそのまま伝播する。"""
    original = FineTuneError(FineTuneFailureKind.API_ERROR, "unauthorized")
    _install(monkeypatch, [], error=original)

    with pytest.raises(FineTuneError) as exc_info:
        await wait_job(_CLIENT, "ftjob-1", timeout=10.0)
    assert exc_info.value is original


# ----------------------------------------------------------------------
# ADR 0031 Confirmation 4: ポーリングループの隔離（静的検証）
# ----------------------------------------------------------------------


def _asyncio_sleep_sites() -> set[tuple[str, str]]:
    """`src/oai_agentspec/` 配下で `asyncio.sleep` を呼ぶ (相対パス, 関数名) を集める。

    ast で関数単位まで絞り込む（テストコードは走査対象に含めないため偽陽性が出ない）。
    ネストした関数は最も内側の関数名を採用する。

    Returns:
        (`src/oai_agentspec/` からの相対パス, 関数名) の集合。
    """
    import ast

    root = Path(__file__).resolve().parents[3] / "src" / "oai_agentspec"
    sites: set[tuple[str, str]] = set()

    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        stack: list[str] = []

        def visit(node: ast.AST, stack: list[str] = stack, path: Path = path) -> None:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                stack.append(node.name)
                for child in ast.iter_child_nodes(node):
                    visit(child)
                stack.pop()
                return
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "sleep"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "asyncio"
            ):
                sites.add((path.relative_to(root).as_posix(), stack[-1] if stack else "<module>"))
            for child in ast.iter_child_nodes(node):
                visit(child)

        visit(tree)

    return sites


def test_asyncio_sleep_exists_only_in_wait_job_adr_0031() -> None:
    """ADR 0031 Confirmation 4: `asyncio.sleep` は wait_job 実装にのみ存在する。

    lib 内唯一のポーリングループが `wait_job` へ隔離されていることの機械検証。走査対象は
    `src/oai_agentspec/` に限定し（テストコードの sleep で偽陽性にしない）、集合の `==`
    比較により新たな待機ループの混入（過大側）と wait_job からの消失（過小側）の双方を
    検知する。
    """
    assert _asyncio_sleep_sites() == {("runtime/finetune/jobs.py", "wait_job")}


# ----------------------------------------------------------------------
# client 不在の検証（FR-10・API 呼び出し前の fail-fast）
# ----------------------------------------------------------------------


async def test_submit_job_rejects_none_client(adapter: _FakeAdapter) -> None:
    """submit_job は client 不在を CONFIG_MISSING で弾く（API_ERROR にしない）。

    検証はアップロード・ジョブ作成より前に行い、従量課金操作を 1 度も発生させない。
    """
    with pytest.raises(FineTuneError) as exc_info:
        await submit_job(None, train="file-abc123", model=_MODEL, method="sft")

    assert exc_info.value.kind == FineTuneFailureKind.CONFIG_MISSING
    assert "client" in exc_info.value.message
    assert adapter.upload_calls == []
    assert adapter.body is None


async def test_submit_job_rejects_none_client_before_upload(
    adapter: _FakeAdapter, tmp_path: Path
) -> None:
    """アップロードを要する受理形でも client 不在の検証が先に走る（アップロード 0 回）。"""
    source = tmp_path / "train.jsonl"
    source.write_text('{"messages": []}\n', encoding="utf-8")

    with pytest.raises(FineTuneError) as exc_info:
        await submit_job(None, train=source, model=_MODEL, method="sft")

    assert exc_info.value.kind == FineTuneFailureKind.CONFIG_MISSING
    assert adapter.upload_calls == []


async def test_get_job_rejects_none_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_job は client 不在を CONFIG_MISSING で弾き、照会を行わない。"""
    retriever, _clock = _install(monkeypatch, [_payload("succeeded")])

    with pytest.raises(FineTuneError) as exc_info:
        await get_job(None, "ftjob-1")

    assert exc_info.value.kind == FineTuneFailureKind.CONFIG_MISSING
    assert "client" in exc_info.value.message
    assert retriever.count == 0


async def test_wait_job_rejects_none_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """wait_job は client 不在を CONFIG_MISSING で弾き、照会も待機も行わない。"""
    retriever, clock = _install(monkeypatch, [_payload("succeeded")])

    with pytest.raises(FineTuneError) as exc_info:
        await wait_job(None, "ftjob-1", timeout=10.0)

    assert exc_info.value.kind == FineTuneFailureKind.CONFIG_MISSING
    assert "client" in exc_info.value.message
    assert retriever.count == 0
    assert clock.sleeps == []


# ----------------------------------------------------------------------
# val の空判定（空データの課金アップロード防止）
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    ["", "   ", [], (), DatasetBuildResult(records=(), skipped=0)],
)
async def test_empty_val_raises_config_missing_without_upload(
    adapter: _FakeAdapter, value: Any
) -> None:
    """val が None 以外で空なら CONFIG_MISSING（空データをアップロードしない）。

    空の検証データを送るとアップロード課金だけが発生して意味を持たないため、
    「省略」と「空」を区別して fail-closed にする。
    """
    with pytest.raises(FineTuneError) as exc_info:
        await _submit_minimal(val=value)

    assert exc_info.value.kind == FineTuneFailureKind.CONFIG_MISSING
    assert "val" in exc_info.value.message
    assert adapter.upload_calls == []
    assert adapter.body is None


async def test_empty_val_is_rejected_even_when_train_needs_upload(
    adapter: _FakeAdapter, tmp_path: Path
) -> None:
    """train のアップロードが必要でも、空 val の検証が先に走り 1 件もアップロードしない。"""
    source = tmp_path / "train.jsonl"
    source.write_text('{"messages": []}\n', encoding="utf-8")

    with pytest.raises(FineTuneError) as exc_info:
        await _submit_minimal(train=source, val=[])

    assert exc_info.value.kind == FineTuneFailureKind.CONFIG_MISSING
    assert adapter.upload_calls == []
