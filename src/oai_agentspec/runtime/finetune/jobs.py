"""マネージド Fine-Tuning の学習ジョブ管理（純ロジック層・SDK 非接触）。

`submit_job` は利用者が指定した学習設定を検証し、必要ならデータを JSONL としてアップロード
してから、プラットフォームへ送るリクエスト body を組み立てて `_adapters.finetune` へ委譲する。
本モジュールは `openai` / `agents` を import せず（SDK 接触は adapter 経由のみ・NFR-1）、
環境変数も読まない（接続設定は利用者が構築した client で受領する・NFR-3）。

設計の核:
    - **受理形は型で分岐する**: `str` は**アップロード済みファイル id**、ローカル JSONL は
      `Path` で渡す（`validate_dataset` の `str` = パスとは意味が異なる）。`DatasetBuildResult`
      とレコード列は JSONL bytes へ直列化してアップロードする。
    - **lib は設定を発明しない**: 利用者が指定しなかったフィールドは body へ入れない
      （`None` を明示送信しない）。値の妥当性（suffix 長・model と method の組み合わせ等）は
      検証せず、プラットフォームのエラー文言をそのまま利用者へ渡す（FR-5 / FR-10）。
    - **`extra_body` は非解釈で合成する**: ただし lib が組み立てるキー（占有集合）と交差した
      場合は、どちらが勝つかを曖昧にせず `CONFIG_MISSING` で衝突キーを全件列挙して失敗する。
      占有集合は常時占有キーと「実際に指定された」任意引数の担当キーのみで、省略した引数の
      担当キーは占有しない。
    - **ジョブ完了の待機は `wait_job` にのみ存在する**: `wait_job` は build-don't-run の例外
      として lib が実装する唯一のポーリングループを持つ（ADR 0031）。`wait_job` は単発照会の
      反復のみで、ジョブの起動・取消・再試行・モデル切替を行わない。`submit_job` は
      データをアップロードした場合に限りファイル処理の完了を待つ（ジョブ作成の前提条件で
      あり、ジョブ完了の待機ではない・待機の実体は SDK ヘルパ・ADR 0032）。`get_job` は
      単発照会のまま暗黙の待機を持たない。
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from .types import (
    DatasetBuildResult,
    FineTuneError,
    FineTuneFailureKind,
    JobRef,
    JobResult,
    _map_status,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

# `method` の短縮名 -> プラットフォームの type 値。未知の値は写像せずそのまま載せる。
_METHOD_ALIASES: Final[dict[str, str]] = {"sft": "supervised", "dpo": "dpo"}

# `wait_job` の照会間隔の既定値（秒・ADR 0031）。
_DEFAULT_POLL_INTERVAL: Final[float] = 30.0

# `submit_job` がアップロードしたファイルの処理完了を待つ上限の既定値（秒・ADR 0032）。
_DEFAULT_FILE_WAIT_TIMEOUT: Final[float] = 300.0

# 引数の指定有無によらず lib が常に組み立てる body 直下のキー。
_ALWAYS_OCCUPIED: Final[frozenset[str]] = frozenset({"model", "training_file", "method"})

# 任意引数 -> body 直下の担当キー（wire key）。`training_type` のみ名前が異なる。
_OPTIONAL_WIRE_KEYS: Final[dict[str, str]] = {
    "val": "validation_file",
    "suffix": "suffix",
    "seed": "seed",
    "metadata": "metadata",
    "integrations": "integrations",
    "training_type": "trainingType",
}

_TRAIN_FILENAME: Final[str] = "train.jsonl"
_VALIDATION_FILENAME: Final[str] = "validation.jsonl"


def _config_error(message: str) -> FineTuneError:
    """`CONFIG_MISSING` の構造化エラーを組み立てる。

    Args:
        message: 人間可読のエラーメッセージ。

    Returns:
        `FineTuneError`（kind は `CONFIG_MISSING`）。
    """
    return FineTuneError(FineTuneFailureKind.CONFIG_MISSING, message)


def _extra_missing_error(exc: ImportError) -> FineTuneError:
    """adapter から伝播した `ImportError` を `EXTRA_MISSING` の構造化エラーへ変換する。

    SDK 窓口（`_adapters/finetune.py`）は導入ヒント付きの生 `ImportError` に留め、失敗種別への
    変換は本モジュール側で行う（FR-10: extra 不在で未捕捉例外にしない）。

    Args:
        exc: adapter が送出した `ImportError`（導入ヒントを含む）。

    Returns:
        `FineTuneError`（kind は `EXTRA_MISSING`・元メッセージを保全）。
    """
    return FineTuneError(
        FineTuneFailureKind.EXTRA_MISSING,
        "Fine-Tuning のジョブ管理に必要な依存が導入されていません"
        "（pip install 'oai-agentspec[finetune]'）"
        f": {exc}",
    )


def _is_empty(value: Any) -> bool:
    """設定値が「未指定と同等の空」かを判定する。

    Args:
        value: 判定対象（str / Mapping / Sequence / `DatasetBuildResult` 等）。

    Returns:
        None・空白のみの文字列・空のコレクション・レコード 0 件の `DatasetBuildResult` なら
        True。`Path` は中身を見ずに False（内容の妥当性はプラットフォームが判断する）。
    """
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, Path):
        return False
    if isinstance(value, DatasetBuildResult):
        return len(value.records) == 0
    try:
        return len(value) == 0
    except TypeError:
        return False


def _to_jsonl_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    """レコード列を JSONL（1 行 1 JSON + 末尾改行）の utf-8 bytes へ直列化する。

    非 ASCII はエスケープしない（`DatasetBuildResult.save` と同一契約）。

    Args:
        records: 直列化するレコード列。

    Returns:
        JSONL の utf-8 bytes。

    Raises:
        TypeError: レコードに JSON へ変換できない値が含まれる場合（`json.dumps` から伝播）。
    """
    body = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    return body.encode("utf-8")


async def _resolve_file_id(
    client: Any, source: Any, *, filename: str, file_wait_timeout: float
) -> str:
    """データの受理形に応じてファイル id を決める（必要な場合のみアップロードする）。

    アップロードした場合は、処理が完了（`processed`）するまで待ってから id を返す
    （未処理のファイル id でのジョブ作成はプラットフォームが 400 で拒む・ADR 0032）。
    `str`（アップロード済みファイル id）経路ではアップロードも待機も行わない。

    Args:
        client: 利用者が構築した OpenAI / Azure OpenAI の非同期クライアント。
        source: `str`（アップロード済みファイル id）/ `Path`（ローカル JSONL）/
            `DatasetBuildResult` / レコード列。
        filename: アップロード時に送るファイル名（keyword-only）。
        file_wait_timeout: ファイル処理完了を待つ上限秒数（keyword-only）。

    Returns:
        アップロード済みファイルの id。

    Raises:
        FineTuneError: アップロードに失敗した場合、またはファイル処理が `processed` 以外で
            終端した場合（`API_ERROR`）、処理完了の待機が上限を超過した場合（`TIMEOUT`）。
            いずれも adapter から伝播する。
        OSError: `Path` を読み取れない場合。
    """
    from ..._adapters import finetune as _ft

    if isinstance(source, str):
        return source
    if isinstance(source, Path):
        data = source.read_bytes()
    elif isinstance(source, DatasetBuildResult):
        data = _to_jsonl_bytes(source.records)
    else:
        data = _to_jsonl_bytes(list(source))
    file_id = await _ft.upload_file(client, data=data, filename=filename)
    return await _ft.wait_file_processed(client, file_id, timeout=file_wait_timeout)


def _build_method(
    method: str | Mapping[str, Any],
    hyperparameters: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """`method` 引数から body の `method` オブジェクトを組み立てる。

    `str` は `sft` -> `supervised` / `dpo` -> `dpo` のエイリアス写像のみ行い、未知の値は
    写像せずそのまま `type` へ載せる。`hyperparameters` は写像後の type 値の下へ入れる。
    Mapping は丸ごと非解釈で透過する。

    Args:
        method: 学習手法（短縮名・type 値・`method` オブジェクト全体のいずれか）。
        hyperparameters: ハイパーパラメータ（未指定なら None）。

    Returns:
        body の `method` オブジェクト。

    Raises:
        FineTuneError: `method` が Mapping かつ `hyperparameters` を併用した場合
            （配置先が一意に決まらない・`CONFIG_MISSING`）。
    """
    if isinstance(method, str):
        type_value = _METHOD_ALIASES.get(method, method)
        built: dict[str, Any] = {"type": type_value}
        if hyperparameters is not None:
            built[type_value] = {"hyperparameters": dict(hyperparameters)}
        return built
    if hyperparameters is not None:
        raise _config_error(
            "method に Mapping を指定した場合は hyperparameters を併用できません"
            "（配置先が一意に決まらないため、hyperparameters は method の中へ含めてください）"
        )
    return dict(method)


def _occupied_keys(specified: Mapping[str, Any]) -> set[str]:
    """lib が組み立てる body 直下のキー集合（占有集合）を求める。

    Args:
        specified: 任意引数名 -> 指定値の mapping（None は未指定とみなす）。

    Returns:
        常時占有キーと、実際に指定された任意引数の担当キー（wire key）の和集合。
    """
    occupied = set(_ALWAYS_OCCUPIED)
    for name, value in specified.items():
        if value is not None:
            occupied.add(_OPTIONAL_WIRE_KEYS[name])
    return occupied


def _check_extra_body_collision(extra_body: Mapping[str, Any] | None, occupied: set[str]) -> None:
    """`extra_body` と占有集合の交差を検出する（交差があれば失敗させる）。

    Args:
        extra_body: 利用者が渡す追加 body（未指定なら None）。
        occupied: lib が組み立てる body 直下のキー集合。

    Raises:
        FineTuneError: 交差が 1 件でもある場合（`CONFIG_MISSING`・衝突キーを全件列挙）。
    """
    if not extra_body:
        return
    conflicts = sorted(occupied & set(extra_body))
    if conflicts:
        raise _config_error(
            "extra_body のキーが専用引数の指定と衝突しています"
            f"（衝突したキー: {', '.join(conflicts)}）。"
            "どちらか一方で指定してください"
        )


async def submit_job(
    client: Any,
    *,
    train: Any,
    model: str,
    method: str | Mapping[str, Any],
    val: Any = None,
    hyperparameters: Mapping[str, Any] | None = None,
    training_type: str | None = None,
    suffix: str | None = None,
    seed: int | None = None,
    metadata: Mapping[str, Any] | None = None,
    integrations: Sequence[Mapping[str, Any]] | None = None,
    extra_body: Mapping[str, Any] | None = None,
    file_wait_timeout: float = _DEFAULT_FILE_WAIT_TIMEOUT,
) -> JobRef:
    """学習ジョブを投入する（必要ならデータをアップロードしてから作成する）。

    `train` / `val` の受理形は型で分岐する。**`str` はアップロード済みファイル id** を意味し、
    ローカル JSONL ファイルは `Path` で渡すこと（`validate_dataset` の `str` = ファイルパスとは
    意味が異なる）。`DatasetBuildResult` とレコード列は JSONL へ直列化してアップロードする。

    利用者が指定しなかった設定はリクエストへ入れない（lib は既定値を発明しない）。値の妥当性
    （suffix 長・model と method の組み合わせ等）は検証せず、プラットフォームのエラー文言を
    そのまま伝える。

    Args:
        client: 利用者が構築した OpenAI / Azure OpenAI の非同期クライアント
            （`AsyncOpenAI` / `AsyncAzureOpenAI` 等・必須）。
        train: 学習データ（ファイル id の `str` / `Path` / `DatasetBuildResult` / レコード列）。
            データ（`Path` / `DatasetBuildResult` / レコード列）を渡した場合はアップロード後に
            処理完了（`processed`）まで待ってからジョブを作る。**ファイル id（`str`）を渡す
            場合は待機しない**ため、`processed` になっていることを利用者側で確認してから
            渡すこと（ADR 0032）。`Path` はシンボリックリンク解決・拡張子検証をせずそのまま
            読み取り、内容を外部プラットフォームへ送信する。外部入力から組み立てたパスを
            渡さないこと。
        model: ベースモデル名（プラットフォームの表記をそのまま渡す）。
        method: 学習手法。`"sft"` -> `supervised` / `"dpo"` -> `dpo` の短縮名、任意の type 値、
            または `method` オブジェクト全体の Mapping（丸ごと非解釈で透過）。
        val: 検証データ（`train` と同じ受理形・独立に判定する。`Path` の扱いも `train` と同じ）。
            データを渡した場合は `train` と同様に処理完了まで待ち、ファイル id（`str`）を渡す
            場合は待機しない（`processed` まで待ってから渡すこと）。渡さない場合は指定を省略
            すること（空値を渡すと `CONFIG_MISSING`）。
        hyperparameters: ハイパーパラメータ。写像後の type 値の下へ配置する。`method` が
            Mapping の場合は併用できない（method の中へ含めること）。
        training_type: プラットフォーム固有の学習種別。body 直下の `trainingType` へ載せる。
        suffix: 学習済みモデル名のサフィックス（プラットフォームにより扱いが異なる）。
        seed: 再現性のためのシード（body 直下・`method` の中には入れない）。
        metadata: ジョブに付けるメタデータ。
        integrations: 外部連携設定（非解釈で透過）。
        extra_body: 追加のリクエスト body（非解釈で body 直下へ合成）。専用引数が組み立てる
            キーと交差した場合は `CONFIG_MISSING` で失敗する。
        file_wait_timeout: アップロードしたファイルの処理完了を待つ上限秒数（既定 300.0）。
            非正値は API を 1 度も呼ばずに `CONFIG_MISSING` で失敗する。待機の照会間隔は
            2 秒に固定されているため、**2 秒未満を指定しても 1 ポール分は必ず待つ**。
            大きなデータセットで既定値が不足する場合は、値を増やすか、自前で
            `client.files.create(...)` -> `client.files.wait_for_processing(...)` を行い、
            得たファイル id を `str` として `train=` / `val=` へ渡すこと（この経路では
            lib は待機しない）。

    Returns:
        `JobRef`（ジョブ id と学習 / 検証ファイル id）。`val` 省略時は
        `validation_file_id` が None。

    Raises:
        FineTuneError: 必須設定（`client` / `train` / `model` / `method`）が不在、`val` が空値、
            `method` Mapping と `hyperparameters` の併用、`extra_body` のキー衝突（いずれも
            `CONFIG_MISSING`）、またはアップロード / ジョブ作成の API 失敗（`API_ERROR`）。
            アップロードしたファイルの処理完了が上限（`file_wait_timeout`）を超過した場合は
            `TIMEOUT`。`finetune` extra（openai）が未導入の場合は `EXTRA_MISSING`
            （導入ヒント付き）。`API_ERROR` の `message` はプラットフォームのレスポンス本文を
            逐語で含む。送信した設定値（`integrations` の資格情報等）がエコーバックされうる
            ため、信頼できない相手へそのまま提示・記録しない。
        OSError: `Path` で渡したファイルを読み取れない場合。
        TypeError: レコードが JSON 直列化不能な値を含む場合（`json.dumps` から伝播）。

    Note:
        `create_job` 段階で `API_ERROR` が発生した場合、直前にアップロードしたファイルは
        プラットフォーム上に残る（`JobRef` を返さないため id は利用者へ渡らない）。ファイル
        処理の失敗（`API_ERROR`）・待機の上限超過（`TIMEOUT`）で失敗した場合も同様に、
        アップロード済みファイルは残る。削除は利用者責任（`client.files.delete` 等）で
        行うこと。

        データは全量をメモリ上で JSONL 化してから送信する。GB 級のデータセットは、事前に
        アップロード済みのファイル id（`str`）を渡す経路を使うこと。
    """
    from ..._adapters import finetune as _ft

    if client is None:
        raise _config_error(
            "client が指定されていません"
            "（AsyncOpenAI / AsyncAzureOpenAI 等の非同期クライアントが必須です）"
        )
    if _is_empty(model):
        raise _config_error("model が指定されていません（ベースモデル名は必須です）")
    if _is_empty(method):
        raise _config_error("method が指定されていません（学習手法は必須です）")
    if _is_empty(train):
        raise _config_error("train が指定されていません（学習データは必須です）")
    if val is not None and _is_empty(val):
        raise _config_error(
            "val が空です（空の検証データは送信しません）。"
            "検証データを渡さない場合は val を省略してください"
        )
    if file_wait_timeout <= 0:
        raise _config_error(
            f"file_wait_timeout には正の秒数を指定してください（受領値: {file_wait_timeout}）"
        )

    method_body = _build_method(method, hyperparameters)
    occupied = _occupied_keys(
        {
            "val": val,
            "suffix": suffix,
            "seed": seed,
            "metadata": metadata,
            "integrations": integrations,
            "training_type": training_type,
        }
    )
    _check_extra_body_collision(extra_body, occupied)

    try:
        training_file_id = await _resolve_file_id(
            client, train, filename=_TRAIN_FILENAME, file_wait_timeout=file_wait_timeout
        )
        validation_file_id = (
            None
            if val is None
            else await _resolve_file_id(
                client, val, filename=_VALIDATION_FILENAME, file_wait_timeout=file_wait_timeout
            )
        )
    except ImportError as exc:
        raise _extra_missing_error(exc) from exc

    body: dict[str, Any] = {
        "model": model,
        "training_file": training_file_id,
        "method": method_body,
    }
    if validation_file_id is not None:
        body["validation_file"] = validation_file_id
    if suffix is not None:
        body["suffix"] = suffix
    if seed is not None:
        body["seed"] = seed
    if metadata is not None:
        body["metadata"] = dict(metadata)
    if integrations is not None:
        body["integrations"] = list(integrations)
    if training_type is not None:
        body["trainingType"] = training_type
    if extra_body:
        body.update(extra_body)

    try:
        job_id = await _ft.create_job(client, body)
    except ImportError as exc:
        raise _extra_missing_error(exc) from exc
    return JobRef(
        job_id=job_id,
        training_file_id=training_file_id,
        validation_file_id=validation_file_id,
    )


async def get_job(client: Any, job_id: str) -> JobResult:
    """学習ジョブの現在状態を単発で照会する（待機しない）。

    プラットフォームの生の状態文字列は `raw_status` へ保全し、`status` は終端 3 種 + queued の
    みを判定した写像結果を持つ（未知の状態値は RUNNING へ倒す・FR-6）。失敗したジョブも例外に
    せず、理由を `error_message` に持つ `JobResult` として返す。

    Args:
        client: 利用者が構築した OpenAI / Azure OpenAI の非同期クライアント
            （`AsyncOpenAI` / `AsyncAzureOpenAI` 等・必須）。
        job_id: 照会するジョブ id。

    Returns:
        `JobResult`（状態・生の状態文字列・学習済みモデル参照・失敗理由）。

    Raises:
        FineTuneError: `client` が不在の場合（`CONFIG_MISSING`）、照会に失敗した場合
            （`API_ERROR`・adapter からそのまま伝播）。`finetune` extra（openai）が未導入の
            場合は `EXTRA_MISSING`（導入ヒント付き）。`API_ERROR` の `message` は
            プラットフォームのレスポンス本文を逐語で含む。送信した設定値がエコーバックされうる
            ため、信頼できない相手へそのまま提示・記録しない。
    """
    from ..._adapters import finetune as _ft

    if client is None:
        raise _config_error(
            "client が指定されていません"
            "（AsyncOpenAI / AsyncAzureOpenAI 等の非同期クライアントが必須です）"
        )
    try:
        payload = await _ft.retrieve_job(client, job_id)
    except ImportError as exc:
        raise _extra_missing_error(exc) from exc
    raw_status = payload["status"]
    return JobResult(
        job_id=payload["id"],
        status=_map_status(raw_status),
        raw_status=raw_status,
        model_ref=payload["fine_tuned_model"],
        error_message=payload["error_message"],
    )


async def wait_job(
    client: Any,
    job_id: str,
    *,
    timeout: float,
    poll_interval: float = _DEFAULT_POLL_INTERVAL,
) -> JobResult:
    """学習ジョブが終端状態へ到達するまで単発照会を反復して待機する（ADR 0031）。

    lib 内で唯一のポーリングループであり、build-don't-run の明示的な例外として、利用者が本関数
    を明示的に呼び出したときにのみ待機する。ループの中身は `get_job` 相当の単発照会だけで、
    ジョブの起動・取消・再試行・モデル切替は行わない。初回照会は sleep 前に即時実行し、待機は
    `min(poll_interval, 残時間)` 秒ずつ行うため deadline を超過しない。未知の状態値は非終端と
    して待機を継続する。

    `timeout` は既定値のない必須 keyword 引数で、無限待機の経路を構造的に排除する。timeout
    到達時もジョブは取り消さないため、同じ job_id で `get_job` / `wait_job` を再実行できる。

    Args:
        client: 利用者が構築した OpenAI / Azure OpenAI の非同期クライアント
            （`AsyncOpenAI` / `AsyncAzureOpenAI` 等・必須）。
        job_id: 待機するジョブ id。
        timeout: 待機の上限秒数（必須 keyword・正値）。
        poll_interval: 照会間隔の秒数（正値・既定 30.0）。小さすぎる値はレート制限・課金に
            直結する（FT ジョブは分〜時間単位のため既定 30.0 を推奨）。

    Returns:
        終端状態（succeeded / failed / cancelled）へ到達した `JobResult`。失敗して終了した
        場合も例外にせず、理由を `error_message` に持つ結果を返す。

    Raises:
        FineTuneError: `client` が不在、または `timeout` / `poll_interval` が非正値の場合
            （`CONFIG_MISSING`）、終端へ到達しないまま `timeout` を超過した場合（`TIMEOUT`）、
            または照会が失敗した場合（`API_ERROR`・adapter からそのまま伝播し待機を打ち切る）。
            `finetune` extra（openai）が未導入の場合は `EXTRA_MISSING`（`get_job` が変換・
            待機を打ち切る）。`API_ERROR` の `message` はプラットフォームのレスポンス本文を
            逐語で含む。送信した設定値がエコーバックされうるため、信頼できない相手へそのまま
            提示・記録しない。
    """
    if client is None:
        raise _config_error(
            "client が指定されていません"
            "（AsyncOpenAI / AsyncAzureOpenAI 等の非同期クライアントが必須です）"
        )
    if timeout <= 0:
        raise _config_error(f"timeout には正の秒数を指定してください（受領値: {timeout}）")
    if poll_interval <= 0:
        raise _config_error(
            f"poll_interval には正の秒数を指定してください（受領値: {poll_interval}）"
        )

    deadline = time.monotonic() + timeout
    while True:
        result = await get_job(client, job_id)
        if result.is_terminal:
            return result
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise FineTuneError(
                FineTuneFailureKind.TIMEOUT,
                f"ジョブ {job_id} が {timeout} 秒以内に終端状態へ到達しませんでした"
                f"（最後に観測した状態: {result.raw_status}）。"
                "ジョブは取り消していないため、同じ job_id で get_job / wait_job を"
                "再実行して待機を続けられます",
            )
        await asyncio.sleep(min(poll_interval, remaining))
