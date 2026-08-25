"""マネージド Fine-Tuning の SDK 窓口（openai 結合を `_adapters` に閉じる・NFR-1）。

`import openai` を本モジュールの関数内遅延 import（`_require_openai`）に閉じ、学習ファイルの
アップロード（`upload_file`）・ファイル処理完了の待機（`wait_file_processed`）・学習ジョブの
作成（`create_job`）・ジョブ状態の照会（`retrieve_job`）という単発の SDK 呼び出しだけを提供する。
ロジック層（`runtime/finetune`）へは plain な str / dict のみを返し、openai の型は一切出さない。

設計の核:
    - **分配規則**: `create_job` は `model` / `training_file` のみを SDK ネイティブ引数として渡し、
      残りのキーはすべて `extra_body` へ一括委譲する。SDK ネイティブ引数のキー名リストを本
      モジュールへ持たないため、プラットフォーム固有キー（`trainingType` 等）や SDK の版差に
      追随する必要がない。body の組み立て・衝突検出は `runtime/finetune/jobs.py` の責務。
    - **例外変換**: 呼び出しの失敗は `FineTuneError(API_ERROR)` へ変換し、プラットフォームの
      理由文言を保全する（lib 側で妥当性を先回り検証しない・FR-5 / FR-10）。既に
      `FineTuneError` の失敗は kind / message を保つため再ラップせず re-raise する。理由文言は
      逐語で載るため、`API_ERROR` の `message` にはプラットフォームのレスポンス本文（送信した
      設定値のエコーバックを含みうる）が入る。信頼できない相手へそのまま提示・記録しない。
    - **extra 未導入の変換はしない**: `_require_openai` は導入ヒント付きの生 `ImportError` に
      留め、`EXTRA_MISSING` への変換は `jobs.py` 側で行う（lightning / judge と同型の分業）。
    - lib 独自のポーリングループ・再試行は持たない（build-don't-run）。ファイル処理の完了待機は
      SDK ヘルパ（`files.wait_for_processing`）への 1 回の委譲として `wait_file_processed` が
      担い、ジョブ完了の待機は `runtime/finetune.wait_job` にある。

env は参照しない（接続設定は利用者が構築した client で受領する・NFR-3）。
"""

from __future__ import annotations

from typing import Any, Final

# `files.wait_for_processing` へ渡す照会間隔（秒）。SDK 既定には依存せず lib 側で固定する。
_FILE_POLL_INTERVAL: Final[float] = 2.0

# finetune extra（openai）未導入時の案内。
_FINETUNE_INSTALL_HINT = (
    "Fine-Tuning ジョブ管理には openai が必要です。"
    "次でインストールしてください: pip install 'oai-agentspec[finetune]'"
)


def _require_openai() -> Any:
    """openai を遅延 import する（未導入時は案内付き ImportError）。

    Returns:
        openai モジュール。

    Raises:
        ImportError: openai が未導入の場合（案内文字列付き・`FineTuneError` へは変換しない）。
    """
    try:
        import openai  # noqa: F401
    except ImportError as exc:  # pragma: no cover - 環境依存
        raise ImportError(_FINETUNE_INSTALL_HINT) from exc
    return openai


def _describe(exc: BaseException) -> str:
    """例外を「型名は常に・本文は非空のときだけ」の形式で整形する。

    Args:
        exc: 整形対象の例外。

    Returns:
        `"型名: 本文"`（本文非空時）または `"型名"`（本文空時）。
    """
    body = str(exc)
    return f"{type(exc).__name__}: {body}" if body else type(exc).__name__


def _raise_api_error(exc: BaseException, *, operation: str) -> None:
    """SDK 呼び出しの失敗を `FineTuneError(API_ERROR)` へ変換して送出する。

    既に `FineTuneError` の失敗は kind / message を保つため再ラップせず同一オブジェクトを
    re-raise する（二重変換の回避）。

    Args:
        exc: 捕捉した例外。
        operation: 失敗した操作の説明（メッセージへ埋め込む）。

    Raises:
        FineTuneError: 常に送出する（本関数は戻らない）。
    """
    from ..runtime.finetune.types import FineTuneError, FineTuneFailureKind

    if isinstance(exc, FineTuneError):
        raise exc
    raise FineTuneError(
        FineTuneFailureKind.API_ERROR,
        f"{operation}に失敗しました: {_describe(exc)}",
    ) from exc


async def upload_file(client: Any, *, data: bytes, filename: str) -> str:
    """学習 / 検証データを Fine-Tuning 用ファイルとしてアップロードする（単発）。

    `purpose="fine-tune"` を明示し、ファイル名付きタプル `(filename, data)` で送る
    （生 bytes だとプラットフォーム側が拡張子を判別できない）。

    Args:
        client: 利用者が構築した OpenAI / Azure OpenAI の非同期クライアント。
        data: JSONL のバイト列（keyword-only）。
        filename: 送信時のファイル名（`"train.jsonl"` / `"validation.jsonl"`・keyword-only）。

    Returns:
        アップロード済みファイルの id（**処理完了は保証しない**。ジョブ作成に使う前に
        `wait_file_processed` で待つこと）。

    Raises:
        FineTuneError: アップロードに失敗した場合（`API_ERROR`・理由文言を保全）。
        ImportError: openai が未導入の場合。
    """
    _require_openai()
    try:
        response = await client.files.create(file=(filename, data), purpose="fine-tune")
    except Exception as exc:
        _raise_api_error(exc, operation="学習ファイルのアップロード")
    return str(response.id)


async def wait_file_processed(client: Any, file_id: str, *, timeout: float) -> str:
    """アップロード済みファイルが `processed` へ到達するまで待つ（SDK ヘルパへ 1 回委譲）。

    プラットフォームはアップロード直後のファイル（`status="pending"`）でのジョブ作成を 400 で
    拒む（ADR 0032）。待機の実体は SDK の `files.wait_for_processing` で、lib 側は
    `poll_interval` / `max_wait_seconds` の双方を明示指定するだけの薄い結線に留める
    （SDK 既定値の版差に依存しない）。SDK は `error` / `deleted` の終端でも例外を投げずに返す
    ため、`processed` 以外は一律 fail-closed で失敗させる。

    Args:
        client: 利用者が構築した OpenAI / Azure OpenAI の非同期クライアント。
        file_id: 待機対象のアップロード済みファイル id。
        timeout: 待機の上限秒数（keyword-only・`max_wait_seconds` へそのまま渡す）。

    Returns:
        `processed` へ到達した場合の `file_id`（受け取った値をそのまま返す）。

    Raises:
        FineTuneError: 終端 status が `processed` 以外の場合（`API_ERROR`・
            `status_details` があれば逐語で含める）、SDK の上限超過 `RuntimeError` の場合
            （`TIMEOUT`）、その他の SDK 呼び出し失敗の場合（`API_ERROR`・理由文言を保全）。
        ImportError: openai が未導入の場合。
    """
    from ..runtime.finetune.types import FineTuneError, FineTuneFailureKind

    _require_openai()
    try:
        file_obj = await client.files.wait_for_processing(
            file_id,
            poll_interval=_FILE_POLL_INTERVAL,
            max_wait_seconds=timeout,
        )
    except RuntimeError as exc:
        raise FineTuneError(
            FineTuneFailureKind.TIMEOUT,
            f"ファイル {file_id} の処理が上限 {timeout} 秒（概算）以内に完了しませんでした。"
            "ファイルは削除していないため、処理完了後に同じファイル id を train= / val= へ"
            "文字列で渡して再実行できます",
        ) from exc
    except Exception as exc:
        _raise_api_error(exc, operation="学習ファイルの処理完了待機")

    status = getattr(file_obj, "status", None)
    if status != "processed":
        details = getattr(file_obj, "status_details", None)
        suffix = f"（理由: {details}）" if details else ""
        raise FineTuneError(
            FineTuneFailureKind.API_ERROR,
            f"ファイル {file_id} の処理が完了しませんでした（status: {status}）{suffix}",
        )
    return file_id


async def create_job(client: Any, body: dict[str, Any]) -> str:
    """学習ジョブを作成する（単発）。

    `model` / `training_file` のみを SDK ネイティブ引数として渡し、残りのキーはすべて
    `extra_body` へ委譲する。`body` は破壊しない（コピーしてから除去する）。

    Args:
        client: 利用者が構築した OpenAI / Azure OpenAI の非同期クライアント。
        body: `jobs.py` が組み上げたリクエスト body（`model` / `training_file` を含む）。

    Returns:
        作成したジョブの id。

    Raises:
        FineTuneError: 作成に失敗した場合（`API_ERROR`・理由文言を保全）。既に
            `FineTuneError` の失敗はそのまま re-raise する。
        ImportError: openai が未導入の場合。
    """
    _require_openai()
    extra_body = {
        key: value for key, value in body.items() if key not in ("model", "training_file")
    }
    try:
        response = await client.fine_tuning.jobs.create(
            model=body["model"],
            training_file=body["training_file"],
            extra_body=extra_body,
        )
    except Exception as exc:
        _raise_api_error(exc, operation="学習ジョブの作成")
    return str(response.id)


async def retrieve_job(client: Any, job_id: str) -> dict[str, Any]:
    """学習ジョブの現在状態を単発で照会する（ポーリングはしない）。

    `status` は加工せず生文字列のまま返す（`JobStatus` への写像は
    `runtime/finetune/types._map_status` の責務）。

    Args:
        client: 利用者が構築した OpenAI / Azure OpenAI の非同期クライアント。
        job_id: 照会するジョブ id。

    Returns:
        `{"id", "status", "fine_tuned_model", "error_message"}` の plain dict。
        `error_message` は失敗理由が無い場合 None。

    Raises:
        FineTuneError: 照会に失敗した場合（`API_ERROR`・理由文言を保全）。既に
            `FineTuneError` の失敗はそのまま re-raise する。
        ImportError: openai が未導入の場合。
    """
    _require_openai()
    try:
        job = await client.fine_tuning.jobs.retrieve(job_id)
    except Exception as exc:
        _raise_api_error(exc, operation="学習ジョブの照会")
    error = getattr(job, "error", None)
    message = getattr(error, "message", None) if error is not None else None
    return {
        "id": job.id,
        "status": job.status,
        "fine_tuned_model": job.fine_tuned_model,
        "error_message": str(message) if message is not None else None,
    }
