"""Fine-Tuning の client レベル fake（実 API のファイル処理制約を再現する）。

契約の出所（実 API・Azure OpenAI / OpenAI の双方で確認）:
    - `files.create` 直後のファイルは `status="pending"`。遷移は
      `pending` -> `running` -> `processed` で、終端は `{processed, error, deleted}`。
    - **未処理のファイル id でジョブを作成すると 400**:
      `The specified file reference must point to a completed file import.`
    - `files.wait_for_processing(id, *, poll_interval, max_wait_seconds)` は終端まで待ち、
      `error` / `deleted` でも例外を投げずに FileObject を返す。上限超過は `RuntimeError`。

本 fake は adapter を差し替えずに「`submit_job` -> 実 adapter -> client」の鎖を測るために
使う（adapter 関数自体を差し替える fake ではこの欠陥が届かない）。

検出力の限界（意図的な設計）:
    **本 fake が発行していないファイル id は `processed` 扱いとし 400 を出さない**
    （未知 id ポリシー）。`create_job(client, body)` を直接呼ぶ既存テスト
    （`training_file="file-abc"` 等）を偽陽性で赤化させないためであり、その代償として
    fake 非発行 id を使う経路には未処理ファイル検出の力を持たない。
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

# 実 API が未処理ファイルでのジョブ作成に返す 400 の本文（逐語）。
UNPROCESSED_FILE_MESSAGE = "The specified file reference must point to a completed file import."


class FakeFileNotProcessedError(Exception):
    """未処理ファイル id でのジョブ作成に対する 400 相当の例外（実 API の文言を持つ）。

    openai の例外型は使わない（adapter は `except Exception` で捕捉して `API_ERROR` へ
    変換するため、型ではなく文言の保全が測定点になる）。
    """


class FakeFineTuneClient:
    """`files` / `fine_tuning.jobs` を持つ疑似 client（ファイル処理状態を持つ）。

    Attributes:
        calls: SDK 呼び出しの発生順（`"files.create"` / `"wait_for_processing"` /
            `"jobs.create"`）。
        file_calls: `files.create` へ渡された kwargs の列。
        wait_calls: `wait_for_processing` の呼び出し記録
            （`{"file_id", "poll_interval", "max_wait_seconds"}`）。
        job_calls: `fine_tuning.jobs.create` へ渡された kwargs の列。
        statuses: 本 fake が発行したファイル id -> 現在の status。
    """

    def __init__(
        self,
        *,
        job_id: str = "ftjob-fake",
        file_id_prefix: str = "file-fake",
        wait_status: str = "processed",
        status_details: str | None = None,
        wait_error: BaseException | None = None,
    ) -> None:
        """fake client を生成する。

        Args:
            job_id: `fine_tuning.jobs.create` が返すジョブ id。
            file_id_prefix: 発行するファイル id の接頭辞（`file-fake-1` のように連番を付ける）。
            wait_status: `wait_for_processing` が遷移させる先の status
                （`"processed"` / `"error"` / `"deleted"` を注入できる）。
            status_details: 返す FileObject に載せる `status_details`（None なら属性を持たせない）。
            wait_error: `wait_for_processing` が送出する例外（`RuntimeError` 等を注入できる）。
        """
        self.job_id = job_id
        self._file_id_prefix = file_id_prefix
        self.wait_status = wait_status
        self.status_details = status_details
        self.wait_error = wait_error

        self.calls: list[str] = []
        self.file_calls: list[dict[str, Any]] = []
        self.wait_calls: list[dict[str, Any]] = []
        self.job_calls: list[dict[str, Any]] = []
        self.statuses: dict[str, str] = {}

        outer = self

        class _Files:
            async def create(self, **kwargs: Any) -> Any:
                return outer._files_create(**kwargs)

            async def wait_for_processing(
                self,
                file_id: str,
                *,
                poll_interval: float,
                max_wait_seconds: float,
            ) -> Any:
                return outer._wait_for_processing(
                    file_id,
                    poll_interval=poll_interval,
                    max_wait_seconds=max_wait_seconds,
                )

        class _Jobs:
            async def create(self, **kwargs: Any) -> Any:
                return outer._jobs_create(**kwargs)

        class _FineTuning:
            jobs = _Jobs()

        self.files = _Files()
        self.fine_tuning = _FineTuning()

    # ------------------------------------------------------------------
    # SDK 呼び出しの実体
    # ------------------------------------------------------------------

    def _file_object(self, file_id: str) -> Any:
        """現在の status を反映した FileObject 相当を組み立てる。"""
        attrs: dict[str, Any] = {"id": file_id, "status": self.statuses[file_id]}
        if self.status_details is not None:
            attrs["status_details"] = self.status_details
        return SimpleNamespace(**attrs)

    def _files_create(self, **kwargs: Any) -> Any:
        """アップロード直後の `status="pending"` なファイルを発行する（実測どおり）。"""
        self.calls.append("files.create")
        self.file_calls.append(kwargs)
        file_id = f"{self._file_id_prefix}-{len(self.file_calls)}"
        self.statuses[file_id] = "pending"
        return self._file_object(file_id)

    def _wait_for_processing(
        self, file_id: str, *, poll_interval: float, max_wait_seconds: float
    ) -> Any:
        """`pending` -> 注入 status の 1 段遷移を行う（`running` は挟まない）。"""
        self.calls.append("wait_for_processing")
        self.wait_calls.append(
            {
                "file_id": file_id,
                "poll_interval": poll_interval,
                "max_wait_seconds": max_wait_seconds,
            }
        )
        if self.wait_error is not None:
            raise self.wait_error
        self.statuses[file_id] = self.wait_status
        return self._file_object(file_id)

    def _jobs_create(self, **kwargs: Any) -> Any:
        """`training_file` / `validation_file` の双方を検査してからジョブを作る。

        未知 id（本 fake が発行していない id）は `processed` 扱いで通す。
        """
        self.calls.append("jobs.create")
        self.job_calls.append(kwargs)
        extra_body = kwargs.get("extra_body") or {}
        candidates = [
            kwargs.get("training_file"),
            extra_body.get("training_file"),
            kwargs.get("validation_file"),
            extra_body.get("validation_file"),
        ]
        for file_id in candidates:
            if not isinstance(file_id, str):
                continue
            status = self.statuses.get(file_id)
            if status is not None and status != "processed":
                raise FakeFileNotProcessedError(UNPROCESSED_FILE_MESSAGE)
        return SimpleNamespace(id=self.job_id)
