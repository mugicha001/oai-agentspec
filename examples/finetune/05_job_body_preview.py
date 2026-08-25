"""`submit_job` の引数がジョブ作成リクエストのどのフィールドになるかを可視化する例（API キー不要）。

`submit_job` は利用者が渡した設定を解釈せずプラットフォームへ透過する。本 example は送信内容を
記録するだけの疑似 client を渡し、組み上がったリクエスト body を印字することで、引数と API
フィールドの対応（`method="sft"` -> `method.type = "supervised"` の写像、`training_type` ->
body 直下の `trainingType`、`hyperparameters` の階層配置）を実行結果として示す。

`tools=` を伴うデータセットでは、tools 定義が**アップロードされる JSONL のレコード側**へ入り、
ジョブ作成リクエストの body 直下には現れないことも示す（body 直下に載るのは `training_type` や
`suffix` のようなジョブ設定だけである）。

あわせて lib が守る 2 つの契約を確認する:

- **未指定のフィールドは送信しない**: 最小引数では body のキーが `model` / `training_file` /
  `method` の 3 つだけになる（既定値を発明せず、`None` も明示送信しない）。学習完了後の自動
  デプロイを有効化するフィールドも lib からは付加しない（ホスティング課金の暗黙発生を防ぐ）。
- **`extra_body` は lib が組み立てるキーと衝突しない**: 交差した場合は送信前に
  `FineTuneError`（`CONFIG_MISSING`）で失敗し、暗黙のマージ・上書きをしない。

実 API へ投入する例は `06_submit_job_live.py`（`.env` 必須・課金あり）を参照。

実行:
    uv run python examples/finetune/05_job_body_preview.py

導入: pip install 'oai-agentspec[finetune]'
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

from oai_agentspec import ToolRegistry, ToolSpec
from oai_agentspec.runtime.finetune import (
    FineTuneError,
    submit_job,
    to_dpo_dataset,
    to_sft_dataset,
)


class RecordingClient:
    """送信内容を記録するだけの疑似 client（ネットワークへは一切出ない）。

    `submit_job` は client を不透明値として扱い、次の 3 つを単発で呼ぶ。疑似 client を自作する
    場合はこの 3 つを備える必要がある:

    - `files.create`: 学習 / 検証データのアップロード
    - `files.wait_for_processing`: アップロードしたファイルの処理完了待ち（プラットフォームは
      未処理のファイル id でのジョブ作成を拒否するため。`train` / `val` にデータを渡した場合のみ）
    - `fine_tuning.jobs.create`: ジョブ作成

    Attributes:
        uploads: `files.create` へ渡された kwargs の記録。
        waits: `files.wait_for_processing` へ渡された引数の記録。
        jobs: `fine_tuning.jobs.create` へ渡された kwargs の記録。
    """

    def __init__(self) -> None:
        self.uploads: list[dict[str, Any]] = []
        self.waits: list[dict[str, Any]] = []
        self.jobs: list[dict[str, Any]] = []
        outer = self

        class _Files:
            async def create(self, **kwargs: Any) -> Any:
                outer.uploads.append(kwargs)
                return SimpleNamespace(id=f"file-{len(outer.uploads)}")

            async def wait_for_processing(self, file_id: str, **kwargs: Any) -> Any:
                outer.waits.append({"file_id": file_id, **kwargs})
                return SimpleNamespace(id=file_id, status="processed")

        class _Jobs:
            async def create(self, **kwargs: Any) -> Any:
                outer.jobs.append(kwargs)
                return SimpleNamespace(id="ftjob-preview")

        class _FineTuning:
            jobs = _Jobs()

        self.files = _Files()
        self.fine_tuning = _FineTuning()

    def last_body(self) -> dict[str, Any]:
        """直近のジョブ作成で送信されたリクエスト body を復元する。

        SDK は `model` / `training_file` をネイティブ引数、残りを `extra_body` として受け取る
        （lib はキー名リストを抱えず一括委譲する）。両者を合成したものが実際の body になる。

        Returns:
            body 直下のキーと値。
        """
        call = self.jobs[-1]
        return {
            "model": call["model"],
            "training_file": call["training_file"],
            **call["extra_body"],
        }


def show(label: str, body: dict[str, Any]) -> None:
    """リクエスト body をキー集合つきで印字する。"""
    print(f"--- {label} ---")
    print("キー集合:", sorted(body))
    print(json.dumps(body, ensure_ascii=False, indent=2))
    print()


def show_uploaded_record(label: str, client: RecordingClient) -> None:
    """直近にアップロードされた JSONL の 1 行目を印字する。"""
    filename, data = client.uploads[-1]["file"]
    first_line = data.decode("utf-8").splitlines()[0]
    print(f"--- {label} ---")
    print(f"filename: {filename}")
    print(json.dumps(json.loads(first_line), ensure_ascii=False, indent=2))
    print()


def get_order_status(order_id: str) -> str:
    """注文番号から配送状況を返す（example 用のダミー実装）。

    Args:
        order_id: 注文番号。

    Returns:
        配送状況の説明文。
    """
    return f"{order_id} は本日発送予定です"


async def main() -> None:
    """最小引数・フル指定・DPO・tools 付き・extra_body・衝突検出の各ケースで送信内容を観察する。"""
    cases = [
        {"input": "請求書の再発行をお願いします", "expected_output": "billing"},
        {"input": "アプリが起動しません", "expected_output": "support"},
    ]

    # 1. 最小引数: 未指定のフィールドは 1 つも送らない（NFR-7 の契約）。
    #    train に文字列を渡すとアップロード済みのファイル id として扱われる（再送しない）。
    client = RecordingClient()
    await submit_job(
        client,
        train="file-abc123",
        model="gpt-4.1-mini-2025-04-14",
        method="sft",
    )
    show("最小引数（キーは 3 つだけ）", client.last_body())
    print("アップロード回数:", len(client.uploads), "（str はファイル id 扱いなので 0 回）\n")

    # 2. フル指定: ポータルのウィザードで選ぶ項目に対応する引数をすべて渡す。
    #    method="sft" は API 側の "supervised" へ写像され、hyperparameters は写像後の
    #    type 値の下（method.supervised.hyperparameters）へ入る。
    client = RecordingClient()
    await submit_job(
        client,
        train=cases,
        val=cases,
        model="gpt-4.1-mini-2025-04-14",
        method="sft",
        hyperparameters={"n_epochs": 3, "batch_size": "auto"},
        training_type="Developer",
        suffix="support-bot",
        seed=42,
    )
    show("フル指定（training_type は wire key の trainingType へ）", client.last_body())
    print("アップロード:", [call["file"][0] for call in client.uploads])
    print("（レコード列は JSONL 化してアップロードする。filename で train / validation を区別）")
    print("処理完了待ち:", [(w["file_id"], w["max_wait_seconds"]) for w in client.waits])
    print("（アップロード直後のファイルは未処理でジョブ作成に使えないため、")
    print("  ファイルごとに処理完了を待ってからジョブを作成する）\n")

    # 3. DPO: method="dpo" は写像せずそのまま送られ、hyperparameters は method.dpo 配下へ入る。
    #    beta は DPO 固有のハイパーパラメータだが、lib は構造も許容値も持たず非解釈で透過する
    #    （将来メソッドの識別子・未知のハイパーパラメータも同じ経路で通る）。
    dpo_cases = [
        {
            "input": "配送状況を教えてください",
            "preferred_output": "注文番号を教えていただけますか。すぐにお調べします。",
            "non_preferred_output": "わかりません。",
        },
        {
            "input": "返品したいのですが",
            "preferred_output": "商品到着から 14 日以内でしたら返品を承ります。",
            "non_preferred_output": "返品はできません。",
        },
    ]
    client = RecordingClient()
    await submit_job(
        client,
        train=to_dpo_dataset(dpo_cases).records,
        model="gpt-4.1-mini-2025-04-14",
        method="dpo",
        hyperparameters={"beta": 0.1, "n_epochs": 2},
    )
    show("DPO（beta は method.dpo.hyperparameters へ）", client.last_body())

    # 4. tools 付きデータ: tools はレコード側（アップロードされる JSONL）に入り、
    #    ジョブ作成リクエストの body 直下には現れない。SFT / DPO とも同じ扱い。
    registry = ToolRegistry()
    registry.register(ToolSpec(name="get_order_status", func=get_order_status))

    tool_cases = [
        {
            "input": [
                {"role": "user", "content": "A-1234 の配送状況を教えて"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_order_status",
                                "arguments": '{"order_id": "A-1234"}',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "本日発送予定"},
            ],
            "expected_output": "A-1234 は本日発送予定です",
        }
    ]
    client = RecordingClient()
    await submit_job(
        client,
        train=to_sft_dataset(tool_cases, tools=[registry.get_order_status]).records,
        model="gpt-4.1-mini-2025-04-14",
        method="sft",
    )
    show("tools 付き SFT（body 直下に tools は出ない）", client.last_body())
    show_uploaded_record("アップロードされた JSONL の 1 行目（tools はここに入る）", client)

    dpo_tool_cases = [
        {
            "input": [{"role": "user", "content": "A-1234 の配送状況を教えて"}],
            "preferred_output": "注文番号を確認しました。本日発送予定です。",
            "non_preferred_output": "わかりません。",
        }
    ]
    client = RecordingClient()
    await submit_job(
        client,
        train=to_dpo_dataset(dpo_tool_cases, tools=[registry.get_order_status]).records,
        model="gpt-4.1-mini-2025-04-14",
        method="dpo",
    )
    show("tools 付き DPO（body 直下に tools は出ない）", client.last_body())
    show_uploaded_record("アップロードされた JSONL の 1 行目（tools は input 配下）", client)

    # 5. Azure 固有・将来追加のフィールドは extra_body で透過する。
    #    lib が組み立てるキーと交差した場合は送信前に CONFIG_MISSING で失敗する。
    client = RecordingClient()
    await submit_job(
        client,
        train="file-abc123",
        model="gpt-4.1-mini-2025-04-14",
        method="sft",
        extra_body={"integrations": [{"type": "wandb", "wandb": {"project": "ft-demo"}}]},
    )
    show("extra_body で追加フィールドを透過", client.last_body())

    client = RecordingClient()
    try:
        await submit_job(
            client,
            train="file-abc123",
            model="gpt-4.1-mini-2025-04-14",
            method="sft",
            suffix="support-bot",
            extra_body={"suffix": "conflicting"},
        )
    except FineTuneError as error:
        print("--- 衝突検出 ---")
        print("kind:", error.kind)
        print("message:", error.message)
        print("ジョブ作成の呼び出し回数:", len(client.jobs), "（送信前に失敗する）")


if __name__ == "__main__":
    asyncio.run(main())
