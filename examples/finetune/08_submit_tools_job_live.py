"""ツール定義つきの学習データを実 API へ投入する例（.env 必須・**課金が発生する**）。

`ToolRegistry` に登録したツールを `to_sft_dataset` / `to_dpo_dataset` の `tools=` へ渡し、
`submit_job` でジョブとして投入する。SFT と DPO の両方を扱い、既定は SFT・`--method dpo` で
DPO へ切り替える。

送信内容の形（tools がレコード側に入り body 直下には出ないこと）を課金なしで確かめたい場合は
`05_job_body_preview.py` を使うこと。ツールなしの最小例は `06`（SFT）/ `07`（DPO）にある。

**このスクリプトは従量課金の操作を行う**（学習ファイルのアップロードとジョブ投入）。実行前に
確認プロンプトを出し、費用を抑えるため最小データ・1 エポックを既定にしている。

ツール定義の扱い:

- **学習データと推論時のツール定義が同じ `ToolRegistry` から出る**。`registry.<name>` は属性
  アクセス時に SDK `FunctionTool` を遅延構築して返し、`tools=` はそれをダックタイピングで検出
  して公式 tools 定義形式へ写像する。学習時と推論時の定義がずれない。
- **SFT はレコード直下の `tools`、DPO は `input.tools`** へ入る（形式はプラットフォーム仕様）。
  本ライブラリは内容を解釈せず透過するため、tools 定義の妥当性判定はプラットフォームが行う。
- 学習データには `tool_calls` つき assistant と role `"tool"` のメッセージを含められる
  （ツールを呼ぶ判断そのものを学習させる）。

必要な環境変数（リポジトリ直下の env ファイル）:
    FINETUNE_SFT_BASE_MODEL 学習対象のベースモデル名（`--method sft` のとき）
    FINETUNE_DPO_BASE_MODEL 学習対象のベースモデル名（`--method dpo` のとき）
    FINETUNE_TRAINING_TYPE  学習実行方式（任意・**Azure 専用**・未設定なら非送信）
    その他の接続設定は `06_submit_job_live.py` と同じ（`.env.example` を参照）

実行（確認プロンプトが出る）:
    uv run python examples/finetune/08_submit_tools_job_live.py
    uv run python examples/finetune/08_submit_tools_job_live.py --method dpo

非対話環境で確認を省略して実行する（課金が発生する）:
    uv run python examples/finetune/08_submit_tools_job_live.py --yes

導入: pip install 'oai-agentspec[finetune]'
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

from oai_agentspec import ToolRegistry, ToolSpec
from oai_agentspec.runtime.finetune import (
    FineTuneError,
    get_job,
    submit_job,
    to_dpo_dataset,
    to_sft_dataset,
    validate_dataset,
)

# examples/ 共有ヘルパ _azure を解決するため _shared を import パスへ。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
from _azure import (  # noqa: E402
    build_finetune_client,
    finetune_provider,
    finetune_training_type,
    load_env,
)


def get_order_status(order_id: str) -> str:
    """注文番号から配送状況を返す（example 用のダミー実装）。

    Args:
        order_id: 注文番号。

    Returns:
        配送状況の説明文。
    """
    return f"{order_id} は本日発送予定です"


def _tool_call_case(order_id: str, answer: str) -> dict[str, Any]:
    """ツール呼び出しを含む 1 ケースを組み立てる（SFT 用）。

    Args:
        order_id: 会話に登場する注文番号。
        answer: 最終的な assistant の応答。

    Returns:
        `to_sft_dataset` が受理するケース dict。
    """
    return {
        "input": [
            {"role": "user", "content": f"{order_id} の配送状況を教えて"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_order_status",
                            "arguments": f'{{"order_id": "{order_id}"}}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "本日発送予定"},
        ],
        "expected_output": answer,
    }


# SFT: ツールを呼んでから答える会話を学習させる。
SFT_CASES = [
    _tool_call_case(f"A-{1000 + i}", f"A-{1000 + i} は本日発送予定です") for i in range(10)
]

# DPO: 同じ問い合わせに対し「ツールで確認してから答える」応答を preferred とする。
DPO_CASES = [
    {
        "input": [{"role": "user", "content": f"A-{2000 + i} の配送状況を教えて"}],
        "preferred_output": f"注文 A-{2000 + i} を確認しました。本日発送予定です。",
        "non_preferred_output": "確認できません。",
    }
    for i in range(10)
]

DPO_VALIDATION_CASES = [
    {
        "input": [{"role": "user", "content": f"B-{3000 + i} の配送状況を教えて"}],
        "preferred_output": f"注文 B-{3000 + i} を確認しました。明日発送予定です。",
        "non_preferred_output": "知りません。",
    }
    for i in range(3)
]


def parse_method() -> str:
    """コマンドライン引数から学習メソッドを決める。

    Returns:
        `"sft"`（既定）または `"dpo"`。
    """
    argv = sys.argv[1:]
    if "--method" in argv:
        index = argv.index("--method")
        if index + 1 < len(argv):
            return argv[index + 1].strip().lower()
    return "sft"


def confirm(method: str) -> bool:
    """課金が発生することを提示し、実行の確認を取る。

    `--yes` を付けて実行した場合は確認を省略する（非対話環境で流す場合はこちらを使う）。

    Args:
        method: 学習メソッド（表示に使う）。

    Returns:
        利用者が実行を選んだ場合に True。
    """
    if "--yes" in sys.argv[1:]:
        print("--yes が指定されたため確認を省略します（課金が発生します）")
        return True
    print("=" * 72)
    print(f"このスクリプトはツール定義つきの {method.upper()} ジョブを投入します（従量課金）。")
    print("  - 学習ファイルのアップロードとジョブ作成が実 API に対して行われます")
    print("  - 学習トークン数 x エポック数に応じた課金が発生します")
    print("  - tools 定義の妥当性はプラットフォームが判定します（非対応なら API_ERROR）")
    print("=" * 72)
    try:
        answer = input("実行しますか? [y/N]: ").strip().lower()
    except EOFError:
        print()
        print("標準入力が対話的ではないため中止しました（実行するなら --yes を付けてください）")
        return False
    return answer == "y"


async def main() -> None:
    """ツール定義つきデータで SFT / DPO ジョブを投入し、状態を 1 回照会する。"""
    load_env()
    method = parse_method()
    if method not in ("sft", "dpo"):
        print(f"--method には sft か dpo を指定してください（受領値: {method}）")
        return

    env_key = "FINETUNE_SFT_BASE_MODEL" if method == "sft" else "FINETUNE_DPO_BASE_MODEL"
    base_model = os.getenv(env_key)
    if not base_model:
        print(f"{env_key} が未設定です（{method.upper()} 対象のベースモデル名を設定してください）")
        print(f"例: {env_key}=gpt-4.1-mini-2025-04-14")
        return

    # 推論時に Agent へ渡すのと同じ Registry から学習データの tools 定義を出す。
    registry = ToolRegistry()
    registry.register(ToolSpec(name="get_order_status", func=get_order_status))
    tools = [registry.get_order_status]

    if method == "sft":
        dataset = to_sft_dataset(SFT_CASES, tools=tools)
        validation = None
    else:
        dataset = to_dpo_dataset(DPO_CASES, tools=tools)
        validation = to_dpo_dataset(DPO_VALIDATION_CASES, tools=tools)

    for label, records in (
        ("学習", dataset.records),
        ("検証", validation.records if validation else None),
    ):
        if records is None:
            continue
        report = validate_dataset(records, method=method)
        print(
            f"{label}データ: ok={report.ok} / checked={report.checked}"
            f" / violations={len(report.violations)}"
        )
        if not report.ok:
            for violation in report.violations:
                print(f"  line {violation.line}: {violation.reason}")
            return

    print(f"メソッド: {method} / ベースモデル: {base_model}")
    print(f"ツール定義: {[t.name for t in tools]}")
    if not confirm(method):
        print("中止しました（課金は発生していません）")
        return

    client = build_finetune_client()
    try:
        # 学習実行方式（Azure 固有）は FINETUNE_TRAINING_TYPE で指定する（未設定なら送信しない）。
        training_type = finetune_training_type() if finetune_provider() == "azure" else None
        hyperparameters: dict[str, Any] = {"n_epochs": 1}
        if method == "dpo":
            hyperparameters["beta"] = 0.1
        job = await submit_job(
            client,
            train=dataset.records,
            val=validation.records if validation else None,
            model=base_model,
            method=method,
            hyperparameters=hyperparameters,
            training_type=training_type,
            suffix=f"oas-tools-{method}",
        )
        print(f"投入しました: job_id={job.job_id}")
        print(f"  training_file={job.training_file_id}")
        if job.validation_file_id:
            print(f"  validation_file={job.validation_file_id}")

        result = await get_job(client, job.job_id)
        print(f"状態: {result.status}（生の状態: {result.raw_status}）")
        if result.error_message:
            print(f"失敗理由: {result.error_message}")

        print()
        print("ジョブは投入済みです。完了まで待つ場合の書き方:")
        print(f"    result = await wait_job(client, '{job.job_id}', timeout=3600.0)")
    except FineTuneError as error:
        print(f"失敗しました: kind={error.kind}")
        print(f"  {error.message}")
        if error.kind.value == "api_error":
            print()
            print("tools 定義がプラットフォームに受理されなかった可能性があります")
            print("（本ライブラリは tools の内容を解釈せず透過します）")


if __name__ == "__main__":
    asyncio.run(main())
