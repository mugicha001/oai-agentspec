"""実 API へ FT ジョブを投入し、状態を照会する例（.env 必須・**課金が発生する**）。

`submit_job` で学習ファイルのアップロードとジョブ作成を 1 呼び出しで行い、`get_job` で状態と
完成モデル参照（`model_ref`）を取得する。送信されるリクエスト body の中身を確かめたいだけなら、
課金の発生しない `05_job_body_preview.py` を使うこと。

**このスクリプトは従量課金の操作を行う**（学習ファイルのアップロードとジョブ投入）。実行前に
確認プロンプトを出す。費用を抑えるため、既定で最小データ（10 件）と Azure の Developer training
（最も安価・データレジデンシー保証なし・スポット容量のためプリエンプトあり）を使う。

ジョブは分〜時間単位で完了するため、本 example は投入と 1 回の状態照会までで終える。完了まで
待つ場合は `wait_job`（timeout 必須・lib 内唯一のポーリングループ）を使う:

    result = await wait_job(client, job.job_id, timeout=3600.0)   # poll_interval 既定 30 秒

Azure では完成した `model_ref` はデプロイ前のモデル参照であり、推論に使うには Azure 側での
デプロイ操作（本ライブラリのスコープ外・利用者責任）が別途必要になる。

必要な環境変数（リポジトリ直下の env ファイル）:
    EXAMPLES_LLM_PROVIDER   "azure"（既定）| "openai"
    AZURE_OPENAI_*          Azure の接続情報（provider=azure のとき）
    OPENAI_API_KEY          OpenAI の API キー（provider=openai のとき）
    FINETUNE_SFT_BASE_MODEL 学習対象のベースモデル名（デプロイ名ではなくモデル名）
    FINETUNE_TRAINING_TYPE  学習実行方式（任意・**Azure 専用**。Developer / GlobalStandard 等。
                            未設定なら非送信。OpenAI 直接続では設定されていても送らない）

fine-tuning が使えるリージョンは限られるため、推論とは別のリソースになることがある。その場合は
FT 専用のオーバーライド（AZURE_OPENAI_FINETUNE_ENDPOINT / _API_KEY / _API_VERSION、OpenAI 直接続
なら OPENAI_FINETUNE_API_KEY）を設定する。未設定なら上記の推論用設定へフォールバックする。

実行（確認プロンプトが出る）:
    uv run python examples/finetune/06_submit_job_live.py

非対話環境で確認を省略して実行する（課金が発生する）:
    uv run python examples/finetune/06_submit_job_live.py --yes

導入: pip install 'oai-agentspec[finetune]'
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from oai_agentspec.runtime.finetune import (
    FineTuneError,
    get_job,
    submit_job,
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

# 学習に使う最小データ。実運用では数十〜数百件を用意する（OpenAI の SFT は 10 件が下限目安）。
CASES = [
    {"input": "請求書の再発行をお願いします", "expected_output": "billing"},
    {"input": "支払い方法を変更したい", "expected_output": "billing"},
    {"input": "領収書が届きません", "expected_output": "billing"},
    {"input": "二重に課金されています", "expected_output": "billing"},
    {"input": "アプリが起動しません", "expected_output": "support"},
    {"input": "ログインできなくなりました", "expected_output": "support"},
    {"input": "画面が真っ白になります", "expected_output": "support"},
    {"input": "通知が届かないです", "expected_output": "support"},
    {"input": "解約の手続きを教えてください", "expected_output": "account"},
    {"input": "登録メールアドレスを変えたい", "expected_output": "account"},
]

SYSTEM_PROMPT = "あなたは問い合わせを billing / support / account のいずれかへ分類します。"


def confirm() -> bool:
    """課金が発生することを提示し、実行の確認を取る。

    `--yes` を付けて実行した場合は確認を省略する（非対話環境で流す場合はこちらを使う）。

    Returns:
        利用者が実行を選んだ場合に True。
    """
    if "--yes" in sys.argv[1:]:
        print("--yes が指定されたため確認を省略します（課金が発生します）")
        return True
    print("=" * 72)
    print("このスクリプトは fine-tuning ジョブを投入します（従量課金が発生します）。")
    print("  - 学習ファイルのアップロードとジョブ作成が実 API に対して行われます")
    print("  - 学習トークン数 x エポック数に応じた課金が発生します")
    print("  - ジョブは投入後にキャンセルするまで走り続けます（本 example は取り消しません）")
    print("=" * 72)
    try:
        answer = input("実行しますか? [y/N]: ").strip().lower()
    except EOFError:
        # 非対話環境（パイプ実行・CI 等）では確認を取れないため中止する。
        print()
        print("標準入力が対話的ではないため中止しました（実行するなら --yes を付けてください）")
        return False
    return answer == "y"


async def main() -> None:
    """最小データで SFT ジョブを投入し、状態と model_ref を 1 回照会する。"""
    load_env()
    base_model = os.getenv("FINETUNE_SFT_BASE_MODEL")
    if not base_model:
        print("FINETUNE_SFT_BASE_MODEL が未設定です（SFT 対象のベースモデル名を設定してください）")
        print("例: FINETUNE_SFT_BASE_MODEL=gpt-4.1-mini-2025-04-14")
        return

    # 投入前にデータを整形・検証する。検証はネットワークに触れないので課金は発生しない。
    dataset = to_sft_dataset(CASES, system=SYSTEM_PROMPT)
    report = validate_dataset(dataset.records, method="sft")
    print(f"検証: ok={report.ok} / checked={report.checked} / violations={len(report.violations)}")
    if not report.ok:
        for violation in report.violations:
            print(f"  line {violation.line}: {violation.reason}")
        return

    print(f"ベースモデル: {base_model}")
    print(f"学習レコード: {len(dataset.records)} 件")
    if not confirm():
        print("中止しました（課金は発生していません）")
        return

    client = build_finetune_client()
    try:
        # 学習実行方式（Azure 固有）は FINETUNE_TRAINING_TYPE で指定する（未設定なら送信しない）。
        # OpenAI 直接続では未知フィールドとして拒否されるため、設定されていても送らない。
        training_type = finetune_training_type() if finetune_provider() == "azure" else None
        job = await submit_job(
            client,
            train=dataset.records,
            model=base_model,
            method="sft",
            hyperparameters={"n_epochs": 1},
            training_type=training_type,
            suffix="oas-demo",
        )
        print(f"投入しました: job_id={job.job_id} / training_file={job.training_file_id}")

        result = await get_job(client, job.job_id)
        print(f"状態: {result.status}（生の状態: {result.raw_status}）")
        print(f"終端か: {result.is_terminal}")
        if result.model_ref:
            print(f"完成モデル: {result.model_ref}")
        if result.error_message:
            print(f"失敗理由: {result.error_message}")

        print()
        print("ジョブは投入済みです。完了まで待つ場合の書き方:")
        print(f"    result = await wait_job(client, '{job.job_id}', timeout=3600.0)")
        print("状態の再確認:")
        print(f"    result = await get_job(client, '{job.job_id}')")
    except FineTuneError as error:
        print(f"失敗しました: kind={error.kind}")
        print(f"  {error.message}")


if __name__ == "__main__":
    asyncio.run(main())
