"""実 API へ DPO（選好学習）ジョブを投入し、状態を照会する例（.env 必須・**課金が発生する**）。

`to_dpo_dataset` で preferred / non_preferred のペアを preference 形式へ整形し、`submit_job` へ
`method="dpo"` で渡す。SFT 版は `06_submit_job_live.py`、送信内容だけを確かめたい場合は課金の
発生しない `05_job_body_preview.py` を使うこと。

**このスクリプトは従量課金の操作を行う**（学習・検証ファイルのアップロードとジョブ投入）。
実行前に確認プロンプトを出し、最小データ（学習 10 ペア / 検証 3 ペア）・1 エポックを既定に
している。公式の DPO 手順は training と validation の両方を渡す例を示しているため、本 example
も検証データを渡す（`val=`）。

DPO 固有の注意:

- **対応モデルが SFT より狭い**。本ライブラリは対応モデル一覧を保持しないため、非対応の
  組み合わせはプラットフォームのエラー（`API_ERROR`・理由文言を保全）で判明する。
- **`beta`** は DPO 固有のハイパーパラメータ（大きいほど参照モデルからの乖離に強い罰則）。
  lib は値を解釈せず `method.dpo.hyperparameters` へ透過する。
- **1 例につき最後の assistant メッセージ 1 件**が学習対象になる（プラットフォーム仕様）。
- ベースモデルにも SFT 済みモデルにも適用できる。SFT の後に DPO を重ねる 2 段構成は、利用者が
  2 回のジョブとして実行する（`06` の完成 `model_ref` を `FINETUNE_DPO_BASE_MODEL` へ渡す）。

必要な環境変数（リポジトリ直下の env ファイル）:
    EXAMPLES_LLM_PROVIDER   "azure"（既定）| "openai"
    AZURE_OPENAI_*          Azure の接続情報（provider=azure のとき）
    OPENAI_API_KEY          OpenAI の API キー（provider=openai のとき）
    FINETUNE_DPO_BASE_MODEL 学習対象のベースモデル名（DPO 対応モデルであること）
    FINETUNE_TRAINING_TYPE  学習実行方式（任意・**Azure 専用**。Developer / GlobalStandard 等。
                            未設定なら非送信。OpenAI 直接続では設定されていても送らない）

FT 専用のオーバーライド（AZURE_OPENAI_FINETUNE_ENDPOINT / _API_KEY / _API_VERSION、OpenAI 直接続
なら OPENAI_FINETUNE_API_KEY）を設定すると、推論とは別のリソースへ投入できる。未設定なら上記の
推論用設定へフォールバックする。

実行（確認プロンプトが出る）:
    uv run python examples/finetune/07_submit_dpo_job_live.py

非対話環境で確認を省略して実行する（課金が発生する）:
    uv run python examples/finetune/07_submit_dpo_job_live.py --yes

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
    to_dpo_dataset,
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

# 選好ペア。preferred は「注文番号を尋ねて次の行動へ繋ぐ」応答、non_preferred は
# 「突き放して会話を終わらせる」応答とし、望ましい応答スタイルを学習させる。
CASES = [
    {
        "input": "配送状況を教えてください",
        "preferred_output": "注文番号を教えていただけますか。すぐにお調べします。",
        "non_preferred_output": "わかりません。",
    },
    {
        "input": "返品したいのですが",
        "preferred_output": "商品到着から 14 日以内でしたら承ります。注文番号をお願いします。",
        "non_preferred_output": "返品はできません。",
    },
    {
        "input": "領収書が届きません",
        "preferred_output": "ご登録のメールアドレスをご確認のうえ、注文番号をお知らせください。",
        "non_preferred_output": "迷惑メールを見てください。",
    },
    {
        "input": "支払い方法を変更したい",
        "preferred_output": "マイページの支払い設定から変更いただけます。手順をご案内しますか。",
        "non_preferred_output": "自分で調べてください。",
    },
    {
        "input": "アプリが起動しません",
        "preferred_output": "端末とアプリのバージョンを教えていただけますか。順に確認します。",
        "non_preferred_output": "再インストールしてください。",
    },
    {
        "input": "ログインできなくなりました",
        "preferred_output": "パスワード再設定のメールをお送りできます。ご確認ください。",
        "non_preferred_output": "パスワードを忘れたのが原因です。",
    },
    {
        "input": "解約の手続きを教えてください",
        "preferred_output": "マイページから手続きできます。ご不明点があればお手伝いします。",
        "non_preferred_output": "解約はできません。",
    },
    {
        "input": "二重に課金されています",
        "preferred_output": "ご不便をおかけしました。注文番号を確認のうえ返金をご案内します。",
        "non_preferred_output": "そんなはずはありません。",
    },
    {
        "input": "通知が届かないです",
        "preferred_output": "端末の通知設定とアプリ内設定を順に確認しましょう。",
        "non_preferred_output": "設定の問題です。",
    },
    {
        "input": "登録メールアドレスを変えたい",
        "preferred_output": "マイページのアカウント設定から変更できます。手順をお送りしますか。",
        "non_preferred_output": "変更できません。",
    },
]

# 検証データ。学習データとは別の問い合わせを用意する（同一データを使い回すと検証にならない）。
# 公式の DPO 手順は training と validation の両方を渡す例を示している。
VALIDATION_CASES = [
    {
        "input": "クーポンが適用されません",
        "preferred_output": "クーポンの条件と有効期限を確認します。コードを教えてください。",
        "non_preferred_output": "使えないクーポンです。",
    },
    {
        "input": "パスワードの変更方法を教えてください",
        "preferred_output": "マイページのセキュリティ設定から変更できます。ご案内しますか。",
        "non_preferred_output": "設定を見てください。",
    },
    {
        "input": "注文をキャンセルしたい",
        "preferred_output": "発送前でしたら承ります。注文番号をお知らせください。",
        "non_preferred_output": "キャンセルはできません。",
    },
]


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
    print("このスクリプトは DPO の fine-tuning ジョブを投入します（従量課金が発生します）。")
    print("  - 学習 / 検証ファイルのアップロードとジョブ作成が実 API に対して行われます")
    print("  - 学習トークン数 x エポック数に応じた課金が発生します")
    print("  - DPO 対応モデルは SFT より狭く、非対応なら API_ERROR で失敗します")
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
    """最小データで DPO ジョブを投入し、状態と model_ref を 1 回照会する。"""
    load_env()
    base_model = os.getenv("FINETUNE_DPO_BASE_MODEL")
    if not base_model:
        print("FINETUNE_DPO_BASE_MODEL が未設定です（DPO 対応のベースモデル名を設定してください）")
        print("例: FINETUNE_DPO_BASE_MODEL=gpt-4.1-mini-2025-04-14")
        print("SFT の成果物へ重ねる場合は 06 が返した model_ref（ft:... 形式）を設定する")
        return

    # 投入前に整形・検証する。検証はネットワークに触れないので課金は発生しない。
    dataset = to_dpo_dataset(CASES)
    validation = to_dpo_dataset(VALIDATION_CASES)
    for label, records in (("学習", dataset.records), ("検証", validation.records)):
        report = validate_dataset(records, method="dpo")
        print(
            f"{label}データ: ok={report.ok} / checked={report.checked}"
            f" / violations={len(report.violations)}"
        )
        if not report.ok:
            for violation in report.violations:
                print(f"  line {violation.line}: {violation.reason}")
            return

    print(f"ベースモデル: {base_model}")
    print(f"選好ペア: 学習 {len(dataset.records)} 件 / 検証 {len(validation.records)} 件")
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
            val=validation.records,
            model=base_model,
            method="dpo",
            hyperparameters={"beta": 0.1, "n_epochs": 1},
            training_type=training_type,
            suffix="oas-dpo",
        )
        print(f"投入しました: job_id={job.job_id}")
        print(f"  training_file={job.training_file_id} / validation_file={job.validation_file_id}")

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
    except FineTuneError as error:
        print(f"失敗しました: kind={error.kind}")
        print(f"  {error.message}")
        if error.kind.value == "api_error":
            print()
            print("DPO 非対応のモデルを指定していないか確認してください")
            print("（対応モデルは SFT より狭く、本ライブラリは一覧を保持しません）")


if __name__ == "__main__":
    asyncio.run(main())
