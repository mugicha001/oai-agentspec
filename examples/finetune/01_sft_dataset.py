"""ケース列を SFT（chat 形式）データセットへ変換する最小例（API キー不要）。

`to_sft_dataset` はケース列（plain dict / 属性アクセス型の混在可）を OpenAI 公式 SFT 形式の
レコード列へ変換する。単一ターン（文字列の `input` / `expected_output`）と複数ターン
（messages 形式のリスト）の両方を示し、`system=` の付与・`DatasetBuildResult.save` による
JSONL 書き出し・`validate_dataset` での合格確認までを 1 本で通す。

本 example は純データ変換のみで `agents` / `openai` を import せずネットワークにも触れない
（`oai-agentspec[finetune]` の段階 1 は API 接続を持たない）。

実行:
    uv run python examples/finetune/01_sft_dataset.py

導入: pip install 'oai-agentspec[finetune]'
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from oai_agentspec.runtime.finetune import to_sft_dataset, validate_dataset


def main() -> None:
    """単一ターン / 複数ターンのケースを SFT データセットへ変換し、保存・検証まで通す。"""
    # 単一ターン: input / expected_output は文字列でよい（内部で 1 件のメッセージへ包まれる）。
    single_turn_cases = [
        {"input": "請求書の再発行をお願いします", "expected_output": "billing"},
        {"input": "アプリが起動しません", "expected_output": "support"},
    ]

    # 複数ターン: input / expected_output に messages 形式のリストを渡すと非改変のまま透過する。
    multi_turn_cases = [
        {
            "input": [
                {"role": "user", "content": "配送状況を教えてください"},
                {"role": "assistant", "content": "注文番号を教えてください"},
                {"role": "user", "content": "A-1234 です"},
            ],
            "expected_output": "A-1234 は本日発送予定です",
        },
    ]

    # system= は全レコード先頭へ挿入される（input リスト内に system が既にあると競合エラー）。
    result = to_sft_dataset(
        [*single_turn_cases, *multi_turn_cases],
        system="あなたはカスタマーサポートの担当者です。",
    )

    print(f"records={len(result.records)} skipped={result.skipped}")
    for record in result.records:
        print(record)

    # 保存は opt-in。リポジトリを汚さないよう一時ディレクトリへ書き出す。
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "sft_dataset.jsonl"
        result.save(out)
        print(f"saved (temp): {out} | exists={out.exists()}")

        # 書き出した JSONL を validate_dataset で検証する（method="sft" が既定）。
        report = validate_dataset(out, method="sft")
        print(f"validate: ok={report.ok} checked={report.checked} violations={report.violations}")


if __name__ == "__main__":
    main()
