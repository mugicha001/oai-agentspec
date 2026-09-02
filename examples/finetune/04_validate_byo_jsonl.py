"""持ち込み JSONL を `validate_dataset` で検証する例（API キー不要）。

正しい JSONL と、違反を数種類含む JSONL を一時ファイルへ書き出し、`DatasetValidationReport`
の読み方（`ok` / `checked` / `violations` の `line` と `reason`）を示す。あわせて
`raise_on_invalid=True` の挙動、変換側 `skip_missing=True` によるケース除外 + `skipped` 件数の
報告も示す。

**`validate_dataset` だけでは投入可否を判定できない**。本関数が見るのはメッセージ**単位**の
合法性で、メッセージ**間**の順序制約（ツール往復の並び）は `screen_tool_roundtrips` の責務で
ある。両方を通さないと、1 件ずつは合法なのに並びが不正な学習データが素通りする。両ゲートを
まとめて適用し合格・不合格へ仕分けるなら `partition_dataset` を使う
（`11_screen_and_partition.py` を参照）。

実行:
    uv run python examples/finetune/04_validate_byo_jsonl.py

導入: pip install 'oai-agentspec[finetune]'
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from oai_agentspec.runtime.finetune import FineTuneError, to_sft_dataset, validate_dataset


def main() -> None:
    """正しい JSONL / 違反入り JSONL の検証と、`raise_on_invalid` / `skip_missing` を示す。"""
    valid_lines = [
        {
            "messages": [
                {"role": "user", "content": "質問"},
                {"role": "assistant", "content": "回答"},
            ]
        },
    ]
    # 違反例: role 不正 / weight が不正な float / content が空リスト。
    invalid_lines = [
        {
            "messages": [
                {"role": "narrator", "content": "不正な role"},
                {"role": "assistant", "content": "回答"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "質問"},
                {"role": "assistant", "content": "回答", "weight": 0.5},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": []},
                {"role": "assistant", "content": "回答"},
            ]
        },
    ]

    with tempfile.TemporaryDirectory() as tmp:
        valid_path = Path(tmp) / "valid.jsonl"
        valid_path.write_text(
            "\n".join(json.dumps(line, ensure_ascii=False) for line in valid_lines) + "\n",
            encoding="utf-8",
        )
        invalid_path = Path(tmp) / "invalid.jsonl"
        invalid_path.write_text(
            "\n".join(json.dumps(line, ensure_ascii=False) for line in invalid_lines) + "\n",
            encoding="utf-8",
        )

        print("--- 正しい JSONL の検証")
        ok_report = validate_dataset(valid_path, method="sft")
        print(f"ok={ok_report.ok} checked={ok_report.checked} violations={ok_report.violations}")

        print("--- 違反入り JSONL の検証")
        bad_report = validate_dataset(invalid_path, method="sft")
        print(f"ok={bad_report.ok} checked={bad_report.checked}")
        for violation in bad_report.violations:
            # line は物理行番号（1 始まり）、reason は人間可読の違反理由。
            print(f"  line={violation.line} reason={violation.reason}")

        print("--- raise_on_invalid=True は不合格で FineTuneError を送出する")
        try:
            validate_dataset(invalid_path, method="sft", raise_on_invalid=True)
        except FineTuneError as exc:
            print(f"FineTuneError: {exc} | report.ok={exc.report.ok if exc.report else None}")

    print("--- skip_missing=True は変換側でケース不備を除外し skipped に件数報告する")
    cases = [
        {"input": "質問1", "expected_output": "回答1"},
        {"input": "質問2"},  # expected_output が欠落（skip_missing なしなら FineTuneError）
        {"input": "質問3", "expected_output": "回答3"},
    ]
    result = to_sft_dataset(cases, skip_missing=True)
    print(f"records={len(result.records)} skipped={result.skipped}")


if __name__ == "__main__":
    main()
