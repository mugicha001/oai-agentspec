"""ケース列を DPO（preference 形式）データセットへ変換する最小例（API キー不要）。

`to_dpo_dataset` は `DpoCase`（typed なケース型）と plain dict の両方を受け、出力側は常に
assistant メッセージ配列として `preferred_output` / `non_preferred_output` へ載る。SFT との
差分（`input.messages` の入れ子・`system=` を持たない）を print で確認する。

実行:
    uv run python examples/finetune/02_dpo_dataset.py

導入: pip install 'oai-agentspec[finetune]'
"""

from __future__ import annotations

from oai_agentspec.runtime.finetune import DpoCase, to_dpo_dataset


def main() -> None:
    """`DpoCase` と plain dict のケースを DPO データセットへ変換し、出力構造を確認する。"""
    # DpoCase: typed なケース型。id / metadata は変換結果のレコードには載らない。
    typed_cases = [
        DpoCase(
            input="請求書の再発行をお願いします",
            preferred_output="かしこまりました。ご登録のメールアドレス宛に再送します。",
            non_preferred_output="無理です。",
            id="case-1",
        ),
    ]

    # plain dict でも同じキー名（input / preferred_output / non_preferred_output）で渡せる。
    dict_cases = [
        {
            "input": "アプリが起動しません",
            "preferred_output": "お手数ですが再インストールをお試しください。",
            "non_preferred_output": "知りません。",
        },
    ]

    result = to_dpo_dataset([*typed_cases, *dict_cases])

    print(f"records={len(result.records)} skipped={result.skipped}")
    for record in result.records:
        # SFT の "messages" 直下とは異なり、DPO は "input": {"messages": [...]} の入れ子になる。
        print("input.messages       =", record["input"]["messages"])
        # preferred_output / non_preferred_output は常に assistant メッセージ配列。
        print("preferred_output     =", record["preferred_output"])
        print("non_preferred_output =", record["non_preferred_output"])
        print("---")


if __name__ == "__main__":
    main()
