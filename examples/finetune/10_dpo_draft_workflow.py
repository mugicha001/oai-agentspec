"""会話履歴から DPO 雛形を書き出し、記入して取り込むワークフローの例（API キー不要）。

どちらの応答が「望ましい」かは品質判定であり lib は内蔵しない。そこで
`dpo_dataset_from_session` は 2 つのモードを持つ。

- **雛形モード（`pair_builder` 省略）**: 記入用ケース列を返す。`save_dpo_draft` で CSV /
  JSONL へ書き出し、人がスプレッドシート等で 2 欄を埋め、`finalize_dpo_draft` で最終
  データセットへ取り込む（本 example の主題）。
- **callable モード（`pair_builder` 指定）**: 既存のルール・別モデルの出力等から機械的に
  ペアを組む。本 example の末尾で短く示す。

本 example は SQLite の一時 Session に会話を書き込み、記入もプログラムで代行するため
実 API 接続を伴わない。設計判断は
`docs/adr/0034-session-tool-roundtrip-and-dpo-draft.md` を参照。

押さえておく挙動:

- **記入用ケースは最終レコードではない**。雛形モードの `records` は
  `input` / `preferred_output` / `non_preferred_output` / `response` の 4 キーからなる
  記入用のケースで、最終形になるのは `finalize_dpo_draft` を通した後である。
- **未記入（strip 後に空）のケースは自動で skip される**。空白のみのセルも未記入扱いで、
  `skipped` に計上される（全件未記入でもエラーにならず空の結果が返る）。
- **片欄だけ埋めるとエラーになる**。`ケース N: ...` の位置つき `VALIDATION_FAILED` で
  失敗する（書きかけを暗黙に捨てない）。
- **CSV は utf-8-sig（BOM 付き）で書く**。日本語環境の Excel でそのまま開ける。表計算ソフトが
  cp932 等で保存し直したファイルは取り込めないため、UTF-8 で保存する。
- **長すぎるセルは書き出し時に弾く**。標準 `csv` の読み取り上限（`csv.field_size_limit()`・
  既定 131072 文字）を超えるセルは、記入前に `VALIDATION_FAILED` で失敗させる（書けたのに
  読めない雛形を作らないため）。超える会話を扱うときは利用者側で上限を引き上げる。
- **`input_json` 列は編集しない**。文脈の復元はこの列だけが担う（`context` 列は人が読む
  ための非可逆な整形で、取り込みには使わない）。
- **`response` 列は参照専用**。ログ上の実応答を見ながら記入するための欄で、最終レコードには
  現れない。
- **記入値は非改変で載る**。strip は未記入判定にだけ使い、採用した値は前後の空白・改行を
  含めてそのまま学習データになる。
- **ツール出力は雛形ファイルにそのまま載る**。機密情報・個人情報を含む場合は
  `save_dpo_draft` へ渡す前に `draft.records` を加工して除去する。`=` などで始まる値を
  表計算ソフトが数式として解釈しうる点にも注意する。
- **ツール定義は `finalize_dpo_draft(source, tools=...)` で渡す**。雛形ファイルは定義を
  保持しない（CSV 列は変わらない）。
- **雛形モードで `tools=` を渡すとエラーになる**。記入用ケース列は `to_dpo_dataset` へ委譲
  しないため反映先が無く、`CONFIG_MISSING` で fail-closed に弾く（callable モードと
  `finalize_dpo_draft` では渡せる）。

実行:
    uv run python examples/finetune/10_dpo_draft_workflow.py

導入: pip install 'oai-agentspec[finetune]'
"""

from __future__ import annotations

import asyncio
import csv
import json
import tempfile
from pathlib import Path
from typing import Any

from agents import SQLiteSession

from oai_agentspec.runtime.finetune import (
    dpo_dataset_from_session,
    finalize_dpo_draft,
    save_dpo_draft,
    validate_dataset,
)

# ツール往復を含む会話ログ。function_call / function_call_output は chat 形式へ変換されて
# 文脈（input）に残るため、ツール利用の根拠ごと preference 学習に使える。
CONVERSATION: list[dict[str, Any]] = [
    {"role": "user", "content": "在庫を確認してほしい"},
    {
        "type": "function_call",
        "name": "check_stock",
        "arguments": '{"sku": "A-100"}',
        "call_id": "call_1",
    },
    {"type": "function_call_output", "call_id": "call_1", "output": '{"count": 3}'},
    {"role": "assistant", "content": "在庫は 3 個です。"},
    {"role": "user", "content": "取り置きできますか?"},
    {"role": "assistant", "content": "できます。"},
]

# 人が記入する内容の代役（実運用では CSV をスプレッドシートで開いて人が埋める）。
# ケース 2 は「まだ判断できない」ものとして両欄を空のままにし、skip されることを示す。
FILLS: list[tuple[str, str]] = [
    ("在庫は 3 個です。すぐに発送できます。", "3"),
    ("", ""),
]

# 会話で使われた check_stock に対応するツール定義（plain dict）。雛形は定義を持ち回らない
# ため、finalize_dpo_draft へ渡した時点で record["input"]["tools"] へ載る。
TOOLS: list[dict[str, object]] = [
    {
        "type": "function",
        "function": {
            "name": "check_stock",
            "description": "SKU の在庫数を確認する",
            "parameters": {
                "type": "object",
                "properties": {"sku": {"type": "string"}},
                "required": ["sku"],
            },
        },
    }
]


def fill_csv_as_human_would(path: Path, fills: list[tuple[str, str]]) -> None:
    """記入 2 列へ値を書き込む（スプレッドシートでの人手記入の代役）。

    参照列（case_index / context / response）と `input_json` 列は触らない。列名ベースで
    読み書きするため列順は問われないが、`input_json` を壊すと文脈を復元できなくなる。

    Args:
        path: `save_dpo_draft` が書き出した CSV のパス。
        fills: 各ケースへ入れる `(preferred_output, non_preferred_output)` の列。
    """
    with path.open(encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        columns = list(reader.fieldnames or [])
        rows = list(reader)

    for row, (preferred, non_preferred) in zip(rows, fills, strict=True):
        row["preferred_output"] = preferred
        row["non_preferred_output"] = non_preferred

    with path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def build_pair_from_rule(material: dict[str, Any]) -> dict[str, Any] | None:
    """ケース素材から機械的に preference ペアを組む（callable モードの例）。

    素材は `{"input": 累積文脈, "response": ログ上の実応答}` の 2 キー。`None` を返した
    ケースは skip され `skipped` に計上される。

    Args:
        material: ケース素材。

    Returns:
        `preferred_output` / `non_preferred_output` を持つ dict、または skip を表す None。
    """
    response = str(material["response"])
    if len(response) < 8:
        # 短すぎる応答は「望ましくない側」の見本にすらしない、という判断の例。
        return None
    return {"preferred_output": response, "non_preferred_output": "わかりません。"}


async def main() -> None:
    """雛形の書き出し -> 記入 -> 取り込み -> 検証を一通り通し、callable モードも示す。"""
    with tempfile.TemporaryDirectory() as tmp:
        session = SQLiteSession("support-2026-09", Path(tmp) / "history.db")
        await session.add_items(CONVERSATION)

        # 1. 雛形モード: 記入用ケース列を得る（まだ最終レコードではない）。
        draft = await dpo_dataset_from_session(session)
        print(f"[1] 記入用ケース: {len(draft.records)} 件 / skipped: {draft.skipped}")
        print(json.dumps(draft.records[0], ensure_ascii=False, indent=2))

        # 2. CSV へ書き出す（.jsonl も同じ関数で扱える）。
        draft_path = Path(tmp) / "dpo_draft.csv"
        save_dpo_draft(draft, draft_path)
        print(f"\n[2] 雛形を書き出した: {draft_path.name}")
        print(draft_path.read_text(encoding="utf-8-sig").rstrip())

        # 3. 記入（実運用ではここで人がスプレッドシートを開いて 2 欄を埋める）。
        fill_csv_as_human_would(draft_path, FILLS)
        print("\n[3] 記入した（ケース 2 は両欄を空のままにして skip させる）")

        # 4. 取り込み: 記入済みの雛形を最終 preference データセットへ変換する。
        # ツール定義はここで初めて渡す（雛形は持ち回らないため）。record["input"]["tools"]
        # へ載る（SFT のレコード直下とは透過位置が異なる）。
        result = finalize_dpo_draft(draft_path, tools=TOOLS)
        print(f"\n[4] 最終レコード: {len(result.records)} 件 / skipped: {result.skipped}")
        for record in result.records:
            print(json.dumps(record, ensure_ascii=False, indent=2))

        # 5. 検証（fail-closed・違反ゼロのときだけ ok=True）。
        report = validate_dataset(result.records, method="dpo")
        print(f"\n[5] 検証: ok={report.ok} checked={report.checked}")

        # 6. 参考: callable モードなら記入ワークフローを挟まずに 1 度で組める。
        # callable モードは雛形モードと違い tools= をそのまま渡せる（反映先がある）。
        auto = await dpo_dataset_from_session(
            session, pair_builder=build_pair_from_rule, tools=TOOLS
        )
        print(f"\n[6] callable モード: {len(auto.records)} 件 / skipped: {auto.skipped}")
        print("    tools =", auto.records[0]["input"]["tools"])

    print()
    print("この結果はそのまま投入できる:")
    print("    job = await submit_job(client, train=result, model=..., method='dpo')")


if __name__ == "__main__":
    asyncio.run(main())
