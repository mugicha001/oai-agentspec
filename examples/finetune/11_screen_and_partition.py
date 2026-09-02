"""学習データのツール往復の並びを submit 前に検査する例（API キー不要）。

`validate_dataset`（FR-3）はメッセージ**単位**の合法性を見る。しかしメッセージが 1 件ずつ
合法でも、並べたときに「応答のない `tool_calls`」になっているデータは推論時 API が拒否する。
この**メッセージ間**の順序制約を見るのが `screen_tool_roundtrips`（FR-13）で、submit 前に 2 つを
並べて呼ぶ。設計判断は `docs/adr/0037-tool-roundtrip-screening-separation.md` を参照。

押さえておく挙動:

- **`validate_dataset` は非隣接な往復を合格にする**。準拠先が違うため（公式データ形式 対
  推論時 API の順序要求）で、片方の合格は他方の合格を含意しない。本 example の 2 件目が
  その実例である。
- **生成はツール往復の並びを理由にケースを捨てない**。`dataset_from_session` /
  `dpo_dataset_from_session` は履歴を忠実に変換し、形式の判定は本ゲートへ集約する
  （生成と精査の責務分離）。捨てるのは空文脈・空応答と、利用者が `case_filter` /
  `pair_builder` で外したものだけである。
- **持ち込み JSONL も同じゲートを通せる**。source はファイルパスとレコード列の二形を受ける
  ため、手で作ったデータ・別ツールの出力・過去に生成したファイルも検査できる。
- **群内の順序は問わない**。並列ツール呼び出しの対応は id ベースであり順序に意味を持たない
  ため、出力が呼び出し順と逆でも合格する。
- **構造違反は報告しない**。レコードが非 dict / messages が欠落・非リストといった構造の
  壊れは `validate_dataset` の責務で、二重報告しない（どちらを直せばよいか分からなくなる）。
- **`raise_on_invalid=True` で fail-closed にできる**。`submit_job` は暗黙のスクリーニングを
  行わないため、投入前のゲートは利用者が明示的に置く。
- **全件止めずに進めるなら `partition_dataset`**。両ゲートを合成して「投入できるもの」と
  「できないもの」へ仕分ける。合格側は `DatasetBuildResult` なので `submit_job(train=...)` と
  `save(path)` へ詰め替えなしで渡せ、不合格側は元レコードと理由を抱えて返るため、直して
  再投入する動線が切れない。
- **違反理由は `messages[N]:` 形式で位置を示す**。`validate_dataset` と同じ書式なので、
  構造の違反と並びの違反が同じ読み方で並ぶ。

実行:
    uv run python examples/finetune/11_screen_and_partition.py

導入: pip install 'oai-agentspec[finetune]'
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any

from agents import SQLiteSession

from oai_agentspec.runtime.finetune import (
    FineTuneError,
    dataset_from_session,
    partition_dataset,
    screen_tool_roundtrips,
    validate_dataset,
)

# メッセージ単位で違反するレコード（role が不正）。screening は素通しするが
# partition_dataset は validate 側の違反として不合格へ落とす。
BAD_STRUCTURE: dict[str, Any] = {
    "messages": [
        {"role": "customer", "content": "役割が不正"},
        {"role": "assistant", "content": "応答"},
    ]
}


def _assistant_calls(*call_ids: str) -> dict[str, Any]:
    """指定 id の `tool_calls` を持つ assistant メッセージを組む（持ち込みデータの模擬）。"""
    return {
        "role": "assistant",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "check_stock", "arguments": '{"sku": "A-100"}'},
            }
            for call_id in call_ids
        ],
    }


def _tool(call_id: str) -> dict[str, Any]:
    """指定 id へ応答する role `"tool"` メッセージを組む。"""
    return {"role": "tool", "tool_call_id": call_id, "content": '{"count": 3}'}


# 手で組んだ持ち込みレコード。3 件目だけが順序制約に違反している。
BROUGHT_IN: list[dict[str, Any]] = [
    # 逐次呼び出し（呼び出しごとに直後で応答）。
    {
        "messages": [
            {"role": "user", "content": "A-100 の在庫を教えて"},
            _assistant_calls("call_a"),
            _tool("call_a"),
            {"role": "assistant", "content": "在庫は 3 個です。"},
        ]
    },
    # 並列呼び出し。出力順が呼び出し順と逆でも合格する（群内は順序非依存）。
    {
        "messages": [
            {"role": "user", "content": "2 件まとめて調べて"},
            _assistant_calls("call_a", "call_b"),
            _tool("call_b"),
            _tool("call_a"),
            {"role": "assistant", "content": "どちらも在庫があります。"},
        ]
    },
    # 呼び出しと応答の間に assistant テキストが挟まる（承認待ちの通知を書き込む等）。
    # メッセージ単位では全て合法だが、この並びは推論時 API が拒否する。
    {
        "messages": [
            {"role": "user", "content": "A-100 の在庫を教えて"},
            _assistant_calls("call_c"),
            {"role": "assistant", "content": "確認します。少々お待ちください。"},
            _tool("call_c"),
            {"role": "assistant", "content": "在庫は 3 個です。"},
        ]
    },
]

# HITL 承認で中断されたランを模した履歴。累積ペアリングは往復の途中でも切り出すため、
# 生成結果には順序制約に違反するケースが含まれる（生成はこれを捨てない）。
INTERRUPTED_RUN: list[dict[str, Any]] = [
    {"role": "user", "content": "在庫を確認して"},
    {
        "type": "function_call",
        "name": "check_stock",
        "arguments": '{"sku": "A-100"}',
        "call_id": "call_1",
    },
    {"role": "assistant", "content": "確認します。少々お待ちください。"},
    {"type": "function_call_output", "call_id": "call_1", "output": '{"count": 3}'},
    {"role": "assistant", "content": "在庫は 3 個です。"},
]


def report_lines(label: str, records: list[dict[str, Any]] | Path) -> None:
    """2 つのゲートを並べて呼び、結果を出力する。"""
    validation = validate_dataset(records, method="sft")
    screening = screen_tool_roundtrips(records, method="sft")
    print(f"[{label}] validate_dataset: ok={validation.ok} / checked={validation.checked}")
    print(f"[{label}] screen_tool_roundtrips  : ok={screening.ok} / checked={screening.checked}")
    for violation in screening.violations:
        print(f"    line {violation.line}: {violation.reason}")


async def main() -> None:
    """持ち込みレコード・JSONL ファイル・会話ログ生成物の 3 経路を検査する。"""
    # 1. 持ち込みレコード列。メッセージ単位では全件合法だが並びに違反がある。
    report_lines("records", BROUGHT_IN)

    with tempfile.TemporaryDirectory() as tmp:
        # 2. JSONL ファイル。line は 1 始まりの物理行番号になる（列なら要素位置）。
        path = Path(tmp) / "train.jsonl"
        path.write_text(
            "\n".join(json.dumps(record, ensure_ascii=False) for record in BROUGHT_IN) + "\n",
            encoding="utf-8",
        )
        print()
        report_lines("jsonl", path)

        # 3. 会話ログからの生成物。生成は並びを理由に捨てないため、中断ランの文脈が
        # そのまま残り、screening が検出する。
        session = SQLiteSession("support-2026-09", Path(tmp) / "history.db")
        await session.add_items(INTERRUPTED_RUN)
        result = await dataset_from_session(session)
        print()
        print(f"[session] 生成レコード: {len(result.records)} 件 / skipped: {result.skipped}")
        report_lines("session", list(result.records))

    # 4. submit 前のゲートは raise_on_invalid=True で fail-closed にできる。
    print()
    try:
        screen_tool_roundtrips(BROUGHT_IN, method="sft", raise_on_invalid=True)
    except FineTuneError as exc:
        print(f"[gate] 投入を中止した: {exc.kind} / 違反 {len(exc.report.violations)} 件")

    # 5. 全件止めずに「投入できる分だけ進める」なら partition_dataset を使う。
    # 両ゲートを合成するため、構造の違反も並びの違反もまとめて不合格側へ落ちる。
    print()
    part = partition_dataset([*BROUGHT_IN, BAD_STRUCTURE], method="sft")
    print(f"[partition] ok={part.ok} / 検査 {part.checked} 件")
    print(f"    合格: {len(part.passed.records)} 件（passed.skipped={part.passed.skipped}）")
    for rejected in part.rejected:
        print(f"    不合格 line {rejected.line}:")
        for reason in rejected.reasons:
            print(f"      - {reason}")

    with tempfile.TemporaryDirectory() as tmp:
        # 合格側は DatasetBuildResult なので save() も submit_job(train=...) もそのまま通る。
        clean = Path(tmp) / "clean.jsonl"
        part.passed.save(clean)
        print(f"    合格分を書き出した: {len(clean.read_text(encoding='utf-8').splitlines())} 行")

    print()
    print("投入前に全件止めるなら 2 つのゲートを並べて呼ぶ:")
    print("    validate_dataset(records, method='sft', raise_on_invalid=True)")
    print("    screen_tool_roundtrips(records, method='sft', raise_on_invalid=True)")
    print("    job = await submit_job(client, train=records, model=..., method='sft')")
    print()
    print("投入できる分だけ進めるなら仕分ける:")
    print("    part = partition_dataset(records, method='sft')")
    print("    job = await submit_job(client, train=part.passed, model=..., method='sft')")


if __name__ == "__main__":
    asyncio.run(main())
