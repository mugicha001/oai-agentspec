"""会話履歴（SDK Session）から SFT データセットを生成する例（API キー不要）。

運用中の会話履歴を学習データとして再利用する入口が `dataset_from_session` である。
`Session` の履歴を読み取り（読み取り専用・書込しない）、各 assistant 応答を正解、
それ以前の全ターンを文脈とするケースを累積ペアリングで生成し、`to_sft_dataset` と
同じ `DatasetBuildResult` を返す。結果はそのまま `submit_job(train=result)` /
`result.save(path)` へ渡せる。

本 example は SQLite の一時 Session に会話を書き込んでから生成するため、実 API 接続を
伴わない。生成規則の設計判断（累積ペアリング・破棄規則）は
`docs/adr/0033-session-dataset-pairing.md` を参照。

押さえておく挙動:

- **role なし item（function_call 等）と system / developer / tool の item は破棄される**。
  学習用の system は履歴からではなく `system=` で明示指定する（履歴内 system と競合しない）。
- **filter / transform は利用者供給**。品質選別・個人情報マスキングは lib が内蔵しない。
  filter で除外した件数は `skipped` に載る（全件除外でもエラーにならず空の結果が返る）。
- **transform は新しい dict を返すこと**。ケース間で turn dict が共有参照のため、
  in-place 変更は他ケースへ波及する。
- **compaction 済み履歴では畳まれたターンは学習ケースにならない**（compaction は履歴自体を
  置換する不可逆操作）。全ターンを学習データに使いたい Session では compaction を
  有効化しないか、発火前に生成する。

実行:
    uv run python examples/finetune/09_dataset_from_session.py

導入: pip install 'oai-agentspec[finetune]'
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any

from agents import SQLiteSession

from oai_agentspec.runtime.finetune import dataset_from_session

# 実運用の会話ログを模した履歴。function_call 系 item（role なし）が混ざっていても
# 生成時に破棄されるため、推論時の Session をそのまま渡してよい。
CONVERSATION: list[dict[str, Any]] = [
    {"role": "user", "content": "会員登録の手順を教えて"},
    {
        "type": "function_call",
        "name": "lookup_faq",
        "arguments": '{"q": "register"}',
        "call_id": "call_1",
    },
    {"type": "function_call_output", "call_id": "call_1", "output": '{"answer": "..."}'},
    {"role": "assistant", "content": "手順は 3 ステップです: メール登録、確認、プロフィール入力。"},
    {"role": "user", "content": "料金は? 私のメールは taro@example.com です"},
    {"role": "assistant", "content": "月額 500 円です。"},
]


def drop_short_answers(case: dict[str, Any]) -> bool:
    """短すぎる応答のケースを学習対象から外す（品質選別は利用者責務の例）。"""
    return len(str(case["expected_output"])) >= 10


def mask_emails(case: dict[str, Any]) -> dict[str, Any]:
    """input 中のメールアドレスを伏せる（マスキングは利用者責務の例）。

    turn dict はケース間で共有参照のため in-place 変更せず、新しい dict を組んで返す。
    """
    masked_input = [{**turn, "content": _mask(turn["content"])} for turn in case["input"]]
    return {**case, "input": masked_input}


def _mask(text: str) -> str:
    """メールアドレスらしき文字列を置換する（example 用の素朴な実装）。"""
    words = [w if "@" not in w else "<email>" for w in text.split(" ")]
    return " ".join(words)


async def main() -> None:
    """一時 Session へ会話を書き込み、フィルタ・マスキング付きで生成する。"""
    with tempfile.TemporaryDirectory() as tmp:
        session = SQLiteSession("support-2026-08", Path(tmp) / "history.db")
        await session.add_items(CONVERSATION)

        result = await dataset_from_session(
            session,
            system="サポート担当として簡潔に答える",
            case_filter=drop_short_answers,
            case_transform=mask_emails,
        )

    print(f"生成レコード: {len(result.records)} 件 / skipped: {result.skipped}")
    for record in result.records:
        print(json.dumps(record, ensure_ascii=False, indent=2))

    print()
    print("この結果はそのまま投入できる:")
    print("    job = await submit_job(client, train=result, model=..., method='sft')")


if __name__ == "__main__":
    asyncio.run(main())
