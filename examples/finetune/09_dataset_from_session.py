"""会話履歴（SDK Session）から SFT データセットを生成する例（API キー不要）。

運用中の会話履歴を学習データとして再利用する入口が `dataset_from_session` である。
`Session` の履歴を読み取り（読み取り専用・書込しない）、各 assistant 応答を正解、
それ以前の全ターンを文脈とするケースを累積ペアリングで生成し、`to_sft_dataset` と
同じ `DatasetBuildResult` を返す。結果はそのまま `submit_job(train=result)` /
`result.save(path)` へ渡せる。

本 example は SQLite の一時 Session に会話を書き込んでから生成するため、実 API 接続を
伴わない。生成規則の設計判断（累積ペアリング・破棄規則）は
`docs/adr/0033-session-dataset-pairing.md`、ツール往復の変換保持は
`docs/adr/0034-session-tool-roundtrip-and-dpo-draft.md`、併合条件とツール出力の型写像は
`docs/adr/0036-session-normalization-merge-and-tool-output-typing.md` を参照。

押さえておく挙動:

- **ツール往復は chat 形式へ変換されて文脈に残る**。function_call は `tool_calls` つきの
  assistant メッセージへ、function_call_output は role `"tool"` のメッセージへ 1:1 で写す
  （`arguments` は解釈・改変せず透過）。ツールを使った応答を、その根拠となる
  ツール入出力ごと学習できる。
- **ツール出力は文字列として文脈に載る**。`output` が dict / list なら JSON 文字列へ写す
  （中身は解釈・要約しない）。`output` が無ければ空文字になる。
- **それ以外の補助 item と生 role の system / developer / tool は破棄される**（reasoning /
  compaction / web_search_call 等・chat 形式に対応物が無いため）。対応する `call_id` の
  相手を欠く孤児 function_call / function_call_output も当該 item だけ落ちる。
  学習用の system は履歴からではなく `system=` で明示指定する（履歴内 system と競合しない）。
- **ケースになるのはテキスト応答の assistant ターンだけ**。変換で生まれた `tool_calls`
  つき assistant は文脈にのみ現れ、それ自体は学習ケースを生まない。
- **ツール往復の途中で切れるケースは生成されない**。`function_call` と対応する
  `function_call_output` の間に assistant テキストが入る履歴（承認待ちの通知を書き込む等）
  では、その assistant のケースの文脈が「応答のない `tool_calls`」で終わる。この並びは
  推論時の API が拒否するため生成せず `skipped` に載せる。
- **filter / transform は利用者供給**。品質選別・個人情報マスキングは lib が内蔵しない。
  filter で除外した件数は `skipped` に載る（全件除外でもエラーにならず空の結果が返る）。
- **transform は新しい dict を返すこと**。ケース間で turn dict が共有参照のため、
  in-place 変更は他ケースへ波及する。
- **compaction 済み履歴では畳まれたターンは学習ケースにならない**（compaction は履歴自体を
  置換する不可逆操作）。全ターンを学習データに使いたい Session では compaction を
  有効化しないか、発火前に生成する。
- **`tools=` を渡すとレコード直下の `"tools"` に載る**（DPO 経路は `input.tools` で透過位置が
  異なる）。会話ログからツール定義そのものは復元されない（`function_call` の `arguments` は
  transcript として残るが、スキーマは残らない）ため、利用者が定義を供給する。

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

# 実運用の会話ログを模した履歴。function_call 系 item（role なし）は chat 形式へ変換されて
# 文脈に残るため、推論時の Session をそのまま渡してよい。
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

# 会話で使われた lookup_faq に対応するツール定義（plain dict）。会話ログからツール定義
# そのものは復元されないため、利用者が供給する。
TOOLS: list[dict[str, object]] = [
    {
        "type": "function",
        "function": {
            "name": "lookup_faq",
            "description": "FAQ をキーワードで検索する",
            "parameters": {
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "required": ["q"],
            },
        },
    }
]


def drop_short_answers(case: dict[str, Any]) -> bool:
    """短すぎる応答のケースを学習対象から外す（品質選別は利用者責務の例）。"""
    return len(str(case["expected_output"])) >= 10


def mask_emails(case: dict[str, Any]) -> dict[str, Any]:
    """input 中のメールアドレスを伏せる（マスキングは利用者責務の例）。

    turn dict はケース間で共有参照のため in-place 変更せず、新しい dict を組んで返す。
    `input` には変換済みツールメッセージが混ざり、`tool_calls` つきの assistant は
    `content` キーを持たないため、`content` の有無を確かめてから書き換える
    （マスキングを書くときの定型）。ツール出力（role `"tool"` の `content`）も
    そのまま載るので、機密を含むなら同じ経路で除去する。
    """
    masked_input = [
        {**turn, "content": _mask(turn["content"])} if "content" in turn else turn
        for turn in case["input"]
    ]
    return {**case, "input": masked_input}


def _mask(text: Any) -> Any:
    """メールアドレスらしき文字列を置換する（example 用の素朴な実装）。

    ツール出力の `content` は文字列とは限らない（lib は非改変で載せる）ため、
    文字列以外はそのまま返す。
    """
    if not isinstance(text, str):
        return text
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
            tools=TOOLS,
        )

    print(f"生成レコード: {len(result.records)} 件 / skipped: {result.skipped}")
    for record in result.records:
        print(json.dumps(record, ensure_ascii=False, indent=2))
    print("tools =", result.records[0]["tools"])

    print()
    print("この結果はそのまま投入できる:")
    print("    job = await submit_job(client, train=result, model=..., method='sft')")


if __name__ == "__main__":
    asyncio.run(main())
