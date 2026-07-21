"""意図予測に会話履歴を渡す例（`agents.SQLiteSession` 経由・実 API）。

`IntentQuery.history` に `agents.SQLiteSession` を渡すと、`DefaultContextBuilder` が
直近履歴を `IntentContext.history_items` として抽出する。履歴は SDK が multi-turn
として LLM へ送るため、`prompt(context)` は現在発話のみを組み立てる。分類実行時の
合成プロンプトも表示する。

Azure OpenAI の環境変数（examples/_shared/_azure.py 参照）を設定して実行:
    uv run python examples/intent/02_with_session_history.py
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

from agents import SQLiteSession

from oai_agentspec.runtime.intent import (
    IntentCategory,
    IntentContext,
    IntentPolicy,
    IntentQuery,
    intent_classifier_from_model,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
from _timing import stopwatch  # noqa: E402

from _azure import azure_model  # noqa: E402


def build_prompt(context: IntentContext) -> str:
    """現在発話のみを組み立てる。履歴（`context.history_items`）は SDK が multi-turn
    として LLM に届けるため、prompt callable では扱わない。history_items はライブラリ側で
    sanitize されないため、内容を prompt に混ぜる利用では利用側で fenced block を推奨
    （README 参照）。

    utterance が空（履歴のみモード）のときは空文字を返す。adapter は空の user_content に
    対して user turn を追加しないため、履歴だけが LLM に送られる。固定接頭辞を無条件に
    付けると空発話の指示文 turn が送られてしまう点に注意。
    """
    if not context.utterance:
        return ""
    return f"次の発話をカテゴリに分類してください:\n{context.utterance}"


async def main() -> None:
    policy = IntentPolicy(
        categories=(
            IntentCategory(name="follow_up", description="直前の話題への追加質問"),
            IntentCategory(name="new_topic", description="新しい話題への切り替え"),
            IntentCategory(name="confirm", description="確認・同意"),
        ),
    )

    with tempfile.TemporaryDirectory() as tmp:
        session = SQLiteSession(session_id="demo-intent", db_path=str(Path(tmp) / "conv.db"))
        # 履歴を仕込む（実利用では ConversationService や Runner が積む）。
        await session.add_items(
            [
                {"role": "user", "content": "御社の料金プランは何種類ありますか?"},
                {"role": "assistant", "content": "3 種類あります。Basic / Pro / Enterprise です。"},
            ]
        )

        classifier = intent_classifier_from_model(
            model=azure_model(),
            prompt=build_prompt,
            policy=policy,
            history_limit=10,
        )

        query = IntentQuery(utterance="Pro プランの詳細を教えてください", history=session)

        context = await classifier.context_builder.build(query)
        print("=" * 60)
        print("[SYSTEM] policy.render_prompt()")
        print("=" * 60)
        print(policy.render_prompt())
        print()
        print("=" * 60)
        print("[USER] prompt(context) -- 現在発話のみ")
        print("=" * 60)
        print(build_prompt(context))
        print()
        print("=" * 60)
        print("[HISTORY] context.history_items -- SDK が multi-turn として送る履歴")
        print("=" * 60)
        for item in context.history_items:
            print(f"  {item}")
        print()

        with stopwatch("classify (履歴 + 現在発話)"):
            prediction = await classifier.classify(query)

        print("=" * 60)
        print("[RESULT] 直前の話題（料金プラン）を踏まえた分類")
        print("=" * 60)
        for i, c in enumerate(prediction.candidates):
            print(f"  #{i + 1} text={c.text} level={c.level.value}")

        # --- 履歴のみモード: utterance を省略し「ここまでの会話」を分類する ---
        # utterance 既定は "" で、その場合 user turn は追加されず履歴だけが LLM に送られる。
        history_only = IntentQuery(history=session)
        with stopwatch("classify (履歴のみ)"):
            prediction2 = await classifier.classify(history_only)

        print()
        print("=" * 60)
        print("[RESULT] 履歴のみモード（utterance 省略・会話全体から分類）")
        print("=" * 60)
        for i, c in enumerate(prediction2.candidates):
            print(f"  #{i + 1} text={c.text} level={c.level.value}")


if __name__ == "__main__":
    asyncio.run(main())
