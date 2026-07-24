"""triage <-> support の相互 handoff（循環）を宣言し RealtimeRunner でセッションを実行する例。

宣言（RealtimeAgentSpec / RealtimeAgentRegistry）と実行時 Config（RealtimeRunConfig）の
責務分担を示す:

  - 宣言側（本ライブラリ）: エージェントの静的な構造（instructions / tools / handoffs 等）を
    RealtimeAgentSpec で宣言し、registry が RealtimeAgent を構築する。
  - 実行側（SDK / 利用者）: model_name / voice / modalities 等の実行時 Config は
    RealtimeAgentSpec が型として持たない。セッション開始時に `RealtimeRunner` の
    `config=RealtimeRunConfig(model_settings=...)` として利用者が渡す。

音声 I/O は作り込まず、テキスト送信とテキスト系イベントの表示のみの最小構成。

接続先は他の examples と同様 Azure OpenAI を優先する（`AZURE_OPENAI_ENDPOINT` /
`AZURE_OPENAI_API_KEY` / `AZURE_OPENAI_REALTIME_DEPLOYMENT` が設定されていれば
`RealtimeModelConfig` の `url` + `api-key` ヘッダーで Azure の Realtime WebSocket へ接続）。
Azure 未設定の場合は `OPENAI_API_KEY` で api.openai.com へフォールバックする:

    uv run python examples/realtime/handoff_session.py

環境変数の取り扱いは examples 共通の補助（examples/_shared/_azure.py の load_env）に従い、
リポジトリ直下の .env からも読み込む。ライブラリ本体は env 非依存だが、examples は直接実行の
利便のため .env を読む。
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from agents.realtime import RealtimeRunner

from oai_agentspec.realtime import (
    RealtimeAgentRegistry,
    RealtimeAgentSpec,
    RealtimeHandoffGraph,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
from _connection import build_model_config, require_credentials, scrub  # noqa: E402

from _azure import load_env  # noqa: E402

# 実行時 Config（宣言ではなく実行時に利用者が渡す）。
# RealtimeAgentSpec はこれらを非対応フィールドとして型に持たない。
MODEL_SETTINGS = {
    "model_name": "gpt-4o-realtime-preview",
    "voice": "alloy",
    # テキストのみで受け取り、音声 I/O は本例では扱わない。
    "modalities": ["text"],
}


def build_registry() -> RealtimeAgentRegistry:
    """triage <-> support の相互 handoff（循環）を宣言・登録して registry を返す。

    Returns:
        validate 済みの RealtimeAgentRegistry（entry は triage）。
    """
    # spec はエージェントの中身のみ宣言する（トポロジはグラフ側の責務）。
    specs = [
        RealtimeAgentSpec(
            name="triage",
            instructions="受付担当。技術的な問い合わせはサポート担当へ引き継ぐ。",
            handoff_description="最初の受付・振り分け担当。",
        ),
        RealtimeAgentSpec(
            name="support",
            instructions=(
                "製品の技術的な問い合わせに答えるサポート担当。"
                "解決したら、または技術以外の話題になったら受付担当へ戻す。"
            ),
            handoff_description="技術サポート担当。",
        ),
    ]

    # 相互 handoff（循環）をグラフ DSL で宣言し spec 群へ一括反映する
    # （spec.handoffs 直接宣言と同一の結線になる）。
    graph = RealtimeHandoffGraph(entry="triage")
    graph.edge("triage", "support", tool_description="技術的な問い合わせを引き継ぐ")
    graph.edge("support", "triage", tool_description="解決したら受付へ戻す")
    graph.apply(specs)

    registry = RealtimeAgentRegistry()
    for spec in specs:
        registry.register(spec)
    registry.validate()
    return registry


async def main() -> None:
    load_env()

    require_credentials()

    registry = build_registry()
    entry = registry.get("triage")

    # 実行時 Config は宣言側が関知しない: model_settings（model_name / voice 等）は
    # RealtimeRunner 構築時、接続先（model_config）はセッション開始（run()）時に渡す。
    runner = RealtimeRunner(
        entry,
        config={"model_settings": MODEL_SETTINGS},  # type: ignore[typeddict-item]
    )

    model_config = build_model_config()
    print(f"接続先: {'Azure OpenAI' if model_config else 'OpenAI (api.openai.com)'}")
    async with await runner.run(model_config=model_config) as session:
        await session.send_message("ログインできません。エラーコードは E42 です。")
        handoff_seen = False
        try:
            async with asyncio.timeout(60):
                async for event in session:
                    # 音声デルタ以外のイベント種別を表示（観察用の最小構成）。
                    if event.type == "handoff":
                        handoff_seen = True
                        print(f"[handoff] {event.from_agent.name} -> {event.to_agent.name}")
                    elif event.type == "history_updated":
                        print(f"[history] items={len(event.history)}")
                    elif event.type == "error":
                        print(f"[error] {scrub(str(event.error))[:200]}")
                        break
                    elif event.type != "raw_model_event":
                        print(f"[event] {event.type}")
                    # 1 ターンの応答が終わったら終了する（handoff の有無はモデルの判断に
                    # 依存し非決定的なため、完了条件にはしない）。
                    if event.type == "agent_end":
                        break
        except TimeoutError:
            print("[timeout] 60 秒以内に完了しませんでした（イベントは上記まで）")
        print(f"完了: handoff 発生 = {handoff_seen}")


if __name__ == "__main__":
    asyncio.run(main())
