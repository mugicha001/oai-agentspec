"""Agent 365 オブザーバビリティ連携でトレースと相関ログをコンソールに出す最小例（実 API）。

`enable_agent365_tracing` で SDK トレーシングへ Agent 365 の計装を適用し、
`enable_otel_logging` で標準 `logging` を OpenTelemetry Logs として送出する。どちらも
エクスポート先を指定しなければコンソールへ出力されるため、Agent 365 の実サービスや認証情報
なしで動作を確認できる。エージェント実行中に出したログには実行中スパンの trace_id / span_id が
付与されるので、span 出力と突き合わせて相関を確認できる。

注意（examples/_shared/_azure.py との差分）:
    `azure_model()` は内部で `set_tracing_disabled(True)` を呼ぶ。トレーシングが無効なままだと
    Agent 365 へは何も送られないため、本 example ではモデル構築の**後**に明示的に
    `set_tracing_disabled(False)` で有効化し直す。有効化 API は `set_tracing_disabled(True)` に
    よる無効化を検知して `RuntimeWarning` を出すが、環境変数 `OPENAI_AGENTS_DISABLE_TRACING` に
    よる無効化は検知できない（無警告でも送信されているとは限らない）。

Azure OpenAI の環境変数（AZURE_OPENAI_* 。examples/_shared/_azure.py 参照）を設定して実行:
    uv run python examples/observability/01_trace_and_logs.py

導入: pip install 'oai-agentspec[observability]'
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

from agents import Runner, set_tracing_disabled

from oai_agentspec import AgentRegistry, AgentSpec, function_tool
from oai_agentspec.runtime.observability import (
    Agent365TracingConfig,
    OtelLoggingConfig,
    enable_agent365_tracing,
    enable_otel_logging,
)

# examples/ 共有ヘルパ _azure を解決するため _shared を import パスへ。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
from _azure import azure_model  # noqa: E402

logger = logging.getLogger("examples.observability")


@function_tool
def current_greeting_style() -> str:
    """挨拶のトーンを返す（ツール実行スパンの内側でログを出すためのダミーツール）。"""
    # ツール実行中は SDK のスパンが有効なので、このログには trace_id / span_id が付与される。
    logger.info("tool called inside an active span")
    return "casual"


async def main() -> None:
    model = azure_model()
    # `azure_model()` が立てた無効化を戻す（本 example はトレースの送出そのものが目的）。
    set_tracing_disabled(False)

    enable_agent365_tracing(
        Agent365TracingConfig(
            service_name="oai-agentspec-example",
            service_namespace="examples",
        )
    )
    # root logger へハンドラが付くため、アプリ全体のログが OTel Logs として送出される。
    # root のレベルは変更しないので、INFO を拾いたい場合は利用者側で設定する。
    enable_otel_logging(OtelLoggingConfig(service_name="oai-agentspec-example"))
    logging.getLogger().setLevel(logging.INFO)

    registry = AgentRegistry()
    registry.register(
        AgentSpec(
            name="greeter",
            instructions=(
                "You are a concise assistant. Always call the current_greeting_style tool "
                "first, then answer in one short sentence."
            ),
            model=model,
            tools=[current_greeting_style],
        )
    )

    # スパン外のログ（アクティブなスパンが無いので trace_id / span_id は付与されない）。
    logger.info("agent run starting")

    result = await Runner.run(registry.get("greeter"), "Say hello in Japanese.")

    # スパン外のログ（実行が終わっているため相関 ID は付かない）。
    logger.info("agent run finished")

    print(f"\n=== final output ===\n{result.final_output}")
    print(
        "\n=== 確認ポイント ===\n"
        "- 上に流れた span（invoke_agent / execute_tool / chat）が Agent 365 の計装による出力\n"
        '- ログ "tool called inside an active span" には trace_id / span_id が入る\n'
        '  （ツール実行スパンの内側で出したため）。一方 "agent run starting" / "finished" は\n'
        "  スパン外なので全ゼロになり、偽の相関 ID は付かない\n"
        "- エクスポート先を切り替える場合は Agent365TracingConfig の exporter_options /\n"
        "  token_resolver、ログ側は OtelLoggingConfig(otlp_enabled=True) を使う"
    )


if __name__ == "__main__":
    asyncio.run(main())
