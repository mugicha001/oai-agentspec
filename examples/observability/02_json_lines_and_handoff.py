"""1 行 JSON 出力とハンドオフ時のスパン種別を確認する例（実 API）。

`01_trace_and_logs.py` との差分は次の 2 点。

1. `console_json_lines=True` でログのコンソール出力を 1 行 JSON（JSON Lines）にする。既定の
   整形済み出力は 1 レコードが 20 行以上に分かれるため、コンテナの標準出力を「1 行 = 1 レコード」
   で取り込むログ収集基盤では 1 つのログが複数レコードへ分割されてしまう。この構成で使う。
2. ハンドオフを含む 2 エージェント構成にして、スパンに付く種別
   （`gen_ai.operation.name`）が `invoke_agent` / `execute_tool` / `chat` / `chain` と
   複数現れる様子を観測する。ハンドオフは SDK 上ツール呼び出しとして表現される。

重要（ログとトレースで制御口が違う）:
    本 example の `console_json_lines` / `otlp_enabled` は**ログ側にしか効かない**。トレースの
    エクスポートは Agent 365 の構成関数が担っており、整形方法を指定する引数が公開されていない
    ため、**スパンのコンソール出力は 1 行化できず整形済みのまま**になる。実行すると同じ標準出力に
    「1 行のログ」と「複数行のスパン」が混在するのはこのため。

    トレースを外部へ送る場合は Agent 365 側の環境変数を使う（本ライブラリの設定ではない）:

        ENABLE_OTLP_EXPORTER=true OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 \
            uv run python examples/observability/02_json_lines_and_handoff.py

    標準出力の収集でトレースも扱いたい場合は、この OTLP 経路でコレクタへ送り、保存形式や
    ローテーションはコレクタ側で決めるのが素直（ログも `otlp_enabled=True` で同じコレクタへ
    送れる）。

出力先はトレース・ログとも標準出力。ログだけをファイルへ残したい場合はシェルでリダイレクトする
（1 行 JSON にしてあるためそのまま JSON Lines ファイルになる。ただし上記のとおりスパンも同じ
標準出力に混ざるため、ログだけを取り出すには行頭で選別する）:

    uv run python examples/observability/02_json_lines_and_handoff.py \
        | grep '^{"body"' > logs/app.jsonl

Azure OpenAI の環境変数（AZURE_OPENAI_* 。examples/_shared/_azure.py 参照）を設定して実行:
    uv run python examples/observability/02_json_lines_and_handoff.py

導入: pip install 'oai-agentspec[observability]'
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

from agents import Runner, set_tracing_disabled

from oai_agentspec import AgentRegistry, AgentSpec, HandoffGraph
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


async def main() -> None:
    model = azure_model()
    # `azure_model()` が立てた無効化を戻す（トレースの送出そのものが目的のため）。
    set_tracing_disabled(False)

    enable_agent365_tracing(
        Agent365TracingConfig(
            service_name="oai-agentspec-example",
            service_namespace="examples",
        )
    )
    enable_otel_logging(
        OtelLoggingConfig(
            service_name="oai-agentspec-example",
            # 1 レコード = 1 行にする（ログ収集基盤へそのまま取り込める形）。
            console_json_lines=True,
            # 外部の OTLP コレクタへも送る場合に有効化する（接続先は OpenTelemetry 標準の
            # 環境変数 OTEL_EXPORTER_OTLP_ENDPOINT 等で解決される）。コンソール出力は
            # 置換されず併用になる。**これはログ側だけの設定で、トレースには効かない**
            # （トレースは環境変数 ENABLE_OTLP_EXPORTER を使う。冒頭 docstring 参照）。
            # otlp_enabled=True,
        )
    )
    logging.getLogger().setLevel(logging.INFO)

    registry = AgentRegistry()
    for name, instructions in (
        ("triage", "You route requests. Hand off billing questions to the billing agent."),
        ("billing", "You answer billing questions in one short sentence."),
    ):
        registry.register(AgentSpec(name=name, instructions=instructions, model=model))

    graph = HandoffGraph(entry="triage")
    graph.edge("triage", "billing", description="請求に関する問い合わせは billing へ")
    graph.apply(registry)
    registry.validate()

    logger.info("handoff run starting")
    result = await Runner.run(registry.get("triage"), "請求書の発行日を教えて")
    logger.info("handoff run finished")

    print(f"\n=== final output ===\n{result.final_output}")
    print(
        "\n=== 確認ポイント ===\n"
        "- ログレコードが 1 行 JSON で出ている（console_json_lines=True の効果）\n"
        "- 一方スパンは整形済みの複数行のまま。console_json_lines はログ側だけの設定で、\n"
        "  トレースの出力形式は Agent 365 側が握っており本ライブラリからは変えられない\n"
        "- そのため標準出力にはこの 2 形式が混在する。ログだけを取り出すなら行頭で選別し、\n"
        "  トレースも収集するなら ENABLE_OTLP_EXPORTER で OTLP コレクタへ送る\n"
        "- スパンの gen_ai.operation.name に種別が付く:\n"
        "    invoke_agent  = エージェント起動（ハンドオフで 2 回現れる）\n"
        "    execute_tool  = ツール実行。ハンドオフ自体もここに含まれる\n"
        "    chat          = モデル呼び出し\n"
        "    chain         = ワークフロー / ターン等の入れ物\n"
        "- OTel 標準の kind は全て INTERNAL なので、分類には operation.name を使う"
    )


if __name__ == "__main__":
    asyncio.run(main())
