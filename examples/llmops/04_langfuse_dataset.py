"""Langfuse Datasets を register → fetch → use で扱う例（Scores/Traces + push 専用 Prompt Mgmt）。

Datasets は Langfuse が source（mirror push しない）:
  1. `register_dataset(cfg, name, cases)` で一度きり登録（冪等・再実行可）。
  2. `load_dataset(cfg, name)` で fetch して使う。
  3. `evaluate(..., langfuse=cfg)` は既存 dataset item に run を link するだけ（毎回 upsert なし）。

`evaluate(langfuse=...)` の送信内容:
  - Tracing / Scores: 常時（評価対象の入出力・判定・観点別スコア・統合 verdict）。
  - Datasets（`dataset_name` 設定時）: 各ケースの trace を既存 dataset item × dataset run に link。
    同一データセット上で run/variant を横並び比較できる（A/B・回帰比較）。
  - push 専用 Prompt Management（`prompt_name` 設定時）: 評価対象プロンプトを register/upsert し
    評価 trace（generation 種別）を prompt version にリンク。取得/配信はしない（SoT=PromptStore）。

Langfuse は利用者が用意した稼働中インスタンス（self-host / cloud）へ送信するだけ（oai-agentspec は
サーバを立てない・サーバ側評価はさせない）。未設定/送信失敗でもローカル verdict は返る。

必要な環境変数（.env 等）:
    AZURE_OPENAI_*          採点・実行用モデル（examples/_shared/_azure.py 参照）
    LANGFUSE_PUBLIC_KEY     Langfuse public key
    LANGFUSE_SECRET_KEY     Langfuse secret key
    LANGFUSE_HOST           Langfuse 接続先（self-host の URL / cloud。未設定で SDK 既定）

実行:
    uv run python examples/llmops/04_langfuse_dataset.py

導入: pip install 'oai-agentspec[llmops,llmops-langfuse]'（採点 + 観測）。
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from oai_agentspec import AgentSpec
from oai_agentspec.runtime.llmops import (
    Conciseness,
    EvalCase,
    Faithfulness,
    LangfuseConfig,
    Relevance,
    evaluate,
    load_dataset,
    register_dataset,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
from _azure import azure_model, load_env  # noqa: E402


def _langfuse_config() -> LangfuseConfig:
    """環境変数から Langfuse 設定を組む（認証情報は利用者から受領・env 境界は利用側）。

    `.env` は事前に `load_env()` で読み込んでおく（main 冒頭で実施）。
    """
    try:
        public_key = os.environ["LANGFUSE_PUBLIC_KEY"]
        secret_key = os.environ["LANGFUSE_SECRET_KEY"]
    except KeyError as exc:
        raise SystemExit(
            "LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY が未設定です。"
            ".env に Langfuse インスタンス（self-host / cloud）の認証情報を設定してください"
            "（.env.example の Langfuse セクション参照）。"
        ) from exc
    return LangfuseConfig(
        public_key=public_key,
        secret_key=secret_key,
        host=os.environ.get("LANGFUSE_HOST"),
        dataset_name="oai-agentspec-llmops-demo",
        run_name="example-run-01",
        prompt_name="jp-assistant",
        prompt_label="example",
    )


async def main() -> None:
    # .env を明示的に読み込む（AZURE_OPENAI_* と LANGFUSE_* の両方をこの後利用）。
    load_env()

    target = AgentSpec(
        name="jp-assistant",
        instructions="あなたは事実に基づき簡潔に答える日本語アシスタントです。",
        model=azure_model(),
    )

    langfuse = _langfuse_config()
    dataset_name = "oai-agentspec-llmops-demo"

    # EvalCase に id を付けると Langfuse dataset item の安定キーになり、run 横断で対応づく。
    # expected_output（正解文）は item.expected_output へ、reference_context は item.metadata へ
    # 反映される。提供時は G-Eval が EXPECTED_OUTPUT として参照する（評価では必須でない）。
    seed_cases = [
        EvalCase("日本の首都はどこですか?", id="capital", expected_output="日本の首都は東京です。"),
        EvalCase(
            "富士山の標高は?",
            id="fuji-altitude",
            reference_context=["富士山の標高は 3776 メートルである。"],
            expected_output="富士山の標高は 3776 メートルです。",
        ),
    ]

    # (1) 登録は一度だけ（冪等なので再実行可・以降は fetch して使う）。実運用では別スクリプトや
    #     Langfuse UI で一度行い、evaluate 側は load → use に徹してよい。
    register_dataset(langfuse, dataset_name, seed_cases)

    # (2) Langfuse が source。登録済み dataset を fetch して評価に使う（毎回 push しない）。
    dataset = load_dataset(langfuse, dataset_name)

    # 観点はオブジェクトで宣言（G-Eval rubric は利用者提供）。reference_context 無のケースでは
    # Faithfulness が自動的に not_applicable になる。
    criteria = [
        Relevance(),
        Conciseness(rubric="回答が冗長でなく簡潔で要点を押さえているか。"),
        Faithfulness(),
    ]

    # (3) evaluate は既存 dataset item に run を link するだけ（dataset_name 設定時）。
    result = await evaluate(
        target, dataset, judge=azure_model(), criteria=criteria, langfuse=langfuse
    )

    print(f"\n=== verdict: {result.verdict.value} ===")
    for i, case in enumerate(result.cases):
        statuses = ", ".join(f"{c.criterion}={c.status.value}" for c in case.criteria)
        print(f"[case {i}] {case.case_input!r}: {statuses}")
    print(
        "\nLangfuse へ送信しました（送信は best-effort）。Langfuse UI で dataset "
        "'oai-agentspec-llmops-demo' の run 'example-run-01' とスコア、Prompts の "
        "'jp-assistant' を確認してください。"
    )


if __name__ == "__main__":
    asyncio.run(main())
