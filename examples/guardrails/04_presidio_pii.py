"""外部検知器（Presidio PII）guardrail（出力・OWASP LLM02）の最小例（実 API + 任意依存）。

`external_detector_guardrail` は重い専門検知（PII / モデレーション等）を**外部 DI** で薄く包む。
本ライブラリは検知本体を同梱しないため、Presidio は本 example 内で遅延 import し、未導入時は
導入方法を案内して終了する（lib の依存には足さない）。

ここでは Presidio Analyzer で出力テキストから PII（メール / 電話 / クレジットカード等）を検出し、
1 件でも見つかれば `Detection(triggered=True)` を返す検知 callable を作って guardrail に DI する。
trip すると SDK `Runner` が `OutputGuardrailTripwireTriggered` を送出して応答を止める。

Azure OpenAI の環境変数（AZURE_OPENAI_* 。examples/_shared/_azure.py 参照）を設定して実行:
    uv run python examples/guardrails/04_presidio_pii.py

導入:
    pip install 'oai-agentspec[guardrails]'                 # 本機能（依存ゼロ）
    pip install presidio-analyzer presidio-anonymizer         # 任意の外部検知器（PII）
    python -m spacy download en_core_web_lg                   # Presidio の既定 NLP モデル
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

from agents import OutputGuardrailTripwireTriggered, Runner

from oai_agentspec import AgentRegistry, AgentSpec
from oai_agentspec.runtime.guardrails import Detection, external_detector_guardrail

# examples/ 共有ヘルパ _azure を解決するため _shared を import パスへ。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
from _azure import azure_model  # noqa: E402


def _load_presidio() -> Any:
    """Presidio Analyzer を遅延 import する（未導入時は導入方法を案内して終了）。

    Returns:
        構築済み `AnalyzerEngine` インスタンス。

    Raises:
        SystemExit: presidio-analyzer が未導入の場合（案内を表示して終了）。
    """
    try:
        from presidio_analyzer import AnalyzerEngine
    except ImportError:
        print(
            "Presidio が未導入です。次でインストールしてください:\n"
            "  pip install presidio-analyzer presidio-anonymizer\n"
            "  python -m spacy download en_core_web_lg"
        )
        raise SystemExit(1) from None
    return AnalyzerEngine()


def _make_pii_detector(analyzer: Any) -> Any:
    """Presidio Analyzer を oai-agentspec の検知 callable（テキスト -> Detection）へ包む。

    Args:
        analyzer: 構築済み Presidio `AnalyzerEngine`。

    Returns:
        `Callable[[str], Detection]`。PII を 1 件でも検出したら triggered=True。
    """

    def detect(text: str) -> Detection:
        # 検出する PII 種別は利用者要件に応じて調整する（ここでは代表的な数種）。
        results = analyzer.analyze(
            text=text,
            language="en",
            entities=["EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD", "US_SSN"],
        )
        if results:
            kinds = sorted({r.entity_type for r in results})
            return Detection(triggered=True, reason="PII detected", info={"entities": kinds})
        return Detection(triggered=False)

    return detect


def build_agent(detector: Any) -> AgentSpec:
    """Presidio PII 検知を DI した出力 guardrail を装着したエージェント spec を組む。

    Args:
        detector: PII 検知 callable（`Callable[[str], Detection]`）。

    Returns:
        `output_guardrails` フィールドへ guardrail を渡した `AgentSpec`。
    """
    guardrail = external_detector_guardrail(detector, on="output")
    return AgentSpec(
        name="support-bot",
        instructions=(
            "You are a support assistant. Answer the user's request. "
            "If asked to echo contact details, include them verbatim."
        ),
        model=azure_model(),
        output_guardrails=[guardrail],
    )


async def _ask(registry: AgentRegistry, text: str) -> None:
    """1 入力を Runner で実行し、出力に PII が含まれて trip したかを表示する。

    Args:
        registry: spec を登録済みの `AgentRegistry`。
        text: ユーザー入力。
    """
    agent = registry.get("support-bot")
    try:
        result = await Runner.run(agent, input=text)
        print(f"[pass] input={text!r}\n       output: {result.final_output[:80]}")
    except OutputGuardrailTripwireTriggered:
        print(f"[trip] input={text!r}  -> 出力に PII を検知して応答を停止しました")


async def main() -> None:
    analyzer = _load_presidio()
    detector = _make_pii_detector(analyzer)

    registry = AgentRegistry()
    registry.register(build_agent(detector))
    registry.validate()

    # PII を含まない応答は通過する。
    await _ask(registry, "How do I reset my password?")

    # 連絡先をそのまま返させる入力。出力に PII が含まれれば trip する。
    await _ask(
        registry,
        "Please confirm my contact on file: john.doe@example.com and +1-202-555-0143.",
    )


if __name__ == "__main__":
    asyncio.run(main())
