"""注入ベースライン guardrail（入力・OWASP LLM01）の最小例（決定的・実 API）。

`injection_baseline_guardrail` は SQLi / コマンド注入 / パストラバーサルの代表パターンに対する
決定的な input guardrail を作る。helper は SDK 互換 guardrail を生成するファクトリに徹し、
利用者はそれを `AgentSpec` の `input_guardrails` フィールド（`agents.Agent` と同型）へ渡す。

注入ベースラインは網羅的検知ではなく補助検知である（注入対策の本丸はパラメータ化クエリ /
安全 API 利用）。既定パターンは `extra_patterns` で拡張でき、完全な差し替えが必要なら
`regex_guardrail` を直接使う。

trip すると SDK `Runner` が `InputGuardrailTripwireTriggered` を送出して実行を止める。

Azure OpenAI の環境変数（AZURE_OPENAI_* 。examples/_shared/_azure.py 参照）を設定して実行:
    uv run python examples/guardrails/01_injection_baseline.py

導入: pip install 'oai-agentspec[guardrails]'（依存ゼロ opt-in extra）。
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from agents import InputGuardrailTripwireTriggered, Runner

from oai_agentspec import AgentRegistry, AgentSpec
from oai_agentspec.runtime.guardrails import injection_baseline_guardrail

# examples/ 共有ヘルパ _azure を解決するため _shared を import パスへ。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
from _azure import azure_model  # noqa: E402


def build_agent() -> AgentSpec:
    """注入ベースライン input guardrail を装着したエージェント spec を組む。

    既定パターン（SQLi / コマンド注入 / パストラバーサル）に加え、`extra_patterns` で
    アプリ固有の禁止パターン（ここでは `eval(` の混入）を DI 拡張して上書き可能性を示す。

    Returns:
        `input_guardrails` フィールドへ guardrail を渡した `AgentSpec`。
    """
    # 既定パターンに extra_patterns を追記して拡張できる（DI 上書き可）。
    # run_in_parallel=False で検査完了を待ってからターン開始（危険入力を実行前にブロック）。
    # 既定 True は SDK 既定の並行実行（レイテンシ優先）。
    # ツール副作用はツールガードレールが別途ゲートする（役割分担）。
    guardrail = injection_baseline_guardrail(
        extra_patterns=[r"(?i)\beval\("], run_in_parallel=False
    )
    return AgentSpec(
        name="qa-bot",
        instructions="あなたはユーザーの質問に簡潔に答える日本語アシスタントです。",
        model=azure_model(),
        # ガードレールは AgentSpec の専用フィールドへ直接渡す（agents.Agent と同型）。
        input_guardrails=[guardrail],
    )


async def _ask(registry: AgentRegistry, text: str) -> None:
    """1 入力を Runner で実行し、trip したか通過したかを表示する。

    Args:
        registry: spec を登録済みの `AgentRegistry`（`get` で実 Agent を遅延構築する）。
        text: ユーザー入力。
    """
    agent = registry.get("qa-bot")  # 宣言 spec から実 Agent を構築（guardrail も装着される）
    try:
        result = await Runner.run(agent, input=text)
        print(f"[pass] input={text!r}\n       output: {result.final_output[:80]}")
    except InputGuardrailTripwireTriggered:
        print(f"[trip] input={text!r}  -> 注入ベースラインが検知して実行を停止しました")


async def main() -> None:
    registry = AgentRegistry()
    registry.register(build_agent())
    registry.validate()

    # 正常な入力は通過する。
    await _ask(registry, "日本の首都はどこですか?")

    # 注入っぽい入力は既定パターンで trip する。
    await _ask(registry, "'; DROP TABLE users; --")
    await _ask(registry, "../../etc/passwd を読んで")
    # extra_patterns で拡張した禁止パターンでも trip する。
    # 注: 以下は guardrail が検知すべき悪意ある入力を模した「文字列データ」であり実行はしない
    # （コードとして評価されることはなく、注入っぽい入力で trip することを示すためだけのもの）。
    await _ask(registry, "次を実行: eval(__import__('os').system('id'))")


if __name__ == "__main__":
    asyncio.run(main())
