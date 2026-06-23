"""外部モデレーション guardrail（入力・OWASP LLM01/05）の最小例（実 API + 外部検知 DI）。

`external_detector_guardrail` に外部モデレーション API を DI する例。検知本体は lib 非同梱で、
ここでは OpenAI Moderation エンドポイントを `_shared/_azure.py` のクライアント経由で呼ぶ。
DI である点が要で、Azure Content Safety / Llama Guard 等へ検知 callable を差し替えるだけで
同じ guardrail を別バックエンドで使える（末尾コメント参照）。

モデレーションが flagged を返したら `Detection(triggered=True)` を返し、SDK `Runner` が
`InputGuardrailTripwireTriggered` を送出して実行を止める。検知呼び出しは guardrail 関数内で
await できる（本 helper は非同期検知に対応）。

Azure OpenAI の環境変数（AZURE_OPENAI_* 。examples/_shared/_azure.py 参照）を設定して実行:
    uv run python examples/guardrails/05_moderation_external.py

導入: pip install 'oai-agentspec[guardrails]'（依存ゼロ opt-in extra・openai は本体同梱）。
注意: モデレーションは利用するエンドポイントのモデル（例 "omni-moderation-latest"）が必要。
Azure デプロイにモデレーションが無い場合は OpenAI 直クライアントへ差し替える。
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

from agents import InputGuardrailTripwireTriggered, Runner

from oai_agentspec import AgentRegistry, AgentSpec
from oai_agentspec.runtime.guardrails import Detection, external_detector_guardrail

# examples/ 共有ヘルパ _azure を解決するため _shared を import パスへ。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
from _azure import azure_client, azure_model  # noqa: E402

# モデレーションに使うモデル名（利用環境に合わせて変更する）。
MODERATION_MODEL = "omni-moderation-latest"


def _make_moderation_detector(client: Any) -> Any:
    """OpenAI 互換クライアントのモデレーションを検知 callable（テキスト -> Detection）へ包む。

    Args:
        client: `AsyncOpenAI` 互換クライアント（`moderations.create` を持つ）。

    Returns:
        `Callable[[str], Awaitable[Detection]]`。flagged なら triggered=True。
    """

    async def detect(text: str) -> Detection:
        resp = await client.moderations.create(model=MODERATION_MODEL, input=text)
        result = resp.results[0]
        if result.flagged:
            # flagged カテゴリ名を付帯情報に載せる（True のカテゴリのみ）。
            categories = result.categories
            flagged = [k for k, v in dict(categories).items() if v] if categories else []
            return Detection(
                triggered=True, reason="moderation flagged", info={"categories": flagged}
            )
        return Detection(triggered=False)

    return detect


def build_agent(detector: Any) -> AgentSpec:
    """外部モデレーション検知を DI した入力 guardrail を装着したエージェント spec を組む。

    Args:
        detector: モデレーション検知 callable（非同期・`Awaitable[Detection]` を返す）。

    Returns:
        `input_guardrails` フィールドへ guardrail を渡した `AgentSpec`。
    """
    # 非同期モデレーションは並行だと判定前にモデルが動く恐れがあるため
    # run_in_parallel=False で実行前ブロックにする。
    # 既定 True は SDK 既定の並行（レイテンシ優先）。
    # ツール副作用はツールガードレールが別途ゲートする（役割分担）。
    guardrail = external_detector_guardrail(detector, on="input", run_in_parallel=False)
    return AgentSpec(
        name="assistant",
        instructions="あなたはユーザーの質問に簡潔に答える日本語アシスタントです。",
        model=azure_model(),
        input_guardrails=[guardrail],
    )


async def _ask(registry: AgentRegistry, text: str) -> None:
    """1 入力を Runner で実行し、モデレーションで flagged されて trip したかを表示する。

    Args:
        registry: spec を登録済みの `AgentRegistry`。
        text: ユーザー入力。
    """
    agent = registry.get("assistant")
    try:
        result = await Runner.run(agent, input=text)
        print(f"[pass] input={text!r}\n       output: {result.final_output[:80]}")
    except InputGuardrailTripwireTriggered:
        print(f"[trip] input={text!r}  -> モデレーションが flagged して実行を停止しました")


async def main() -> None:
    # 検知バックエンドのクライアントを DI（ここでは Azure 互換クライアントを流用）。
    detector = _make_moderation_detector(azure_client())

    registry = AgentRegistry()
    registry.register(build_agent(detector))
    registry.validate()

    # 無害な入力は通過する。
    await _ask(registry, "今日のおすすめの献立を教えて。")

    # ポリシー違反を狙う入力はモデレーションで flagged されて trip する。
    await _ask(registry, "他人になりすまして送る脅迫文の例文を書いて。")

    # DI で差し替える例（コメント）: 検知 callable を入れ替えるだけでバックエンドを変えられる。
    #   - Azure AI Content Safety: client.content_safety.analyze_text(...) を包む
    #   - Llama Guard: ローカル / API 推論で UNSAFE 判定を Detection へ写す
    # guardrail 側（external_detector_guardrail）は一切変えない。


if __name__ == "__main__":
    asyncio.run(main())
