"""AGT ガバナンスの監査ログ共有と既存 hooks 合成の例（実 API）。

既定の監査 sink（AGT `AuditLog`・tamper-evident ハッシュチェーン）は builder が初回 build で
生成し、以降の build と共有する。複数エージェント構成でも 1 本のチェーンに時系列で記録され、
`GovernedAgentBuilder.audit_sink` プロパティで後から取得・検証できる。出力先を差し替えたい
場合は `audit_sink=` に `record(...)` を持つ任意オブジェクトを DI する（監査 details にはツール
引数 JSON が全文記録されるため、機密引数を扱う場合は sink の永続先を考慮して選定する）。

`spec.hooks` に利用者フックがある場合も上書きされず、「監査記録 -> 既存フックへ委譲」の順で
合成される（ライフサイクル監査: agent_start / tool_start / tool_end / handoff / agent_end）。

ポリシーは YAML パスのほか、構築済みの AGT `GovernancePolicy` オブジェクトでも渡せる
（本例はオブジェクト形）。

Azure OpenAI の環境変数（AZURE_OPENAI_* 。examples/_shared/_azure.py 参照）を設定して実行:
    uv run python examples/governance/02_audit_log.py

導入: pip install 'oai-agentspec[governance]'（AGT を取り込む opt-in extra）。
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

from agents import Runner, function_tool
from openai_agents_trust import GovernancePolicy

from oai_agentspec import AgentRegistry, AgentSpec
from oai_agentspec.runtime.governance import GovernedAgentBuilder

# examples/ 共有ヘルパ _azure を解決するため _shared を import パスへ。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
from _azure import azure_model  # noqa: E402


@function_tool
def classify(text: str) -> str:
    """問い合わせ種別を返す（triage 用ツール）。

    Args:
        text: 問い合わせ本文。

    Returns:
        分類結果の文字列。
    """
    return "category: order"


@function_tool
def lookup_order(order_id: str) -> str:
    """注文の状況を返す（support 用ツール）。

    Args:
        order_id: 注文 ID。

    Returns:
        注文状況の文字列。
    """
    return f"注文 {order_id}: 配送中です"


class PrintHooks:
    """利用者定義の既存フック例（合成後も失われず委譲される）。"""

    async def on_start(self, context: Any, agent: Any) -> None:
        """エージェント開始時に名前を表示する（利用者フックが生きている証跡）。

        Args:
            context: SDK の実行コンテキスト。
            agent: 開始したエージェント。
        """
        print(f"  [user-hook] on_start: {agent.name}")


def build_registry() -> tuple[AgentRegistry, GovernedAgentBuilder]:
    """ポリシーオブジェクト形でガバナンス builder を注入した registry を組む。

    Returns:
        `(registry, builder)`。builder は共有監査 sink の取得用に返す。
    """
    policy = GovernancePolicy(name="ops", allowed_tools=["classify", "lookup_order"])
    builder = GovernedAgentBuilder(policy=policy)  # audit_sink 未指定 -> 既定 sink を共有生成
    registry = AgentRegistry(agent_builder=builder)
    registry.register(
        AgentSpec(
            name="triage",
            instructions="問い合わせを classify ツールで分類して結果だけ短く答えてください。",
            model=azure_model(),
            tools=[classify],
        )
    )
    registry.register(
        AgentSpec(
            name="support",
            instructions="注文の照会には lookup_order ツールを必ず使ってください。",
            model=azure_model(),
            tools=[lookup_order],
            # 利用者フックは上書きされず「監査記録 -> 委譲」の順で合成される。
            hooks=PrintHooks(),
        )
    )
    registry.validate()
    return registry, builder


async def main() -> None:
    registry, builder = build_registry()

    print("triage 実行:")
    await Runner.run(registry.get("triage"), input="注文がまだ届きません")
    print("support 実行:")
    await Runner.run(registry.get("support"), input="注文 B456 の状況を教えて")

    # 既定 sink は build 跨ぎで共有され、1 本のチェーンに両エージェントの記録が並ぶ。
    sink = builder.audit_sink
    print("\n監査ログ（agent_id / action / decision・時系列 1 本のチェーン）:")
    for entry in sink.get_entries():
        print(f"  {entry.agent_id:8s}  {entry.action:24s}  {entry.decision}")

    # ハッシュチェーンの整合検証（改ざんがあれば False になる）。
    print(f"\nチェーン検証: verify_chain() = {sink.verify_chain()}")


if __name__ == "__main__":
    asyncio.run(main())
