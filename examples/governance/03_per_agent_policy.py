"""per-agent ポリシー（builder overrides）の例（実 API）。

`GovernedAgentBuilder(policy=既定, overrides={"エージェント名": ポリシー})` で、エージェント
ごとに異なるポリシーを適用する。overrides に掲載されたエージェントはそのポリシー、未掲載は
既定ポリシーへフォールバックする（引き当ては `spec.name` との完全一致）。

ポイントは**同一ツールのエージェント別出し分け**: 同じ `lookup_order` / `refund` を tools に
持つ 2 エージェントでも、overrides により一方では allow・他方では deny にできる（registry
一括ポリシーではできない制御）。監査 sink は overrides を使っても builder で 1 本共有される。

overrides キーの typo（登録エージェント名と不一致）は黙って既定へフォールバックするため、
全エージェント build 後に `unapplied_overrides` が空であることを確認するのが安全な運用となる。

Azure OpenAI の環境変数（AZURE_OPENAI_* 。examples/_shared/_azure.py 参照）を設定して実行:
    uv run python examples/governance/03_per_agent_policy.py

導入: pip install 'oai-agentspec[governance]'（AGT を取り込む opt-in extra）。
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from agents import Runner, function_tool
from openai_agents_trust import GovernancePolicy

from oai_agentspec import AgentRegistry, AgentSpec
from oai_agentspec.runtime.governance import GovernedAgentBuilder, PolicyViolationError

# examples/ 共有ヘルパ _azure を解決するため _shared を import パスへ。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
from _azure import azure_model  # noqa: E402


@function_tool
def lookup_order(order_id: str) -> str:
    """注文の状況を返す（両エージェントが tools に持つ共通ツール）。

    Args:
        order_id: 注文 ID。

    Returns:
        注文状況の文字列。
    """
    return f"注文 {order_id}: 発送準備中です"


@function_tool
def refund(order_id: str) -> str:
    """注文を返金する（support のみ許可されるツール）。

    Args:
        order_id: 注文 ID。

    Returns:
        返金結果の文字列（triage ではポリシー違反でここに到達しない）。
    """
    return f"注文 {order_id} を返金しました"


def build_registry() -> tuple[AgentRegistry, GovernedAgentBuilder]:
    """既定 + overrides の per-agent ポリシーで registry を組む。

    Returns:
        `(registry, builder)`。builder は unapplied_overrides / audit_sink の確認用に返す。
    """
    builder = GovernedAgentBuilder(
        # 既定: 照会のみ許可（overrides 未掲載の全エージェントの基準線）。
        policy=GovernancePolicy(name="readonly", allowed_tools=["lookup_order"]),
        # support だけ refund も許可（triage は同じ refund ツールを持っていても deny される）。
        overrides={
            "support": GovernancePolicy(name="support", allowed_tools=["lookup_order", "refund"]),
        },
    )
    registry = AgentRegistry(agent_builder=builder)
    instructions = (
        "あなたは注文サポート担当です。注文の照会には lookup_order を、"
        "返金依頼には refund を必ず使ってください。"
    )
    # 2 エージェントは同一のツール群を宣言する（宣言面は同じ・ポリシーだけが違う）。
    registry.register(
        AgentSpec(
            name="triage",
            instructions=instructions,
            model=azure_model(),
            tools=[lookup_order, refund],
        )
    )
    registry.register(
        AgentSpec(
            name="support",
            instructions=instructions,
            model=azure_model(),
            tools=[lookup_order, refund],
        )
    )
    registry.validate()
    return registry, builder


async def _ask(registry: AgentRegistry, agent_name: str, text: str) -> None:
    """1 入力を指定エージェントで実行し、許可 / 拒否を表示する。

    Args:
        registry: spec を登録済みの `AgentRegistry`。
        agent_name: 実行するエージェント名。
        text: ユーザー入力。
    """
    agent = registry.get(agent_name)
    try:
        result = await Runner.run(agent, input=text)
        print(
            f"[allow] agent={agent_name} input={text!r}\n        output: {result.final_output[:80]}"
        )
    except Exception as exc:
        cause = exc.__cause__
        if isinstance(exc, PolicyViolationError) or isinstance(cause, PolicyViolationError):
            reason = cause or exc
            print(f"[deny]  agent={agent_name} input={text!r}\n        -> {reason}")
        else:
            raise


async def main() -> None:
    registry, builder = build_registry()

    # 同じ refund ツールでも、support は allow・triage（既定ポリシー）は deny。
    await _ask(registry, "support", "注文 A123 を返金して")
    await _ask(registry, "triage", "注文 A123 を返金して")

    # 照会は両エージェントとも allow（既定 / override 双方の allowlist に掲載）。
    await _ask(registry, "triage", "注文 A123 の状況を教えて")

    # typo 検知: 全エージェント build 後に未適用キーが残っていないことを確認する。
    assert not builder.unapplied_overrides, (
        f"overrides に未適用キーがあります（typo の疑い）: {builder.unapplied_overrides}"
    )
    print("\nunapplied_overrides は空（overrides は全キー適用済み）")

    # 監査は overrides を使っても 1 本のチェーンに両エージェント分が並ぶ。
    sink = builder.audit_sink
    print("監査ログ（agent_id / action / decision）:")
    for entry in sink.get_entries():
        print(f"  {entry.agent_id:8s}  {entry.action:24s}  {entry.decision}")
    print(f"チェーン検証: verify_chain() = {sink.verify_chain()}")


if __name__ == "__main__":
    asyncio.run(main())
