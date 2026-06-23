"""bundle YAML（制限の全量を 1 ファイルに宣言）の例（実 API）。

`GovernedAgentBuilder.from_yaml("governance.yaml")` で、既定（`default`）と per-agent
（`agents`）の制限を単一の宣言ファイルから構築する。制限定義が YAML（ポリシー本体）と
コード（エージェント名との対応付け）に分離しないため、「どのエージェントが何をできるか」を
1 ファイルの監査で把握できる。コード側はファイル参照の 1 行だけになる。

bundle は構築糖衣であり、通常コンストラクタ（`policy=` / `overrides=`・コードでポリシー
オブジェクトを組む形式）と等価に動く（`03_per_agent_policy.py` と同じ強制・同じ
`unapplied_overrides` / 監査 sink 共有）。どちらの形式を使うかは運用の好みで選べる。

Azure OpenAI の環境変数（AZURE_OPENAI_* 。examples/_shared/_azure.py 参照）を設定して実行:
    uv run python examples/governance/04_policy_bundle.py

導入: pip install 'oai-agentspec[governance]'（AGT を取り込む opt-in extra）。
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from agents import Runner, function_tool

from oai_agentspec import AgentRegistry, AgentSpec
from oai_agentspec.runtime.governance import GovernedAgentBuilder, PolicyViolationError

# examples/ 共有ヘルパ _azure を解決するため _shared を import パスへ。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
from _azure import azure_model  # noqa: E402

BUNDLE_PATH = Path(__file__).resolve().parent / "policies" / "governance.yaml"


@function_tool
def lookup_order(order_id: str) -> str:
    """注文の状況を返す（default / support 双方の allowlist に掲載）。

    Args:
        order_id: 注文 ID。

    Returns:
        注文状況の文字列。
    """
    return f"注文 {order_id}: 発送準備中です"


@function_tool
def refund(order_id: str) -> str:
    """注文を返金する（bundle の agents.support のみ許可）。

    Args:
        order_id: 注文 ID。

    Returns:
        返金結果の文字列（triage ではポリシー違反でここに到達しない）。
    """
    return f"注文 {order_id} を返金しました"


def build_registry() -> tuple[AgentRegistry, GovernedAgentBuilder]:
    """bundle YAML から builder を構築し registry を組む（コード側は参照 1 行）。

    Returns:
        `(registry, builder)`。builder は unapplied_overrides / audit_sink の確認用に返す。
    """
    # 制限の全量は governance.yaml に宣言済み。コード側にポリシー定義を持たない。
    builder = GovernedAgentBuilder.from_yaml(BUNDLE_PATH)
    registry = AgentRegistry(agent_builder=builder)
    instructions = (
        "あなたは注文サポート担当です。注文の照会には lookup_order を、"
        "返金依頼には refund を必ず使ってください。"
    )
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

    # bundle の agents.support により refund は support のみ allow（triage は default で deny）。
    await _ask(registry, "support", "注文 A123 を返金して")
    await _ask(registry, "triage", "注文 A123 を返金して")

    # typo 検知と監査は通常コンストラクタと同じに使える（bundle は構築糖衣）。
    assert not builder.unapplied_overrides, (
        f"bundle の agents に未適用キーがあります（typo の疑い）: {builder.unapplied_overrides}"
    )
    sink = builder.audit_sink
    print("\n監査ログ（agent_id / action / decision）:")
    for entry in sink.get_entries():
        print(f"  {entry.agent_id:8s}  {entry.action:24s}  {entry.decision}")
    print(f"チェーン検証: verify_chain() = {sink.verify_chain()}")


if __name__ == "__main__":
    asyncio.run(main())
