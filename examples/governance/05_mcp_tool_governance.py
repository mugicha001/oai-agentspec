"""MCP サーバ経由のツールにもポリシーを効かせる例（実 API）。

MCP のツールは `spec.tools` に載らず、SDK が **run 時**（ターンごと）にサーバへ list_tools して
`FunctionTool` へ変換する。そのため build 時に実行本体（`on_invoke_tool`）をラップする経路では
統治できないが、`GovernedAgentBuilder` が装着する `AgentHooks.on_tool_start` が MCP 由来ツールを
評価するため、宣言は `spec.tools` と同じ `allowed_tools` / `blocked_patterns` で足りる
（ポリシーの規約は 1 本のまま）。

利用者の追加記述は「builder を 1 つ差し替える」+ ポリシー宣言のみで、`AgentSpec` の宣言面
（`mcp_servers` の指定）は govern の有無で変わらない。

本例は `examples/mcp/_server.py`（同梱の最小 MCP サーバ・stdio・ネットワークへ出ない）を
`MCPServerStdio` から自動起動するため、外部 MCP サーバの準備は不要。

Azure OpenAI の環境変数（AZURE_OPENAI_* 。examples/_shared/_azure.py 参照）を設定して実行:
    uv run python examples/governance/05_mcp_tool_governance.py

導入: pip install 'oai-agentspec[governance]'（AGT を取り込む opt-in extra）。
`mcp` は openai-agents の依存として必ず入るため追加導入は不要。
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

from agents import Runner
from agents.mcp import MCPServerStdio

from oai_agentspec import AgentRegistry, AgentSpec
from oai_agentspec.runtime.governance import GovernedAgentBuilder, PolicyViolationError

# examples/ 共有ヘルパ _azure を解決するため _shared を import パスへ。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
from _azure import azure_model  # noqa: E402

# 同梱の最小 MCP サーバ（get_stock / list_skus を提供する）。
MCP_SERVER_PATH = Path(__file__).resolve().parent.parent / "mcp" / "_server.py"

# ポリシーはコード（AGT オブジェクト）でもよいが、本例は YAML と同じ意味の dict 相当を
# `GovernancePolicy` で組む（`policies/*.yaml` を使う形は 01 / 04 を参照）。
POLICY_ALLOWED = ["list_skus"]  # get_stock は未掲載 -> 呼ぶと拒否される


def build_registry(server: Any, model: Any, policy: Any) -> tuple[AgentRegistry, Any]:
    """govern builder を注入した registry を組む（MCP 宣言は通常どおり）。

    Args:
        server: 接続済みの MCP サーバ（lifecycle は呼び出し側が持つ）。
        model: 使用するモデル（`azure_model()` の戻り値）。
        policy: AGT ポリシーオブジェクト。

    Returns:
        `(registry, builder)`。builder は監査 sink の取得用に返す。
    """
    # 追加はこの 1 行（builder 差し替え）とポリシー宣言のみ。spec は通常どおり書く。
    builder = GovernedAgentBuilder(policy=policy)
    registry = AgentRegistry(agent_builder=builder)
    registry.register(
        AgentSpec(
            name="inventory",
            instructions=(
                "あなたは在庫の問い合わせ担当です。在庫数は get_stock、"
                "SKU 一覧は list_skus を必ず使って答えてください。"
            ),
            model=model,
            # MCP 配線は専用フィールドで宣言する（govern の有無で宣言面は変わらない）。
            mcp_servers=[server],
        )
    )
    registry.validate()
    return registry, builder


async def _ask(registry: AgentRegistry, text: str) -> None:
    """1 入力を Runner で実行し、許可 / 拒否を表示する。

    Args:
        registry: spec を登録済みの `AgentRegistry`。
        text: ユーザー入力。
    """
    agent = registry.get("inventory")
    try:
        result = await Runner.run(agent, input=text)
        print(f"[allow] {text!r}\n        -> {result.final_output[:80]}")
    except Exception as exc:
        # Runner は tool 実行例外を SDK 例外へラップし得るため __cause__ も確認する
        # （MCP 経路も `spec.tools` 経路と同じ着地になる）。
        cause = exc.__cause__
        if isinstance(exc, PolicyViolationError) or isinstance(cause, PolicyViolationError):
            print(f"[deny]  {text!r}\n        -> 実行前に拒否: {cause or exc}")
        else:
            raise


async def main() -> None:
    from openai_agents_trust import GovernancePolicy

    policy = GovernancePolicy(name="inventory_readonly", allowed_tools=POLICY_ALLOWED)

    # 接続 / 切断は利用者責務（lib は lifecycle を持たない）。
    async with MCPServerStdio(
        name="demo-inventory",
        params={"command": sys.executable, "args": [str(MCP_SERVER_PATH)]},
    ) as server:
        registry, builder = build_registry(server, azure_model(), policy)

        # allowlist に載った MCP ツールは通常どおり実行される。
        await _ask(registry, "扱っている SKU を全部教えて")

        # allowlist 未掲載の MCP ツールは実ツールを呼ばずに拒否される
        # （従来は MCP 経由だと評価を受けずに実行されていた）。
        await _ask(registry, "SKU-1 の在庫はいくつ?")

        # 監査には `spec.tools` と同じ `tool:{name}` 形式で評価結果が残る。
        sink = builder.audit_sink
        print("\n監査ログ（tool: 行が per-call の判定結果・tool_start: は呼び出し開始の記録）:")
        for entry in sink.get_entries():
            print(f"  {entry.agent_id}  {entry.action}  {entry.decision}")
        print(f"チェーン検証: verify_chain() = {sink.verify_chain()}")


if __name__ == "__main__":
    asyncio.run(main())
