"""MCP サーバを `AgentSpec` の専用フィールドで宣言する最小例（実 API）。

`AgentSpec(mcp_servers=[...], mcp_config={...})` と書くだけで、build 時に
`agents.Agent` の同名フィールドへ渡る。`extra` へ SDK kwarg を素通しする必要はない
（型・補完が効き、綴り誤りは build 時に `ValueError` で検出される）。

MCP ツールは `spec.tools` に載らず、SDK が **run 時**にサーバへ list_tools して
`FunctionTool` へ変換する（ターンごとに再解決される）。したがって `spec.tools` は空でも
モデルは MCP のツールを呼べる。

サーバの接続 / 切断（`connect()` / `cleanup()`）は**利用者責務**である
（lib は宣言を素通しするだけで lifecycle を持たない = build-don't-run）。本例では
`MCPServerStdio` を async context manager として使い、`with` を抜けるときに切断する。

Azure OpenAI の環境変数（AZURE_OPENAI_* 。examples/_shared/_azure.py 参照）を設定して実行:
    uv run python examples/mcp/01_declarative_mcp_servers.py

追加の extra 導入は不要（`mcp` は openai-agents の依存として必ず入る）。
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

from agents import Runner
from agents.mcp import MCPServerStdio

from oai_agentspec import AgentRegistry, AgentSpec

# examples/ 共有ヘルパ _azure を解決するため _shared を import パスへ。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
from _azure import azure_model  # noqa: E402

SERVER_PATH = Path(__file__).resolve().parent / "_server.py"


def build_registry(server: Any, model: Any) -> AgentRegistry:
    """MCP サーバを宣言した spec を登録した registry を組む。

    `mcp_servers` / `mcp_config` は `AgentSpec` の専用フィールド（kw_only）で、build 時に
    `agents.Agent` の同名フィールドへ渡る。`tools` は空のままでよい（MCP のツールは run 時に
    SDK が解決する）。

    Args:
        server: 接続済みの MCP サーバ（`agents.mcp.MCPServer`）。lifecycle は呼び出し側が持つ。
        model: 使用するモデル（`azure_model()` の戻り値・テストでは FakeModel を渡せる）。

    Returns:
        spec を登録済みの `AgentRegistry`。
    """
    registry = AgentRegistry()
    registry.register(
        AgentSpec(
            name="inventory",
            instructions=(
                "あなたは在庫の問い合わせ担当です。在庫数は get_stock、"
                "SKU 一覧は list_skus を必ず使って答えてください。"
            ),
            model=model,
            # MCP 配線は専用フィールドで宣言する（extra 素通しは不要）。
            mcp_servers=[server],
            # 未指定なら SDK 既定（空 dict）に委ねられる。ここでは既定値を明示している。
            # include_server_in_tool_names=True にするとツールの公開名が
            # `mcp_{サーバ名}__{ツール名}` になる（ポリシー等で名前を参照する場合は追随が必要）。
            mcp_config={"include_server_in_tool_names": False},
        )
    )
    registry.validate()
    return registry


async def main() -> None:
    # 接続 / 切断は利用者責務。async with で抜けるときに cleanup される。
    async with MCPServerStdio(
        name="demo-inventory",
        params={"command": sys.executable, "args": [str(SERVER_PATH)]},
    ) as server:
        registry = build_registry(server, azure_model())
        agent = registry.get("inventory")

        # MCP ツールは build 時ではなく run 時に解決される（agent.tools は空のまま）。
        print(f"agent.tools は空: {agent.tools == []}")
        print(f"宣言した MCP サーバ: {[s.name for s in agent.mcp_servers]}")
        print(f"mcp_config: {agent.mcp_config}")

        result = await Runner.run(agent, input="SKU-1 の在庫はいくつ?")
        print(f"\n[MCP ツール経由の回答] {result.final_output}")

        result = await Runner.run(agent, input="扱っている SKU を全部教えて")
        print(f"[MCP ツール経由の回答] {result.final_output}")


if __name__ == "__main__":
    asyncio.run(main())
