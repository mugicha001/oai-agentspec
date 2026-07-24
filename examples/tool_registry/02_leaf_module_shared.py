"""葉モジュール分散登録パターンの実行例。

`leaf_module_shared` パッケージを import することで、各 Tool ファイル
（`weather.py` / `docs.py` / `notification.py`）の登録が発火し、共有 `tool_registry` に
全 Tool が集約された状態になる。Agent 側は `tool_registry.<name>` の属性アクセスで
FunctionTool を取り出せる（bootstrap 一括登録パターンと同じ使用感）。

Azure OpenAI の環境変数（AZURE_OPENAI_* 。examples/_azure.py 参照）を設定して実行:
    uv run python examples/tool_registry/02_leaf_module_shared.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from agents import Runner

from oai_agentspec import AgentRegistry, AgentSpec

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
# パッケージ import で各 Tool ファイルの登録が発火する（`leaf_module_shared/__init__.py`
# 参照）。tool_registry 変数はこの import 後に「全 Tool 登録済み」の状態で参照できる。
from leaf_module_shared import tool_registry  # noqa: E402

from _azure import azure_model  # noqa: E402


def build_agent_registry() -> AgentRegistry:
    """Agent Registry を組み立てる（Tool は既に leaf_module_shared で登録済み）。"""
    agent_registry = AgentRegistry()
    agent_registry.register(
        AgentSpec(
            name="assistant",
            instructions=("ユーザーの依頼に応じて天気照会・文書検索・通知送信を行うアシスタント。"),
            model=azure_model(),
            tools=[
                tool_registry.get_weather,
                tool_registry.search_docs,
                tool_registry.send_notification,
            ],
        )
    )
    return agent_registry


async def main() -> None:
    """アシスタントを実行し、分散登録が中央 Registry に集約されていることを示す。"""
    agent_registry = build_agent_registry()
    agent = agent_registry.get("assistant")

    result = await Runner.run(agent, "東京の天気を教えて")
    print(f"[result] {result.final_output}\n")

    # 分散登録されたが照会は中央集権的（tool_registry 1 インスタンスに集約）。
    print(f"[registered tools] {tool_registry.names()}")

    # 運用中のトグル例（bootstrap パターンと同じ操作性・FR-4/6）。
    tool_registry.metadata("send_notification").enabled = False
    print(f"[send_notification enabled?] {tool_registry.metadata('send_notification').enabled}")


if __name__ == "__main__":
    # `leaf_module_shared` パッケージを import 可能にする（examples/tool_registry を sys.path へ）。
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    asyncio.run(main())
