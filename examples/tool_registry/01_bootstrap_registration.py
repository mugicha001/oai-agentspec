"""bootstrap 一括登録パターン: 散在する Tool ファイルを bootstrap で 1 箇所に集約。

Tool ファイル（本例では同一ファイル内の関数群で代替）は純粋な Python 関数のみで書き、
lib（`oai_agentspec`）にも SDK（`agents`）にも依存しない。組み立てポイント（bootstrap）で
`ToolRegistry` にメタデータ付きで一括登録する。以後、`tool_registry.<name>` の属性アクセスで
メタデータ適用済み SDK Tool を取り出せる。Agent 少・Tool 多構成での一覧性・照会性を最大化する。

Azure OpenAI の環境変数（AZURE_OPENAI_* 。examples/_azure.py 参照）を設定して実行:
    uv run python examples/tool_registry/01_bootstrap_registration.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from agents import Runner

from oai_agentspec import AgentRegistry, AgentSpec, ToolRegistry, ToolSpec

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
from _azure import azure_model  # noqa: E402


# --- Tool 定義（散在ファイル相当・lib/SDK 非依存の純関数） ------------------
async def get_weather(city: str) -> str:
    """指定都市の天気を返す（例では固定値を返却）。"""
    return f"{city}: sunny, 22C"


async def search_docs(query: str) -> str:
    """ドキュメント検索。"""
    return f"3 results for {query!r}"


async def send_notification(to: str, message: str) -> str:
    """通知送信（副作用あり・冪等ではない想定）。"""
    return f"notified {to}: {message}"


# --- 組み立てポイント（bootstrap）: Tool 一元登録 ---------------------------
def build_registries() -> tuple[AgentRegistry, ToolRegistry]:
    """Agent Registry と Tool Registry を組み立てる。"""
    tool_registry = ToolRegistry()

    # メタデータは Tool の性質・運用要求に応じて宣言。すべての宣言が 1 箇所に集まり、
    # 全 Tool 一覧・有効/無効の運用トグルが tool_registry から統一的に扱える。
    tool_registry.register(ToolSpec(func=get_weather, timeout=10.0))
    tool_registry.register(ToolSpec(func=search_docs, timeout=5.0))
    tool_registry.register(
        ToolSpec(
            func=send_notification,
            needs_approval=True,  # HITL 承認必須（SDK needs_approval へ委譲）
            timeout=15.0,
        )
    )

    agent_registry = AgentRegistry()
    agent_registry.register(
        AgentSpec(
            name="assistant",
            instructions="ユーザーの依頼に応じて天気照会・文書検索・通知送信を行うアシスタント。",
            model=azure_model(),
            tools=[
                tool_registry.get_weather,  # 属性アクセスで FunctionTool を取り出す
                tool_registry.search_docs,
                tool_registry.send_notification,
            ],
        )
    )
    return agent_registry, tool_registry


async def main() -> None:
    """アシスタントを実行し、運用中の Tool 無効化例も示す。"""
    agent_registry, tool_registry = build_registries()

    # 通常実行
    agent = agent_registry.get("assistant")
    result = await Runner.run(agent, "東京の天気を教えて")
    print(f"[result] {result.final_output}\n")

    # 運用中の Tool 無効化例: 障害・保守で send_notification を一時停止したい場合。
    # 再構築不要で次の run から LLM に非提示になる（SDK is_enabled callable 経由）。
    tool_registry.metadata("send_notification").enabled = False
    print(f"[registered tools] {tool_registry.names()}")
    print(f"[send_notification enabled?] {tool_registry.metadata('send_notification').enabled}")


if __name__ == "__main__":
    asyncio.run(main())
