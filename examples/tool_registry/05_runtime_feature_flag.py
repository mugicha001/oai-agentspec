"""enabled 動的トグルをフィーチャーフラグとして使う実運用パターン。

ToolRegistry の `metadata(name).enabled = False/True` は SDK `is_enabled` に closure 経由で
結線されているため、構築済み Agent の再構築なしに次の run から LLM への Tool 提示有無に
反映される（FR-4/6）。本例では同じ Agent で 2 回 run を実行し、途中で `send_email` を
無効化すると 2 回目のモデルが `send_email` を選べなくなる（別の Tool にフォールバック
または回答不可を返す）様子を示す。

想定ユースケース:
- ロール別のツール露出制御（一般ユーザーには危険操作を隠す）
- 障害時の外部 API ツールの一時停止（rollout kill switch）
- 時間帯・レート制限による段階的公開

Azure OpenAI の環境変数（AZURE_OPENAI_* 。examples/_azure.py 参照）を設定して実行:
    uv run python examples/tool_registry/05_runtime_feature_flag.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from agents import Runner

from oai_agentspec import AgentRegistry, AgentSpec, ToolRegistry, ToolSpec

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
from _azure import azure_model  # noqa: E402


async def send_email(to: str, subject: str, body: str) -> str:
    """メール送信（副作用あり）。"""
    return f"sent to {to}: [{subject}] {body[:20]}"


async def draft_email(to: str, subject: str, body: str) -> str:
    """メール送信の代替として下書きだけ作成（副作用なし）。"""
    return f"drafted for {to}: [{subject}] {body[:20]}"


def build(tool_registry: ToolRegistry) -> AgentRegistry:
    """Agent を組み立てる。send_email と draft_email の両方を提示する。"""
    tool_registry.register(ToolSpec(func=send_email, needs_approval=False))
    tool_registry.register(ToolSpec(func=draft_email))

    agent_registry = AgentRegistry()
    agent_registry.register(
        AgentSpec(
            name="mailer",
            instructions=(
                "ユーザーの依頼に応じて、可能ならメールを実送信し、"
                "実送信できない場合は下書きを作成するアシスタント。"
            ),
            model=azure_model(),
            tools=[tool_registry.send_email, tool_registry.draft_email],
        )
    )
    return agent_registry


async def main() -> None:
    tool_registry = ToolRegistry()
    agent_registry = build(tool_registry)
    agent = agent_registry.get("mailer")

    prompt = "取引先の tanaka@example.com に、次回打ち合わせの候補日を送って"

    # --- 1 回目: send_email 有効。モデルは実送信 tool を選ぶことが多い ---
    print("[phase 1] send_email = ENABLED")
    print(f"[registered] {tool_registry.names()}\n")
    r1 = await Runner.run(agent, prompt)
    print(f"[phase 1 result] {r1.final_output}\n")

    # --- Registry 上で send_email を無効化（Agent 再構築なし・FR-4） ---
    tool_registry.metadata("send_email").enabled = False
    print("[phase 2] send_email = DISABLED (feature flag off)\n")

    # --- 2 回目: 同じ Agent インスタンスで再実行。send_email は LLM から隠れる ---
    r2 = await Runner.run(agent, prompt)
    print(f"[phase 2 result] {r2.final_output}\n")

    # フィーチャーフラグを戻す例
    tool_registry.metadata("send_email").enabled = True
    print(f"[phase 3] send_email = ENABLED (restored) — registered={tool_registry.names()}")


if __name__ == "__main__":
    asyncio.run(main())
