"""E2E 統合: ToolRegistry × AgentSpec × AgentRegistry の結線検証（Issue #27 Task 4）。

`ToolRegistry` の属性アクセスで得た SDK `FunctionTool` を、既存の
`AgentSpec(tools=[...])` にそのまま渡して `AgentRegistry` 経由で SDK `Agent` を
ビルドできること（NFR-4 純粋追加）、および `metadata(name).enabled` の動的トグルが
「構築済み Tool を再構築せず」に次の `is_enabled(ctx, agent)` 評価へ反映されること
（FR-4）を実型で検証する。
"""

from __future__ import annotations

import asyncio
import inspect
from unittest.mock import MagicMock

import pytest
from agents import FunctionTool

from oai_agentspec import AgentRegistry, AgentSpec, ToolRegistry, ToolSpec

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# ヘルパ
# ---------------------------------------------------------------------------
async def _get_weather(city: str) -> str:
    """テスト用の素朴な async tool 関数（天気ダミー）。"""
    return f"weather:{city}"


async def _get_time(tz: str) -> str:
    """テスト用の素朴な async tool 関数（時刻ダミー）。"""
    return f"time:{tz}"


async def _get_news(topic: str) -> str:
    """テスト用の素朴な async tool 関数（ニュースダミー）。"""
    return f"news:{topic}"


def _resolve_is_enabled(tool: FunctionTool, ctx: object, agent: object) -> bool:
    """SDK `is_enabled` の MaybeAwaitable[bool] を吸収して bool を得る。"""
    if isinstance(tool.is_enabled, bool):
        return tool.is_enabled
    result = tool.is_enabled(ctx, agent)
    if inspect.iscoroutine(result):
        return asyncio.get_event_loop().run_until_complete(result)
    return bool(result)


# ---------------------------------------------------------------------------
# 1. 属性アクセス → AgentSpec → AgentRegistry ビルド（NFR-4 純粋追加）
# ---------------------------------------------------------------------------
def test_正常系_属性アクセスで得たToolをAgentSpecに渡してAgentRegistry経由でビルドできる() -> None:
    """`tool_registry.<name>` で得た FunctionTool をそのまま AgentSpec.tools に渡し、
    AgentRegistry 経由で SDK Agent がビルドでき、agent.tools に該当 Tool が入る。"""
    tool_registry = ToolRegistry()
    tool_registry.register(ToolSpec(func=_get_weather, name="get_weather"))

    spec = AgentSpec(
        name="assistant",
        instructions="test",
        tools=[tool_registry.get_weather],
    )
    agent_registry = AgentRegistry()
    agent_registry.register(spec)

    agent = agent_registry.get("assistant")
    assert len(agent.tools) == 1
    tool = agent.tools[0]
    assert isinstance(tool, FunctionTool)
    assert tool.name == "get_weather"
    # 同一インスタンスが Agent の tools に入っている（Registry のキャッシュを経由）
    assert tool is tool_registry.get_weather


# ---------------------------------------------------------------------------
# 2. 属性アクセスキャッシュ検証（同一名は同じ FunctionTool を返す）
# ---------------------------------------------------------------------------
def test_正常系_同一名属性アクセスは同じFunctionToolを返す_キャッシュ検証() -> None:
    """`tool_registry.<name>` は初回のみビルドし、以降はキャッシュから同一インスタンスを返す。"""
    tool_registry = ToolRegistry()
    tool_registry.register(ToolSpec(func=_get_weather, name="get_weather"))

    first = tool_registry.get_weather
    second = tool_registry.get_weather
    assert first is second

    # 同一インスタンスを 2 回並べても AgentSpec / AgentRegistry がビルドできる
    spec = AgentSpec(
        name="assistant",
        instructions="test",
        tools=[tool_registry.get_weather, tool_registry.get_weather],
    )
    agent_registry = AgentRegistry()
    agent_registry.register(spec)
    agent = agent_registry.get("assistant")
    assert len(agent.tools) == 2
    assert agent.tools[0] is agent.tools[1] is first


# ---------------------------------------------------------------------------
# 3. FR-4: enabled 動的トグル（再構築なしで is_enabled 評価に反映）
# ---------------------------------------------------------------------------
def test_正常系_enabled動的トグル_metadata経由の変更がis_enabledに即反映される() -> None:
    """`metadata(name).enabled = False` が同じ FunctionTool の `is_enabled` に即反映される
    （構築済み Tool object が再構築されない・FR-4 の核）。"""
    tool_registry = ToolRegistry()
    tool_registry.register(ToolSpec(func=_get_weather, name="tool_a", enabled=True))

    tool = tool_registry.tool_a  # 一度だけビルド
    ctx = MagicMock()
    agent = MagicMock()

    assert _resolve_is_enabled(tool, ctx, agent) is True

    tool_registry.metadata("tool_a").enabled = False
    # 同じ tool インスタンスに対して再度評価
    assert tool is tool_registry.tool_a  # 再構築されていない
    assert _resolve_is_enabled(tool, ctx, agent) is False

    tool_registry.metadata("tool_a").enabled = True
    assert tool is tool_registry.tool_a
    assert _resolve_is_enabled(tool, ctx, agent) is True


# ---------------------------------------------------------------------------
# 4. Agent 構築後にトグルしても同じ FunctionTool が提示される（FR-4 別角度）
# ---------------------------------------------------------------------------
def test_正常系_Agent構築後にenabledトグルしても同じFunctionToolが提示される() -> None:
    """Agent 再構築なしで SDK 側 is_enabled による動的判定に委ねられる（FR-4）。"""
    tool_registry = ToolRegistry()
    tool_registry.register(ToolSpec(func=_get_weather, name="tool_a", enabled=True))

    spec = AgentSpec(
        name="assistant",
        instructions="test",
        tools=[tool_registry.tool_a],
    )
    agent_registry = AgentRegistry()
    agent_registry.register(spec)

    agent = agent_registry.get("assistant")
    assert agent.tools[0] is tool_registry.tool_a

    tool_registry.metadata("tool_a").enabled = False
    # Agent の tools 参照は不変（同一インスタンス）
    assert agent.tools[0] is tool_registry.tool_a

    ctx = MagicMock()
    fake_agent = MagicMock()
    assert _resolve_is_enabled(agent.tools[0], ctx, fake_agent) is False


# ---------------------------------------------------------------------------
# 5. 複数 Tool の登録と Agent 構築（順序保持）
# ---------------------------------------------------------------------------
def test_正常系_複数Toolの登録とAgent構築() -> None:
    """3 つの Tool を登録し AgentSpec.tools に渡すと、Agent.tools に順序保持で反映される。"""
    tool_registry = ToolRegistry()
    tool_registry.register(ToolSpec(func=_get_weather, name="get_weather"))
    tool_registry.register(ToolSpec(func=_get_time, name="get_time"))
    tool_registry.register(ToolSpec(func=_get_news, name="get_news"))

    spec = AgentSpec(
        name="assistant",
        instructions="test",
        tools=[
            tool_registry.get_weather,
            tool_registry.get_time,
            tool_registry.get_news,
        ],
    )
    agent_registry = AgentRegistry()
    agent_registry.register(spec)

    agent = agent_registry.get("assistant")
    assert len(agent.tools) == 3
    assert [t.name for t in agent.tools] == ["get_weather", "get_time", "get_news"]
    assert agent.tools[0] is tool_registry.get_weather
    assert agent.tools[1] is tool_registry.get_time
    assert agent.tools[2] is tool_registry.get_news
