"""天気照会 Tool。定義と登録を同一ファイルにまとめる分散登録の例。"""

from __future__ import annotations

from oai_agentspec import ToolSpec

from ._registry import tool_registry


async def get_weather(city: str) -> str:
    """指定都市の天気を返す（例では固定値）。"""
    return f"{city}: sunny, 22C"


tool_registry.register(ToolSpec(func=get_weather, timeout=10.0))
