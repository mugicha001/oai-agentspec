"""文書検索 Tool。"""

from __future__ import annotations

from oai_agentspec import ToolSpec

from ._registry import tool_registry


async def search_docs(query: str) -> str:
    """ドキュメント検索。"""
    return f"3 results for {query!r}"


tool_registry.register(ToolSpec(func=search_docs, timeout=5.0))
