"""通知送信 Tool。冪等でない副作用があるため HITL 承認必須で登録する。"""

from __future__ import annotations

from oai_agentspec import ToolSpec

from ._registry import tool_registry


async def send_notification(to: str, message: str) -> str:
    """通知送信（副作用あり）。"""
    return f"notified {to}: {message}"


tool_registry.register(ToolSpec(func=send_notification, needs_approval=True, timeout=15.0))
