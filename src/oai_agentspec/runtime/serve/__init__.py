"""会話サーバ入口（FastAPI REST + WebSocket・serve extra・agents 非依存）。

会話コアサービス（`ConversationService`）へ委譲する薄いサーバ入口を提供する。fastapi /
uvicorn は serve extra のため、本サブパッケージは本体 `oai_agentspec.__init__` から強制
import されない（独立サブパッケージとして明示 import される）。SDK（`agents`）は import
しない（NFR-1）。
"""

from __future__ import annotations

from .app import DEFAULT_HOST, DEFAULT_PORT, create_app, start_server

__all__ = [
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "create_app",
    "start_server",
]
