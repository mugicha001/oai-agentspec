"""FastAPI 会話サーバの app factory（serve extra・agents 非依存）。

会話コアサービス（`ConversationService`）へ委譲する FastAPI app を組み立てる薄い factory
`create_app` を提供する。REST ルータ（`routers.rest`）/ WebSocket ルート（`routers.ws`）の
登録と、会話エラー（`ConversationError`）の exception_handler 登録のみを行う。エラー写像は
`errors`、plain<->schema 変換は `mappers`、依存解決は `dependencies` に分離する。

起動入口（`start_server`）と既定バインド先（`DEFAULT_HOST` / `DEFAULT_PORT`）は `server` へ
移したが、従来の import 経路（`from oai_agentspec.runtime.serve.app import start_server` 等）を
保つため本モジュールから再エクスポートする。

SDK（`agents`）は import しない。fastapi / uvicorn は serve extra のため、本体 `__init__` からは
強制 import しない（本サブパッケージは独立サブパッケージとして遅延 import される）。
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from ..conversation import ConversationError, ConversationService
from .errors import _error_status
from .routers.rest import _build_rest_router
from .routers.ws import _register_ws_route
from .schemas import ErrorResponse
from .server import DEFAULT_HOST, DEFAULT_PORT, start_server

__all__ = [
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "create_app",
    "start_server",
]


def create_app(service: ConversationService) -> FastAPI:
    """会話サービスへ委譲する FastAPI app を構築する。

    Args:
        service: 委譲先の会話コアサービス。

    Returns:
        REST + WebSocket ルータを登録した FastAPI app。
    """
    app = FastAPI(title="oai-agentspec conversation server")
    app.state.service = service

    @app.exception_handler(ConversationError)
    async def _conversation_error_handler(_request: object, exc: ConversationError) -> JSONResponse:
        """`ConversationError` を status + 原因ボディの JSON へ変換する。"""
        body = ErrorResponse(code=exc.code.value, message=exc.message)
        return JSONResponse(status_code=_error_status(exc.code), content=body.model_dump())

    app.include_router(_build_rest_router())
    _register_ws_route(app)
    return app
