"""FastAPI 依存解決（serve extra・agents 非依存）。

REST エンドポイントが会話サービスを受け取るための依存性プロバイダ `_get_service` と、その
`Annotated[..., Depends(...)]` 型 `ServiceDep` を提供する。会話サービスは `app.state.service`
から取り出す。SDK（`agents`）は import しない。
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from ..conversation import ConversationService


def _get_service(request: Request) -> ConversationService:
    """リクエストから会話サービスを取り出す（FastAPI 依存性プロバイダ）。

    Args:
        request: FastAPI リクエスト（`app.state.service` を持つ）。

    Returns:
        委譲先の会話コアサービス。
    """
    return request.app.state.service


# REST エンドポイントが会話サービスを受け取るための依存注入型。
ServiceDep = Annotated[ConversationService, Depends(_get_service)]
