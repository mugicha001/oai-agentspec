"""REST ルータ（一覧 / 作成 / 非ストリーミング会話 / 承認・却下・serve extra・agents 非依存）。

会話サービスへ委譲する REST エンドポイント 7 本を登録した `APIRouter` を構築する
`_build_rest_router` を提供する。plain<->schema 変換は `..mappers`、依存解決は
`..dependencies` に委譲する。SDK（`agents`）は import しない。
"""

from __future__ import annotations

from fastapi import APIRouter

from ..dependencies import ServiceDep
from ..mappers import _approvals_response, _decisions_from_request, _send_response
from ..schemas import (
    AgentsResponse,
    ApprovalRequest,
    ApprovalsResponse,
    CreateConversationRequest,
    CreateConversationResponse,
    HistoryResponse,
    SendRequest,
    SendResponse,
    SessionMeta,
    SessionsResponse,
)


def _build_rest_router() -> APIRouter:
    """REST ルータ（一覧 / 作成 / 非ストリーミング会話）を構築する。

    Returns:
        REST エンドポイントを登録した APIRouter。
    """
    router = APIRouter()

    @router.get("/agents", response_model=AgentsResponse)
    async def list_agents(request_service: ServiceDep) -> AgentsResponse:
        """登録済みエージェント名の一覧とエントリエージェントを返す。"""
        return AgentsResponse(agents=request_service.agents(), entry=request_service.entry_agent())

    @router.get("/sessions", response_model=SessionsResponse)
    async def list_sessions(request_service: ServiceDep) -> SessionsResponse:
        """永続化済み session のメタ情報一覧を返す（D5・更新時刻降順）。"""
        infos = await request_service.list_sessions()
        return SessionsResponse(
            sessions=[
                SessionMeta(
                    session_id=info.session_id,
                    updated_at=info.updated_at,
                    turn_count=info.turn_count,
                    preview=info.preview,
                )
                for info in infos
            ]
        )

    @router.get("/sessions/{session_id}/history", response_model=HistoryResponse)
    async def session_history(
        session_id: str, request_service: ServiceDep, limit: int | None = None
    ) -> HistoryResponse:
        """指定 session の過去履歴アイテムを時系列で返す（復元時の表示用・D5）。

        `limit` 未指定時はサービス既定（直近 `DEFAULT_HISTORY_LIMIT` 件）を使う。
        """
        if limit is None:
            items = await request_service.session_history(session_id)
        else:
            items = await request_service.session_history(session_id, limit=limit)
        return HistoryResponse(session_id=session_id, items=items)

    @router.post(
        "/conversations",
        response_model=CreateConversationResponse,
        status_code=201,
    )
    async def create_conversation(
        body: CreateConversationRequest, request_service: ServiceDep
    ) -> CreateConversationResponse:
        """新規会話を作成し conversation_id を返す。"""
        cid = await request_service.create_conversation(
            conversation_id=body.conversation_id, session_id=body.session_id
        )
        return CreateConversationResponse(conversation_id=cid)

    @router.post("/conversations/{conversation_id}/messages", response_model=SendResponse)
    async def send_message(
        conversation_id: str, body: SendRequest, request_service: ServiceDep
    ) -> SendResponse:
        """非ストリーミングで 1 ターン会話し最終応答 or 承認待ちを返す（D-Disc）。

        承認必須ツールの呼び出しが発生したターンは `status="pending"` + 承認待ち一覧を返し、
        承認待ちが無いターンは従来どおり `status="final"` + 最終応答テキストを返す（NFR-6）。
        """
        result = await request_service.send(
            body.agent_name, body.text, conversation_id=conversation_id
        )
        return _send_response(result)

    @router.get("/conversations/{conversation_id}/approvals", response_model=ApprovalsResponse)
    async def get_approvals(conversation_id: str, request_service: ServiceDep) -> ApprovalsResponse:
        """現在の承認待ち一覧を返す（冪等・復元直後の再提示用・D-RestGet）。"""
        pending = await request_service.pending_approvals(conversation_id)
        return _approvals_response(conversation_id, pending)

    @router.post("/conversations/{conversation_id}/approvals", response_model=SendResponse)
    async def resolve_approvals(
        conversation_id: str, body: ApprovalRequest, request_service: ServiceDep
    ) -> SendResponse:
        """承認/却下を call_id 単位で適用し、再開後の最終応答 or 残承認待ちを返す（D-WsMsg）。

        全解決なら再開して `status="final"`、部分解決（未指定 call_id 残）なら再開せず
        `status="pending"` で残承認待ちを返す（段階解決・FR-7）。
        """
        result = await request_service.resolve_approvals(
            conversation_id, _decisions_from_request(body)
        )
        return _send_response(result)

    return router
