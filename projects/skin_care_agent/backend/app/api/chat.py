from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.chat_message import ChatMessage
from app.schemas.chat import ChatMessageOut, ChatRequest, ChatResponse
from app.services import chat_service
from app.services.ai_gateway import rate_limit as rl


router = APIRouter(prefix="/chat", tags=["chat"])


_SEED_USER_ID = 1  # MVP：未接微信登录前所有请求挂到 user_id=1


@router.post("", response_model=ChatResponse, status_code=status.HTTP_201_CREATED)
async def post_chat(
    body: ChatRequest,
    response: Response,
    db: Session = Depends(get_db),
) -> ChatResponse:
    # 医疗兜底命中的情况也占额（避免有人拿关键词绕限流）
    try:
        rl.require(db, user_id=_SEED_USER_ID, kind="chat")
    except rl.QuotaExceeded as e:
        raise HTTPException(
            status_code=429,
            detail={
                "message": "daily quota exceeded",
                "kind": e.result.kind,
                "used": e.result.used,
                "limit": e.result.limit,
            },
        ) from e

    analysis_id = body.context.latest_analysis_id if body.context else None
    history = [h.model_dump() for h in body.history]

    try:
        result = await chat_service.send_chat(
            db,
            user_id=_SEED_USER_ID,
            user_message=body.message,
            analysis_id=analysis_id,
            history=history,
        )
    except chat_service.ChatFailed as e:
        if e.trace_id:
            response.headers["X-Trace-Id"] = e.trace_id
        raise HTTPException(
            status_code=502,
            detail={
                "message": e.message,
                "status": e.status,
                "log_id": e.log_id,
                "trace_id": e.trace_id,
            },
        ) from e

    response.headers["X-Trace-Id"] = result.trace_id
    return ChatResponse(
        chat_id=result.chat.id,
        assistant_message=result.chat.assistant_message,
        provider=result.chat.provider,
        model=result.chat.model,
        medical_intervention=result.medical_intervention,
        compliance_flags=result.chat.compliance_flags,
        created_at=result.chat.created_at,
        trace_id=result.trace_id,
    )


@router.get("/history", response_model=list[ChatMessageOut])
def list_chat_history(
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
) -> list[ChatMessageOut]:
    """当前 seed user 的问答历史，倒序。"""
    rows = (
        db.query(ChatMessage)
        .filter(ChatMessage.user_id == _SEED_USER_ID, ChatMessage.deleted_at.is_(None))
        .order_by(ChatMessage.id.desc())
        .limit(limit)
        .all()
    )
    return [
        ChatMessageOut(
            chat_id=r.id,
            user_message=r.user_message,
            assistant_message=r.assistant_message,
            provider=r.provider,
            model=r.model,
            medical_intervention=r.medical_intervention,
            analysis_id=r.analysis_id,
            created_at=r.created_at,
        )
        for r in rows
    ]
