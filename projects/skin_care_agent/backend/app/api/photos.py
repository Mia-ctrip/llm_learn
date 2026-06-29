from __future__ import annotations

import io
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from PIL import Image, UnidentifiedImageError
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.session import get_db
from app.models.photo import Photo
from app.models.user import User
from app.schemas.photo import PhotoUploadResponse
from app.services.storage_service import get_storage


router = APIRouter(prefix="/photos", tags=["photos"])


_MIME_TO_EXT = {
    "image/jpeg": "jpg",
    "image/png": "png",
    "image/webp": "webp",
}


def _ensure_seed_user(db: Session) -> User:
    """MVP 阶段：未接微信登录前，所有请求挂到 user_id=1。"""
    user = db.get(User, 1)
    if user is None:
        user = User(id=1, nickname="dev")
        db.add(user)
        db.flush()
    return user


def _build_storage_key(user_id: int, ext: str, now: datetime) -> str:
    return (
        f"photos/{user_id}/"
        f"{now.year:04d}/{now.month:02d}/{now.day:02d}/"
        f"{uuid.uuid4().hex}.{ext}"
    )


@router.post(
    "",
    response_model=PhotoUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_photo(
    file: UploadFile = File(...),
    taken_at: datetime | None = Form(default=None),
    db: Session = Depends(get_db),
):
    settings = get_settings()

    if file.content_type not in settings.allowed_mime_set:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"unsupported mime: {file.content_type}",
        )

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty file")
    if len(data) > settings.upload_max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"file too large: {len(data)} > {settings.upload_max_bytes}",
        )

    try:
        with Image.open(io.BytesIO(data)) as img:
            img.verify()
        with Image.open(io.BytesIO(data)) as img:
            width, height = img.size
    except (UnidentifiedImageError, OSError) as e:
        raise HTTPException(status_code=400, detail=f"invalid image: {e}") from e

    user = _ensure_seed_user(db)

    now = datetime.now(tz=timezone.utc)
    ext = _MIME_TO_EXT[file.content_type]
    key = _build_storage_key(user.id, ext, now)

    storage = get_storage()
    storage.put(key, data, file.content_type)

    photo = Photo(
        user_id=user.id,
        storage_key=key,
        mime_type=file.content_type,
        size_bytes=len(data),
        width=width,
        height=height,
        taken_at=taken_at,
    )
    db.add(photo)
    db.commit()
    db.refresh(photo)

    signed = storage.signed_url(key)

    return PhotoUploadResponse(
        photo_id=photo.id,
        storage_key=photo.storage_key,
        mime_type=photo.mime_type,
        size_bytes=photo.size_bytes,
        width=photo.width,
        height=photo.height,
        taken_at=photo.taken_at,
        url=signed.url,
        url_expires_at=signed.expires_at,
    )


@router.get("/{photo_id}/url")
def get_photo_url(photo_id: int, db: Session = Depends(get_db)):
    photo = db.get(Photo, photo_id)
    if photo is None or photo.deleted_at is not None:
        raise HTTPException(status_code=404, detail="photo not found")
    signed = get_storage().signed_url(photo.storage_key)
    return {"url": signed.url, "expires_at": signed.expires_at}
