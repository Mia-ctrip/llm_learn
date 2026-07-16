from __future__ import annotations

import io
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from PIL import Image, ImageOps, UnidentifiedImageError
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.session import get_db
from app.models.check_in import CheckIn
from app.models.photo import Photo
from app.schemas.photo import PhotoUploadResponse
from app.services.storage_service import get_storage
from app.services.user_service import SEED_USER_ID, ensure_seed_user
from app.services.vision.normalization import normalize_photo_for_analysis
from app.services.vision.quality import (
    QualityModelUnavailable,
    assess_photo_quality,
)


router = APIRouter(prefix="/photos", tags=["photos"])

_MIME_TO_EXT = {
    "image/jpeg": "jpg",
    "image/png": "png",
    "image/webp": "webp",
}
_VALID_VIEW_TYPES = {"front", "left", "right"}


def _build_storage_key(user_id: int, ext: str, now: datetime) -> str:
    return (
        f"photos/{user_id}/"
        f"{now.year:04d}/{now.month:02d}/{now.day:02d}/"
        f"{uuid.uuid4().hex}.{ext}"
    )


def _validate_check_in_target(
    db: Session,
    *,
    check_in_id: int | None,
    view_type: str | None,
) -> CheckIn | None:
    if check_in_id is None:
        if view_type is not None:
            raise HTTPException(status_code=400, detail="view_type requires check_in_id")
        return None
    if view_type not in _VALID_VIEW_TYPES:
        raise HTTPException(
            status_code=422,
            detail={"message": "invalid view_type", "allowed": sorted(_VALID_VIEW_TYPES)},
        )
    check_in = db.get(CheckIn, check_in_id)
    if (
        check_in is None
        or check_in.deleted_at is not None
        or check_in.user_id != SEED_USER_ID
    ):
        raise HTTPException(status_code=404, detail="check-in not found")
    if check_in.status != "draft":
        raise HTTPException(status_code=409, detail="completed check-in cannot accept photos")
    return check_in


@router.post(
    "",
    response_model=PhotoUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_photo(
    file: UploadFile = File(...),
    taken_at: datetime | None = Form(default=None),
    check_in_id: int | None = Form(default=None),
    view_type: str | None = Form(default=None),
    db: Session = Depends(get_db),
) -> PhotoUploadResponse:
    settings = get_settings()
    _validate_check_in_target(db, check_in_id=check_in_id, view_type=view_type)

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
            width, height = ImageOps.exif_transpose(img).size
    except (UnidentifiedImageError, OSError) as e:
        raise HTTPException(status_code=400, detail=f"invalid image: {e}") from e

    quality_result = None
    normalized_photo = None
    quality_meta = None
    if check_in_id is not None:
        try:
            quality_result = assess_photo_quality(data, view_type=view_type)
        except QualityModelUnavailable as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "message": "photo quality model unavailable",
                    "error": str(e),
                },
            ) from e
        if not quality_result.passed:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail={
                    "message": "photo quality check failed",
                    "errors": list(quality_result.errors),
                    "quality_meta": quality_result.to_meta(),
                },
            )
        normalized_photo = normalize_photo_for_analysis(data, quality_result)
        quality_meta = quality_result.to_meta()
        quality_meta["normalization"] = normalized_photo.to_meta()

    user = ensure_seed_user(db)
    now = datetime.now(tz=timezone.utc)
    ext = _MIME_TO_EXT[file.content_type]
    key = _build_storage_key(user.id, ext, now)
    processed_key = (
        f"{key.rsplit('.', 1)[0]}.normalized.jpg" if normalized_photo else None
    )

    if check_in_id is not None and view_type is not None:
        existing = (
            db.query(Photo)
            .filter(
                Photo.check_in_id == check_in_id,
                Photo.view_type == view_type,
                Photo.deleted_at.is_(None),
            )
            .first()
        )
        if existing is not None:
            existing.deleted_at = now
            db.flush()

    storage = get_storage()
    try:
        storage.put(key, data, file.content_type)
        if normalized_photo is not None and processed_key is not None:
            storage.put(processed_key, normalized_photo.data, "image/jpeg")
    except Exception:
        db.rollback()
        storage.delete(key)
        if processed_key is not None:
            storage.delete(processed_key)
        raise

    photo = Photo(
        user_id=user.id,
        check_in_id=check_in_id,
        view_type=view_type,
        quality_status=quality_result.status if quality_result else None,
        quality_meta=quality_meta,
        storage_key=key,
        processed_storage_key=processed_key,
        mime_type=file.content_type,
        size_bytes=len(data),
        width=width,
        height=height,
        taken_at=taken_at,
    )
    db.add(photo)
    try:
        db.commit()
    except Exception:
        db.rollback()
        storage.delete(key)
        if processed_key is not None:
            storage.delete(processed_key)
        raise
    db.refresh(photo)

    signed = storage.signed_url(key)
    return PhotoUploadResponse(
        photo_id=photo.id,
        check_in_id=photo.check_in_id,
        view_type=photo.view_type,
        quality_status=photo.quality_status,
        quality_meta=photo.quality_meta,
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