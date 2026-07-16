from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.check_in import CheckIn
from app.models.photo import Photo
from app.schemas.check_in import (
    CheckInAnalysisSummaryOut,
    CheckInCreate,
    CheckInDiary,
    CheckInOut,
    CheckInPhotoOut,
)
from app.services.check_in_aggregation import load_check_in_summary
from app.services.storage_service import get_storage
from app.services.user_service import SEED_USER_ID, ensure_seed_user


router = APIRouter(prefix="/check-ins", tags=["check-ins"])
logger = logging.getLogger(__name__)

_REQUIRED_VIEWS: dict[str, frozenset[str]] = {
    "quick": frozenset(),
    "standard": frozenset({"front", "left", "right"}),
}


def _missing_required_views(kind: str, present_views: set[str]) -> list[str]:
    required = _REQUIRED_VIEWS.get(kind, frozenset())
    return sorted(required - present_views)


def _load_check_in(db: Session, check_in_id: int) -> CheckIn:
    check_in = db.get(CheckIn, check_in_id)
    if check_in is None or check_in.deleted_at is not None or check_in.user_id != SEED_USER_ID:
        raise HTTPException(status_code=404, detail="check-in not found")
    return check_in


def _load_photos(db: Session, check_in_ids: list[int]) -> dict[int, list[Photo]]:
    if not check_in_ids:
        return {}
    rows = db.execute(
        select(Photo)
        .where(
            Photo.check_in_id.in_(check_in_ids),
            Photo.deleted_at.is_(None),
        )
        .order_by(Photo.check_in_id, Photo.id)
    ).scalars()
    result: dict[int, list[Photo]] = defaultdict(list)
    for photo in rows:
        if photo.check_in_id is not None:
            result[photo.check_in_id].append(photo)
    return result


def _serialize_diary(diary: CheckInDiary | None) -> dict[str, object] | None:
    if diary is None:
        return None
    data = diary.model_dump(mode="json", exclude_none=True)
    return data or None


def _to_out(check_in: CheckIn, photos: list[Photo]) -> CheckInOut:
    photo_rows: list[CheckInPhotoOut] = []
    for photo in photos:
        if photo.view_type not in {"front", "left", "right"}:
            continue
        storage = get_storage()
        signed = storage.signed_url(photo.storage_key)
        photo_rows.append(
            CheckInPhotoOut(
                photo_id=photo.id,
                view_type=photo.view_type,
                width=photo.width,
                height=photo.height,
                taken_at=photo.taken_at,
                quality_status=photo.quality_status,
                quality_meta=photo.quality_meta,
                url=signed.url,
                url_expires_at=signed.expires_at,
            )
        )
    return CheckInOut(
        check_in_id=check_in.id,
        kind=check_in.kind,
        status=check_in.status,
        observed_on=check_in.observed_on,
        completed_at=check_in.completed_at,
        created_at=check_in.created_at,
        diary=(
            CheckInDiary.model_validate(check_in.diary_data)
            if check_in.diary_data is not None
            else None
        ),
        diary_updated_at=check_in.diary_updated_at,
        photo_count=len(photo_rows),
        photos=photo_rows,
    )


@router.post("", response_model=CheckInOut, status_code=status.HTTP_201_CREATED)
def create_check_in(body: CheckInCreate, db: Session = Depends(get_db)) -> CheckInOut:
    user = ensure_seed_user(db)
    diary_data = _serialize_diary(body.diary)
    check_in = CheckIn(
        user_id=user.id,
        kind=body.kind,
        status="draft",
        observed_on=body.observed_on,
        diary_data=diary_data,
        diary_updated_at=(datetime.now(tz=timezone.utc) if diary_data is not None else None),
    )
    db.add(check_in)
    db.commit()
    db.refresh(check_in)
    return _to_out(check_in, [])


@router.get("", response_model=list[CheckInOut])
def list_check_ins(
    limit: int = Query(default=30, ge=1, le=100),
    db: Session = Depends(get_db),
) -> list[CheckInOut]:
    rows = list(
        db.execute(
            select(CheckIn)
            .where(CheckIn.user_id == SEED_USER_ID, CheckIn.deleted_at.is_(None))
            .order_by(CheckIn.observed_on.desc(), CheckIn.id.desc())
            .limit(limit)
        ).scalars()
    )
    photos_by_check_in = _load_photos(db, [row.id for row in rows])
    return [_to_out(row, photos_by_check_in.get(row.id, [])) for row in rows]


@router.get(
    "/{check_in_id}/analysis-summary",
    response_model=CheckInAnalysisSummaryOut,
)
def get_check_in_analysis_summary(
    check_in_id: int,
    db: Session = Depends(get_db),
) -> CheckInAnalysisSummaryOut:
    check_in = _load_check_in(db, check_in_id)
    return load_check_in_summary(db, check_in)


@router.get("/{check_in_id}", response_model=CheckInOut)
def get_check_in(check_in_id: int, db: Session = Depends(get_db)) -> CheckInOut:
    check_in = _load_check_in(db, check_in_id)
    photos = _load_photos(db, [check_in.id]).get(check_in.id, [])
    return _to_out(check_in, photos)


@router.put("/{check_in_id}/diary", response_model=CheckInOut)
def replace_check_in_diary(
    check_in_id: int,
    body: CheckInDiary,
    db: Session = Depends(get_db),
) -> CheckInOut:
    """完整替换一次 check-in 的日记；空对象会清空已有日记。"""

    check_in = _load_check_in(db, check_in_id)
    check_in.diary_data = _serialize_diary(body)
    check_in.diary_updated_at = datetime.now(tz=timezone.utc)
    db.commit()
    db.refresh(check_in)
    photos = _load_photos(db, [check_in.id]).get(check_in.id, [])
    return _to_out(check_in, photos)


@router.post("/{check_in_id}/complete", response_model=CheckInOut)
def complete_check_in(check_in_id: int, db: Session = Depends(get_db)) -> CheckInOut:
    check_in = _load_check_in(db, check_in_id)
    if check_in.status == "complete":
        raise HTTPException(status_code=409, detail="check-in already complete")

    photos = _load_photos(db, [check_in.id]).get(check_in.id, [])
    present_views = {photo.view_type for photo in photos if photo.view_type is not None}
    missing = _missing_required_views(check_in.kind, present_views)
    if missing:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "required photo views are missing",
                "missing_views": missing,
            },
        )

    check_in.status = "complete"
    check_in.completed_at = datetime.now(tz=timezone.utc)
    try:
        from app.services.vision.tracker import track_completed_check_in

        track_result = track_completed_check_in(db, check_in)
        logger.info(
            "check-in complete: id=%s new_lineages=%s matched_lineages=%s missing_observations=%s",
            check_in.id,
            track_result.new_lineage_count,
            track_result.matched_lineage_count,
            track_result.missing_observation_count,
        )
    except Exception as exc:  # noqa: BLE001
        db.rollback()
        logger.exception("check-in lifecycle tracking failed: id=%s", check_in.id)
        raise HTTPException(
            status_code=500,
            detail="check-in lifecycle tracking failed",
        ) from exc
    db.refresh(check_in)
    return _to_out(check_in, photos)
