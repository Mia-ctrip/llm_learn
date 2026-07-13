from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.patch_lineage import PatchLineage, PatchLineageSnapshot
from app.models.photo import Photo
from app.schemas.lineage import LineageDetailOut, LineageOut, LineageSnapshotOut


router = APIRouter(prefix="/lineages", tags=["lineages"])


_SEED_USER_ID = 1


def _snap_to_out(s: PatchLineageSnapshot) -> LineageSnapshotOut:
    return LineageSnapshotOut(
        snapshot_id=s.id,
        analysis_id=s.analysis_id,
        photo_id=s.photo_id,
        patch_id=s.patch_id,
        bbox_norm=s.bbox_norm,
        area_ratio=s.area_ratio,
        coverage=s.coverage,
        dominant_type=s.dominant_type,
        estimated_count=s.estimated_count,
        inflammation=s.inflammation,
        severity=s.severity,
        match_info=s.match_info,
        created_at=s.created_at,
    )


def _duration_days(lineage: PatchLineage) -> int:
    return max(0, (lineage.last_seen_at - lineage.first_seen_at).days)


def _summarize_trend(snapshots: list[PatchLineageSnapshot]) -> dict[str, Any]:
    """按时间序列出关键变化摘要。"""
    if not snapshots:
        return {}
    snapshots = sorted(snapshots, key=lambda s: s.created_at)
    first = snapshots[0]
    last = snapshots[-1]
    return {
        "count_delta": last.estimated_count - first.estimated_count,
        "severity_delta": last.severity - first.severity,
        "coverage_trajectory": [s.coverage for s in snapshots],
        "type_trajectory": [s.dominant_type for s in snapshots],
        "inflammation_start": first.inflammation,
        "inflammation_end": last.inflammation,
    }


@router.get("", response_model=list[LineageOut])
def list_lineages(
    status_filter: Optional[str] = Query(default=None, alias="status"),
    region: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
) -> list[LineageOut]:
    """当前用户的所有 lineage，按最新出现时间倒序。"""
    stmt = (
        select(PatchLineage)
        .where(
            PatchLineage.user_id == _SEED_USER_ID,
            PatchLineage.deleted_at.is_(None),
        )
        .order_by(PatchLineage.last_seen_at.desc())
        .limit(limit)
    )
    if status_filter:
        stmt = stmt.where(PatchLineage.status == status_filter)
    if region:
        stmt = stmt.where(PatchLineage.region == region)

    lineages = list(db.execute(stmt).scalars())
    if not lineages:
        return []

    # 一次查所有 lineage 的最新 snapshot（O(N) 而不是 N+1）
    ids = [lineage.id for lineage in lineages]
    snap_stmt = (
        select(PatchLineageSnapshot)
        .where(PatchLineageSnapshot.lineage_id.in_(ids))
        .order_by(
            PatchLineageSnapshot.lineage_id,
            PatchLineageSnapshot.created_at.desc(),
        )
    )
    latest_by_lineage: dict[int, PatchLineageSnapshot] = {}
    for snap in db.execute(snap_stmt).scalars():
        if snap.lineage_id not in latest_by_lineage:
            latest_by_lineage[snap.lineage_id] = snap

    return [
        LineageOut(
            lineage_id=lineage.id,
            region=lineage.region,
            status=lineage.status,
            first_seen_at=lineage.first_seen_at,
            last_seen_at=lineage.last_seen_at,
            snapshot_count=lineage.snapshot_count,
            duration_days=_duration_days(lineage),
            latest=_snap_to_out(latest_by_lineage[lineage.id])
            if lineage.id in latest_by_lineage
            else None,
        )
        for lineage in lineages
    ]


@router.get("/{lineage_id}", response_model=LineageDetailOut)
def get_lineage(lineage_id: int, db: Session = Depends(get_db)) -> LineageDetailOut:
    """一条 lineage 的完整时间线。"""
    lineage = db.get(PatchLineage, lineage_id)
    if lineage is None or lineage.deleted_at is not None or lineage.user_id != _SEED_USER_ID:
        raise HTTPException(status_code=404, detail="lineage not found")

    snaps = list(
        db.execute(
            select(PatchLineageSnapshot)
            .where(PatchLineageSnapshot.lineage_id == lineage_id)
            .order_by(PatchLineageSnapshot.created_at.asc())
        ).scalars()
    )

    return LineageDetailOut(
        lineage_id=lineage.id,
        region=lineage.region,
        status=lineage.status,
        first_seen_at=lineage.first_seen_at,
        last_seen_at=lineage.last_seen_at,
        snapshot_count=lineage.snapshot_count,
        duration_days=_duration_days(lineage),
        latest=_snap_to_out(snaps[-1]) if snaps else None,
        snapshots=[_snap_to_out(s) for s in snaps],
        trend=_summarize_trend(snaps),
    )


@router.get("/by-photo/{photo_id}", response_model=list[LineageOut])
def list_lineages_by_photo(
    photo_id: int, db: Session = Depends(get_db)
) -> list[LineageOut]:
    """一张照片的所有 patch 对应的 lineage（用于展示"这张照片的每片痘斑是从哪来的"）。"""
    photo = db.get(Photo, photo_id)
    if photo is None or photo.deleted_at is not None:
        raise HTTPException(status_code=404, detail="photo not found")

    # 找该 photo 相关的所有 snapshots → 拿唯一的 lineage_ids
    snaps = list(
        db.execute(
            select(PatchLineageSnapshot)
            .where(PatchLineageSnapshot.photo_id == photo_id)
            .order_by(PatchLineageSnapshot.created_at.desc())
        ).scalars()
    )
    if not snaps:
        return []

    lineage_ids = list({s.lineage_id for s in snaps})
    lineages = list(
        db.execute(
            select(PatchLineage).where(PatchLineage.id.in_(lineage_ids))
        ).scalars()
    )

    latest_by_lineage: dict[int, PatchLineageSnapshot] = {}
    for snap in db.execute(
        select(PatchLineageSnapshot)
        .where(PatchLineageSnapshot.lineage_id.in_(lineage_ids))
        .order_by(
            PatchLineageSnapshot.lineage_id,
            PatchLineageSnapshot.created_at.desc(),
        )
    ).scalars():
        if snap.lineage_id not in latest_by_lineage:
            latest_by_lineage[snap.lineage_id] = snap

    return [
        LineageOut(
            lineage_id=lineage.id,
            region=lineage.region,
            status=lineage.status,
            first_seen_at=lineage.first_seen_at,
            last_seen_at=lineage.last_seen_at,
            snapshot_count=lineage.snapshot_count,
            duration_days=_duration_days(lineage),
            latest=_snap_to_out(latest_by_lineage[lineage.id])
            if lineage.id in latest_by_lineage
            else None,
        )
        for lineage in lineages
    ]
