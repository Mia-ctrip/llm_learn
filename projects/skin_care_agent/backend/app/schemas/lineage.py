from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel


class LineageSnapshotOut(BaseModel):
    snapshot_id: int
    analysis_id: int
    photo_id: int
    patch_id: str
    bbox_norm: list[float]
    area_ratio: float
    coverage: str
    dominant_type: str
    estimated_count: int
    inflammation: str
    severity: int
    match_info: Optional[dict[str, Any]] = None
    created_at: datetime


class LineageOut(BaseModel):
    lineage_id: int
    region: str
    status: str
    first_seen_at: datetime
    last_seen_at: datetime
    snapshot_count: int
    duration_days: int
    latest: Optional[LineageSnapshotOut] = None


class LineageDetailOut(LineageOut):
    snapshots: list[LineageSnapshotOut]
    trend: dict[str, Any]
