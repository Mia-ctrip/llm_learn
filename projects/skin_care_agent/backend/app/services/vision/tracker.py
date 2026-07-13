"""Patch lineage 追踪算法。

设计（Q1=B / Q2=A / Q3=A）：
- 一个 region 可能有多条 lineage
- 只做 patch 追踪，point 不追踪
- 匹配算法简单版：同 region 内 bbox 中心距离最小的作为候选，超阈值则新建

生命周期状态：
- active：last_seen_at 距今 <= 1 天
- dormant：1-14 天没出现
- healed：>14 天没出现（不再自动接续）

调用时机：`analysis_service.analyze_photo` 落库 analyses 后立即调用。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.analysis import Analysis
from app.models.patch_lineage import PatchLineage, PatchLineageSnapshot


logger = logging.getLogger(__name__)


# 匹配阈值（bbox 中心归一化距离，0~sqrt(2)）
MATCH_DISTANCE_THRESHOLD = 0.25

# 生命周期时间阈值（天）
DORMANT_AFTER_DAYS = 1
HEALED_AFTER_DAYS = 14

# 候选 lineage 查询窗口（不再考虑超过这个天数的 active/dormant）
CANDIDATE_LOOKBACK_DAYS = 14


@dataclass
class TrackResult:
    """一次分析的 tracker 输出。"""

    new_lineage_count: int
    matched_lineage_count: int
    snapshot_ids: list[int]


def _bbox_center(bbox: list[float]) -> tuple[float, float]:
    """[x1,y1,x2,y2] → (cx, cy)。"""
    if not bbox or len(bbox) != 4:
        return (0.0, 0.0)
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _distance(c1: tuple[float, float], c2: tuple[float, float]) -> float:
    return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def track_patches_for_analysis(
    db: Session, analysis: Analysis
) -> TrackResult:
    """对刚落库的 analysis 做 patch lineage 匹配 + 快照落表。"""
    parsed = analysis.parsed_result or {}
    patches = parsed.get("acne_patches") or []
    now = _now()

    if not patches:
        logger.info(
            "tracker: no patches for analysis_id=%s user_id=%s",
            analysis.id,
            analysis.user_id,
        )
        _update_stale_lineages(db, user_id=analysis.user_id, now=now)
        db.commit()
        return TrackResult(new_lineage_count=0, matched_lineage_count=0, snapshot_ids=[])

    # 拉候选 lineage：同 user + 近 14 天的 active/dormant
    since = now - timedelta(days=CANDIDATE_LOOKBACK_DAYS)
    candidates_by_region: dict[str, list[PatchLineage]] = {}
    stmt = (
        select(PatchLineage)
        .where(
            PatchLineage.user_id == analysis.user_id,
            PatchLineage.deleted_at.is_(None),
            PatchLineage.status.in_(["active", "dormant"]),
            PatchLineage.last_seen_at >= since,
        )
        .order_by(PatchLineage.last_seen_at.desc())
    )
    for lineage in db.execute(stmt).scalars():
        candidates_by_region.setdefault(lineage.region, []).append(lineage)

    # 对每个候选 lineage 拉它最后一条 snapshot，用来提供 bbox 中心比对
    latest_snapshot_by_lineage: dict[int, PatchLineageSnapshot] = {}
    if candidates_by_region:
        all_ids = [lineage.id for group in candidates_by_region.values() for lineage in group]
        snap_stmt = (
            select(PatchLineageSnapshot)
            .where(PatchLineageSnapshot.lineage_id.in_(all_ids))
            .order_by(
                PatchLineageSnapshot.lineage_id, PatchLineageSnapshot.created_at.desc()
            )
        )
        # 每个 lineage 只保留最新一条
        for snap in db.execute(snap_stmt).scalars():
            if snap.lineage_id not in latest_snapshot_by_lineage:
                latest_snapshot_by_lineage[snap.lineage_id] = snap

    # 每条 lineage 一次分析最多匹配一次
    consumed_lineage_ids: set[int] = set()
    snapshots: list[PatchLineageSnapshot] = []
    matched_count = 0
    new_count = 0

    for patch in patches:
        region = patch.get("region", "unknown")
        patch_id = patch.get("id", "")
        bbox = patch.get("bbox_norm") or [0.0, 0.0, 0.0, 0.0]
        center = _bbox_center(bbox)

        # 在该 region 的候选里找最近的 lineage
        region_candidates = candidates_by_region.get(region, [])
        best_lineage: Optional[PatchLineage] = None
        best_distance = float("inf")

        for lineage in region_candidates:
            if lineage.id in consumed_lineage_ids:
                continue
            snap = latest_snapshot_by_lineage.get(lineage.id)
            if snap is None:
                continue
            d = _distance(center, _bbox_center(snap.bbox_norm))
            if d < best_distance:
                best_distance = d
                best_lineage = lineage

        # 决策：匹配还是新建
        if best_lineage is not None and best_distance <= MATCH_DISTANCE_THRESHOLD:
            lineage = best_lineage
            consumed_lineage_ids.add(lineage.id)
            lineage.last_seen_at = now
            lineage.status = "active"
            lineage.snapshot_count = (lineage.snapshot_count or 0) + 1
            matched_count += 1
            match_info: dict[str, Any] = {
                "matched": True,
                "distance": round(best_distance, 4),
                "threshold": MATCH_DISTANCE_THRESHOLD,
            }
        else:
            lineage = PatchLineage(
                user_id=analysis.user_id,
                region=region,
                status="active",
                first_seen_at=now,
                last_seen_at=now,
                snapshot_count=1,
            )
            db.add(lineage)
            db.flush()  # 拿到 id 用于关联 snapshot
            consumed_lineage_ids.add(lineage.id)
            candidates_by_region.setdefault(region, []).append(lineage)
            new_count += 1
            match_info = {
                "matched": False,
                "reason": (
                    "no_candidates" if not region_candidates
                    else f"nearest_distance={round(best_distance, 4)} > threshold={MATCH_DISTANCE_THRESHOLD}"
                ),
            }

        snapshot = PatchLineageSnapshot(
            lineage_id=lineage.id,
            analysis_id=analysis.id,
            photo_id=analysis.photo_id,
            user_id=analysis.user_id,
            patch_id=patch_id,
            region=region,
            bbox_norm=bbox,
            area_ratio=float(patch.get("area_ratio") or 0.0),
            coverage=patch.get("coverage", "sparse"),
            dominant_type=patch.get("dominant_type", "mixed"),
            estimated_count=int(patch.get("estimated_count") or 0),
            inflammation=patch.get("inflammation", "none"),
            severity=int(patch.get("severity") or 1),
            match_info=match_info,
            created_at=now,
        )
        db.add(snapshot)
        snapshots.append(snapshot)
        latest_snapshot_by_lineage[lineage.id] = snapshot

    # 未被 consume 的候选 lineage：更新状态（1-14 天没出现 → dormant）
    _update_stale_lineages(db, user_id=analysis.user_id, now=now)

    db.commit()
    for s in snapshots:
        db.refresh(s)

    result = TrackResult(
        new_lineage_count=new_count,
        matched_lineage_count=matched_count,
        snapshot_ids=[s.id for s in snapshots],
    )
    logger.info(
        "tracker: analysis_id=%s new=%s matched=%s snapshots=%s",
        analysis.id,
        new_count,
        matched_count,
        len(snapshots),
    )
    return result


def _update_stale_lineages(db: Session, *, user_id: int, now: datetime) -> None:
    """处理没出现在本次分析中的 lineage 的状态迁移。"""
    dormant_cutoff = now - timedelta(days=DORMANT_AFTER_DAYS)
    healed_cutoff = now - timedelta(days=HEALED_AFTER_DAYS)

    # active → dormant
    stmt = select(PatchLineage).where(
        PatchLineage.user_id == user_id,
        PatchLineage.deleted_at.is_(None),
        PatchLineage.status == "active",
        PatchLineage.last_seen_at < dormant_cutoff,
    )
    for lineage in db.execute(stmt).scalars():
        lineage.status = "dormant"

    # dormant → healed
    stmt = select(PatchLineage).where(
        PatchLineage.user_id == user_id,
        PatchLineage.deleted_at.is_(None),
        PatchLineage.status == "dormant",
        PatchLineage.last_seen_at < healed_cutoff,
    )
    for lineage in db.execute(stmt).scalars():
        lineage.status = "healed"
