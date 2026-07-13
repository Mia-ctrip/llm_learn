from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.analysis import Analysis
from app.models.patch_lineage import PatchLineage, PatchLineageSnapshot
from app.schemas.trend import DailyPoint, RegionSummary, TrendSummaryOut
from app.services.ai_gateway.schema import REGION_ZH


router = APIRouter(prefix="/trends", tags=["trends"])


_SEED_USER_ID = 1


def _today_utc() -> date:
    return datetime.now(tz=timezone.utc).date()


@router.get("/summary", response_model=TrendSummaryOut)
def trend_summary(
    days: int = Query(default=30, ge=1, le=365),
    db: Session = Depends(get_db),
) -> TrendSummaryOut:
    """当前用户在最近 N 天的趋势总览。

    合并 analyses（皮肤指数曲线）+ lineages（区域生命周期）两条数据源。
    """
    end_date = _today_utc()
    start_date = end_date - timedelta(days=days - 1)
    start_dt = datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc)
    end_dt = datetime.combine(end_date, datetime.max.time(), tzinfo=timezone.utc)

    # 1. Analyses 日聚合
    analyses = list(
        db.execute(
            select(Analysis)
            .where(
                Analysis.user_id == _SEED_USER_ID,
                Analysis.deleted_at.is_(None),
                Analysis.created_at >= start_dt,
                Analysis.created_at <= end_dt,
            )
            .order_by(Analysis.created_at.asc())
        ).scalars()
    )

    by_day: dict[date, list[Analysis]] = defaultdict(list)
    for a in analyses:
        d = a.created_at.astimezone(timezone.utc).date()
        by_day[d].append(a)

    daily_points: list[DailyPoint] = []
    cur = start_date
    while cur <= end_date:
        rows = by_day.get(cur, [])
        if rows:
            sev_vals = [r.overall_severity for r in rows if r.overall_severity is not None]
            idx_vals = [r.skin_health_index for r in rows if r.skin_health_index is not None]
            total_count = 0
            for r in rows:
                patches = (r.parsed_result or {}).get("acne_patches") or []
                total_count += sum(int(p.get("estimated_count") or 0) for p in patches)
            daily_points.append(
                DailyPoint(
                    day=cur,
                    overall_severity=(sum(sev_vals) / len(sev_vals)) if sev_vals else None,
                    skin_health_index=(sum(idx_vals) / len(idx_vals)) if idx_vals else None,
                    total_estimated_count=total_count,
                    analysis_count=len(rows),
                )
            )
        else:
            daily_points.append(
                DailyPoint(
                    day=cur,
                    overall_severity=None,
                    skin_health_index=None,
                    total_estimated_count=0,
                    analysis_count=0,
                )
            )
        cur += timedelta(days=1)

    # 2. Lineage 统计
    lineages = list(
        db.execute(
            select(PatchLineage).where(
                PatchLineage.user_id == _SEED_USER_ID,
                PatchLineage.deleted_at.is_(None),
            )
        ).scalars()
    )

    total_active = sum(1 for lineage in lineages if lineage.status == "active")
    total_new_in_range = sum(1 for lineage in lineages if lineage.first_seen_at >= start_dt)
    # healed_at 我们没单独记，用 status=healed 且 last_seen_at 在窗口内近似
    total_healed_in_range = sum(
        1 for lineage in lineages if lineage.status == "healed" and lineage.last_seen_at >= start_dt
    )

    # 按 region 分组
    region_groups: dict[str, list[PatchLineage]] = defaultdict(list)
    for lineage in lineages:
        region_groups[lineage.region].append(lineage)

    # 拿每 region 的 latest snapshot 用于展示 dominant_type/coverage
    all_ids = [lineage.id for lineage in lineages]
    latest_by_lineage: dict[int, PatchLineageSnapshot] = {}
    if all_ids:
        for snap in db.execute(
            select(PatchLineageSnapshot)
            .where(PatchLineageSnapshot.lineage_id.in_(all_ids))
            .order_by(
                PatchLineageSnapshot.lineage_id,
                PatchLineageSnapshot.created_at.desc(),
            )
        ).scalars():
            if snap.lineage_id not in latest_by_lineage:
                latest_by_lineage[snap.lineage_id] = snap

    region_summaries: list[RegionSummary] = []
    for region, group in region_groups.items():
        active = [lineage for lineage in group if lineage.status == "active"]
        dormant = [lineage for lineage in group if lineage.status == "dormant"]
        healed = [lineage for lineage in group if lineage.status == "healed"]
        # 最新 snapshot：group 里 last_seen_at 最大的 lineage 的 latest snap
        latest_lineage = max(group, key=lambda lineage: lineage.last_seen_at)
        latest_snap = latest_by_lineage.get(latest_lineage.id)
        region_summaries.append(
            RegionSummary(
                region=region,
                active_lineage_count=len(active),
                dormant_lineage_count=len(dormant),
                healed_lineage_count=len(healed),
                latest_dominant_type=latest_snap.dominant_type if latest_snap else None,
                latest_coverage=latest_snap.coverage if latest_snap else None,
            )
        )
    region_summaries.sort(key=lambda r: -r.active_lineage_count)

    # 3. Highlights：给用户看的一句话洞察
    highlights = _build_highlights(
        daily_points=daily_points,
        total_new=total_new_in_range,
        total_healed=total_healed_in_range,
        region_summaries=region_summaries,
        days=days,
    )

    return TrendSummaryOut(
        range_days=days,
        start_date=start_date,
        end_date=end_date,
        total_analyses=len(analyses),
        total_active_lineages=total_active,
        total_new_lineages_in_range=total_new_in_range,
        total_healed_lineages_in_range=total_healed_in_range,
        daily_points=daily_points,
        region_summaries=region_summaries,
        highlights=highlights,
    )


def _build_highlights(
    *,
    daily_points: list[DailyPoint],
    total_new: int,
    total_healed: int,
    region_summaries: list[RegionSummary],
    days: int,
) -> list[str]:
    out: list[str] = []
    valid_idx = [p for p in daily_points if p.skin_health_index is not None]
    if len(valid_idx) >= 2:
        first_idx = valid_idx[0].skin_health_index or 0
        last_idx = valid_idx[-1].skin_health_index or 0
        delta = last_idx - first_idx
        if delta >= 5:
            out.append(f"皮肤指数近 {days} 天上升 {delta:.0f} 分（{first_idx:.0f} → {last_idx:.0f}），趋势向好。")
        elif delta <= -5:
            out.append(f"皮肤指数近 {days} 天下降 {abs(delta):.0f} 分（{first_idx:.0f} → {last_idx:.0f}），需要注意护理。")
        else:
            out.append(f"皮肤指数在 {days} 天内保持稳定（波动 {delta:+.0f} 分）。")

    if total_new > 0:
        out.append(f"近 {days} 天新增 {total_new} 处痘斑区域。")
    if total_healed > 0:
        out.append(f"近 {days} 天有 {total_healed} 处痘斑区域已消退。")

    if region_summaries:
        top = region_summaries[0]
        if top.active_lineage_count > 0:
            zh = REGION_ZH.get(top.region, top.region)
            out.append(
                f"当前活跃痘斑最多的区域：{zh}（{top.active_lineage_count} 处活跃）。"
            )

    if not out:
        out.append("数据不足，请继续追踪几日后查看趋势。")
    return out
