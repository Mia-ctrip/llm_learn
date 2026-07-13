from __future__ import annotations

from datetime import date
from pydantic import BaseModel


class DailyPoint(BaseModel):
    day: date
    overall_severity: float | None = None
    skin_health_index: float | None = None
    total_estimated_count: int = 0
    analysis_count: int = 0


class RegionSummary(BaseModel):
    region: str
    active_lineage_count: int
    dormant_lineage_count: int
    healed_lineage_count: int
    latest_dominant_type: str | None = None
    latest_coverage: str | None = None


class TrendSummaryOut(BaseModel):
    range_days: int
    start_date: date
    end_date: date
    total_analyses: int
    total_active_lineages: int
    total_new_lineages_in_range: int
    total_healed_lineages_in_range: int
    daily_points: list[DailyPoint]
    region_summaries: list[RegionSummary]
    highlights: list[str]  # 人类可读的关键洞察
