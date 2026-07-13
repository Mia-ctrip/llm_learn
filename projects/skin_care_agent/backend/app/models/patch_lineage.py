from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from sqlalchemy import BigInteger, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, IdMixin, TimestampMixin


class PatchLineage(Base, IdMixin, TimestampMixin):
    """一条持续存在的痘斑病灶群的追踪主线。

    - 一个 region 可能同时存在多条 lineage（Q1=B）
    - 状态 active/healed/dormant：last_seen 超过 14 天 → healed；1-14 天 → dormant
    """

    __tablename__ = "patch_lineages"

    user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    region: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="active")

    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    snapshot_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class PatchLineageSnapshot(Base, IdMixin):
    """一次分析中，某个 patch 被匹配（或新建）到某条 lineage 的快照。

    冗余存 patch 属性避免每次要 join 回 analyses.parsed_result 里抽 JSONB。
    """

    __tablename__ = "patch_lineage_snapshots"

    lineage_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("patch_lineages.id", ondelete="CASCADE"), nullable=False, index=True
    )
    analysis_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False, index=True
    )
    photo_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("photos.id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )

    patch_id: Mapped[str] = mapped_column(String(16), nullable=False)  # LLM 输出的 p1/p2/...

    region: Mapped[str] = mapped_column(String(32), nullable=False)
    bbox_norm: Mapped[list[float]] = mapped_column(JSONB, nullable=False)
    area_ratio: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    coverage: Mapped[str] = mapped_column(String(16), nullable=False)
    dominant_type: Mapped[str] = mapped_column(String(16), nullable=False)
    estimated_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    inflammation: Mapped[str] = mapped_column(String(16), nullable=False)
    severity: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    # 匹配元信息（可选）
    match_info: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONB, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
