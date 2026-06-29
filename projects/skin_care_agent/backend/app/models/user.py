from __future__ import annotations

from typing import Optional

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, IdMixin, TimestampMixin


class User(Base, IdMixin, TimestampMixin):
    __tablename__ = "users"

    wx_openid: Mapped[Optional[str]] = mapped_column(String(64), unique=True, nullable=True)
    nickname: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
