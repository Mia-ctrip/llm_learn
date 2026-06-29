"""ORM models package.

Importing this module ensures all models are registered on Base.metadata
(useful for alembic autogenerate).
"""

from app.models.base import Base  # noqa: F401
from app.models.user import User  # noqa: F401
from app.models.photo import Photo  # noqa: F401

__all__ = ["Base", "User", "Photo"]
