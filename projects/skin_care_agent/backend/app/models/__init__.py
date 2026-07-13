"""ORM models package.

Importing this module ensures all models are registered on Base.metadata
(useful for alembic autogenerate).
"""

from app.models.base import Base  # noqa: F401
from app.models.user import User  # noqa: F401
from app.models.photo import Photo  # noqa: F401
from app.models.ai_usage import AIUsageCounter  # noqa: F401
from app.models.ai_call_log import AICallLog  # noqa: F401
from app.models.analysis import Analysis  # noqa: F401
from app.models.chat_message import ChatMessage  # noqa: F401
from app.models.patch_lineage import PatchLineage, PatchLineageSnapshot  # noqa: F401

__all__ = [
    "Base",
    "User",
    "Photo",
    "AIUsageCounter",
    "AICallLog",
    "Analysis",
    "ChatMessage",
    "PatchLineage",
    "PatchLineageSnapshot",
]
