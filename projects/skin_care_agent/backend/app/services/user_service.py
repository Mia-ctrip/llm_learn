from sqlalchemy.orm import Session

from app.models.user import User


SEED_USER_ID = 1


def ensure_seed_user(db: Session, user_id: int = SEED_USER_ID) -> User:
    """MVP 阶段创建或返回固定开发用户；接入微信登录后替换。"""
    user = db.get(User, user_id)
    if user is None:
        user = User(id=user_id, nickname="dev")
        db.add(user)
        db.flush()
    return user