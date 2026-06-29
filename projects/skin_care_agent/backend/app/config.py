from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BACKEND_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BACKEND_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # app
    app_env: str = "dev"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_log_level: str = "INFO"

    # db
    database_url: str = "postgresql+psycopg://skin:skin@localhost:5432/skin_care"

    # storage
    storage_backend: str = "local"
    storage_local_dir: str = "./storage_local"
    storage_local_base_url: str = "http://localhost:8000/files"

    # ai rate limit
    ai_analyze_daily_limit: int = 10
    ai_chat_daily_limit: int = 50

    # ai providers
    ai_provider_primary: str = "mock"
    ai_provider_fallbacks: str = ""

    qwen_api_key: str = ""
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_model: str = "qwen-vl-max"

    glm_api_key: str = ""
    glm_base_url: str = "https://open.bigmodel.cn/api/paas/v4"
    glm_model: str = "glm-4v-plus"

    minimax_api_key: str = ""
    minimax_base_url: str = "https://api.minimax.chat/v1"
    minimax_model: str = "abab6.5-chat"

    doubao_api_key: str = ""
    doubao_base_url: str = "https://ark.cn-beijing.volces.com/api/v3"
    doubao_model: str = "doubao-vision-pro"

    # wechat
    wx_appid: str = ""
    wx_secret: str = Field(default="")

    @property
    def fallback_providers(self) -> list[str]:
        return [p.strip() for p in self.ai_provider_fallbacks.split(",") if p.strip()]

    @property
    def storage_local_path(self) -> Path:
        p = Path(self.storage_local_dir)
        if not p.is_absolute():
            p = BACKEND_ROOT / p
        return p


@lru_cache
def get_settings() -> Settings:
    return Settings()
