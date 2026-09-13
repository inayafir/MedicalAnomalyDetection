from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    ENVIRONMENT: str = "development"

    DATABASE_URL: str = "sqlite:///./storage/dev.db"
    STORAGE_ROOT: str = "./storage"
    MAX_UPLOAD_SIZE_MB: int = 10
    ALLOWED_CONTENT_TYPES: str = "image/png,image/jpeg"
    ML_DEVICE: str = "cpu"
    CORS_ORIGINS: str = "*"
    RESNET_CHECKPOINT: str = "ml_core/checkpoints/resnet50.pth"
    YOLO_CHECKPOINT: str = "ml_core/checkpoints/yolov8m_14class.pt"

    API_KEY: Optional[str] = None
    SENTRY_DSN: Optional[str] = None
    RATE_LIMIT_PER_MINUTE: int = 10
    DATA_RETENTION_DAYS: int = -1

    @property
    def allowed_content_types_list(self) -> list[str]:
        return [t.strip() for t in self.ALLOWED_CONTENT_TYPES.split(",")]

    @property
    def max_upload_size_bytes(self) -> int:
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",")]

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"

    @property
    def retention_enabled(self) -> bool:
        return self.DATA_RETENTION_DAYS >= 0

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()


def validate_production_config(s: Settings | None = None) -> None:
    """Validate config for production. Raises ValueError if any check fails."""
    s = s or settings
    if not s.is_production:
        return

    errors = []

    if not s.API_KEY:
        errors.append(
            "API_KEY must be set when ENVIRONMENT=production. "
            "Refusing to start with authentication disabled."
        )

    if not s.CORS_ORIGINS or s.CORS_ORIGINS.strip() == "*":
        errors.append(
            "CORS_ORIGINS must be an explicit origin list in production, not '*'. "
            "Refusing to start."
        )

    if s.DATABASE_URL.startswith("sqlite"):
        errors.append(
            "SQLite is not supported in production — set DATABASE_URL to a Postgres "
            "connection string."
        )

    if errors:
        raise ValueError("\n".join(errors))
