from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str = "sqlite:///./storage/dev.db"
    STORAGE_ROOT: str = "./storage"
    MAX_UPLOAD_SIZE_MB: int = 10
    ALLOWED_CONTENT_TYPES: str = "image/png,image/jpeg"
    ML_DEVICE: str = "cpu"
    CORS_ORIGINS: str = "*"
    RESNET_CHECKPOINT: str = "ml_core/checkpoints/resnet50.pth"
    YOLO_CHECKPOINT: str = "ml_core/checkpoints/yolov8m_14class.pt"

    @property
    def allowed_content_types_list(self) -> list[str]:
        return [t.strip() for t in self.ALLOWED_CONTENT_TYPES.split(",")]

    @property
    def max_upload_size_bytes(self) -> int:
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",")]

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
