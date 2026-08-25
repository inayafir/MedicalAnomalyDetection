from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str = "sqlite:///./storage/dev.db"
    STORAGE_ROOT: str = "./storage"
    MAX_UPLOAD_SIZE_MB: int = 10
    ALLOWED_CONTENT_TYPES: str = "image/png,image/jpeg"
    ML_DEVICE: str = "cpu"

    @property
    def allowed_content_types_list(self) -> list[str]:
        return [t.strip() for t in self.ALLOWED_CONTENT_TYPES.split(",")]

    @property
    def max_upload_size_bytes(self) -> int:
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
