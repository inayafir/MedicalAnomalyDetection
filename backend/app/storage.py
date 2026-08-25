from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import UploadFile

from app.config import settings


def _storage_root() -> Path:
    root = Path(settings.STORAGE_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    return root


def save_image(file: UploadFile) -> str:
    now = datetime.now(timezone.utc)
    ext = Path(file.filename or "upload.png").suffix or ".png"
    rel_dir = f"images/{now.year}/{now.month:02d}/{now.day:02d}"
    abs_dir = _storage_root() / rel_dir
    abs_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{uuid.uuid4().hex}{ext}"
    abs_path = abs_dir / filename
    content = file.file.read()
    abs_path.write_bytes(content)
    return f"{rel_dir}/{filename}"


def save_heatmap(image_bytes: bytes) -> str:
    now = datetime.now(timezone.utc)
    rel_dir = f"heatmaps/{now.year}/{now.month:02d}/{now.day:02d}"
    abs_dir = _storage_root() / rel_dir
    abs_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{uuid.uuid4().hex}.png"
    abs_path = abs_dir / filename
    abs_path.write_bytes(image_bytes)
    return f"{rel_dir}/{filename}"


def save_report(pdf_bytes: bytes, prediction_id: int) -> str:
    rel_dir = f"reports/{prediction_id}"
    abs_dir = _storage_root() / rel_dir
    abs_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{uuid.uuid4().hex}.pdf"
    abs_path = abs_dir / filename
    abs_path.write_bytes(pdf_bytes)
    return f"{rel_dir}/{filename}"


def get_file(relative_path: str) -> bytes:
    abs_path = _storage_root() / relative_path
    if not abs_path.exists():
        raise FileNotFoundError(f"File not found: {relative_path}")
    return abs_path.read_bytes()
