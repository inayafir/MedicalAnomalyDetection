"""Data retention cleanup script.

Deletes images, predictions, heatmaps, and reports older than DATA_RETENTION_DAYS.
Run via: python scripts/cleanup_retention.py

Set DATA_RETENTION_DAYS=-1 to disable (default).
Set DATA_RETENTION_DAYS=30 to delete records older than 30 days.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.config import settings
from app.db import engine, Base
from app.models import Image, Prediction, Report


def cleanup():
    if not settings.retention_enabled:
        print(f"DATA_RETENTION_DAYS={settings.DATA_RETENTION_DAYS} — retention cleanup disabled")
        return

    cutoff = datetime.now(timezone.utc) - timedelta(days=settings.DATA_RETENTION_DAYS)
    print(f"Deleting records older than {cutoff.isoformat()} ({settings.DATA_RETENTION_DAYS} days)")

    from sqlalchemy.orm import Session
    with Session(engine) as db:
        old_reports = db.query(Report).filter(Report.generated_at < cutoff).all()
        for r in old_reports:
            if r.pdf_path:
                p = Path(settings.STORAGE_ROOT) / r.pdf_path
                p.unlink(missing_ok=True)
        deleted_reports = db.query(Report).filter(Report.generated_at < cutoff).delete()

        old_preds = db.query(Prediction).filter(Prediction.created_at < cutoff).all()
        for p in old_preds:
            if p.heatmap_path:
                hp = Path(settings.STORAGE_ROOT) / p.heatmap_path
                hp.unlink(missing_ok=True)
        deleted_preds = db.query(Prediction).filter(Prediction.created_at < cutoff).delete()

        old_images = db.query(Image).filter(Image.uploaded_at < cutoff).all()
        for img in old_images:
            ip = Path(settings.STORAGE_ROOT) / img.file_path
            ip.unlink(missing_ok=True)
        deleted_images = db.query(Image).filter(Image.uploaded_at < cutoff).delete()

        db.commit()

    print(f"Deleted {deleted_reports} reports, {deleted_preds} predictions, {deleted_images} images")


if __name__ == "__main__":
    Base.metadata.create_all(bind=engine)
    cleanup()
