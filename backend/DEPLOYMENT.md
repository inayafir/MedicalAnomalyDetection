# Deployment Guide

## Environment Variables

All configuration is via environment variables (see `.env.example` for defaults).

### Required for production

| Variable | Example | Description |
|---|---|---|
| `ENVIRONMENT` | `production` | Enables stricter security defaults |
| `DATABASE_URL` | `postgresql://user:pass@host:5432/dbname` | Use Postgres in production |
| `CORS_ORIGINS` | `https://yourapp.com` | Comma-separated allowed origins (no wildcards) |
| `API_KEY` | `a-long-random-string` | Shared secret for all non-health endpoints |
| `RESNET_CHECKPOINT` | `ml_core/checkpoints/resnet50.pth` | ResNet-50 weights (15 classes) |
| `YOLO_CHECKPOINT` | `ml_core/checkpoints/yolov8m_14class.pt` | YOLOv8m weights (14 classes) |

### Optional

| Variable | Default | Description |
|---|---|---|
| `SENTRY_DSN` | (unset) | Sentry DSN for error tracking |
| `RATE_LIMIT_PER_MINUTE` | `10` | Requests/min per IP on upload/prediction |
| `DATA_RETENTION_DAYS` | `0` | Auto-delete old data (0 = disabled) |
| `ML_DEVICE` | `cpu` | `cpu` or `cuda` |

## Checkpoint Delivery

Model checkpoints (~320MB total) are tracked via **Git LFS**. Options:

1. **LFS pull at build time** (recommended): Add `git lfs pull` to your Dockerfile or build script before `COPY . .`
2. **Bake into Docker image**: Large image but no runtime dependency on Git LFS
3. **Mount as volume**: Keep checkpoints on a persistent disk, set env vars to the mount path

## Database

- **Development**: SQLite (default, zero config)
- **Production**: Use managed Postgres (e.g. Render Postgres, Supabase). Update `DATABASE_URL`.
- **Migrations**: The app uses `Base.metadata.create_all()` which is fine for initial setup. For schema changes in production, adopt **Alembic** before making breaking changes to the ORM models.

## Running

```bash
# Local development
make run

# Docker
docker compose up --build

# Production (behind reverse proxy that terminates TLS)
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**HTTPS**: The app does not terminate TLS. Deploy behind a reverse proxy (nginx, Caddy, Cloudflare) or platform that handles TLS (Render, Railway, Fly.io). HTTP-only exposure is not acceptable for anything beyond local dev.

## Rollback

- Keep previous Docker images tagged. Roll back by redeploying the previous tag.
- Database schema changes require a separate rollback migration (another reason to adopt Alembic).
- Checkpoint files are immutable — swapping back to a previous checkpoint is just an env var change.

## Data Retention

Set `DATA_RETENTION_DAYS` to auto-delete old data. Run the cleanup manually:

```bash
python scripts/cleanup_retention.py
```

Or schedule it as a cron job / background worker.

## Research Use Disclaimer

This system is for **research and demonstration purposes only**. Predictions are not a substitute for professional medical diagnosis. Do not use outputs from this system for clinical decision-making without review by a qualified healthcare professional.
