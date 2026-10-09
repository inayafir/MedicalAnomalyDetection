# Stage 1: build dependencies
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-torch-cpu.txt ./
# CPU-only torch/torchvision FIRST, from the dedicated CPU wheel index — see
# requirements-torch-cpu.txt for why this must not come from plain PyPI.
RUN pip install --no-cache-dir --prefix=/install \
        -r requirements-torch-cpu.txt \
        --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: slim runtime
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

COPY . .

RUN mkdir -p storage

# NOTE: runs as root so a Render persistent disk mounted at /app/storage is writable.
# (Render mounts disks root-owned; a non-root user could not save uploads.)

ENV YOLO_CONFIG_DIR=/tmp/Ultralytics
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:' + __import__('os').environ.get('PORT','8000') + '/health/live')" || exit 1

# --forwarded-allow-ips=* so the rate limiter sees real client IPs behind Render's proxy.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips='*'"]
