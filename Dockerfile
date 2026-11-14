FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY src ./src
COPY backend ./backend
COPY configs ./configs
COPY scripts ./scripts
COPY migrations ./migrations
COPY alembic.ini ./

RUN mkdir -p /app/.activepulse_data /app/results /app/data

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/api/v1/health || exit 1

CMD ["sh", "-c", "python scripts/init_db.py && uvicorn backend.api.main:app --host 0.0.0.0 --port 8000"]

