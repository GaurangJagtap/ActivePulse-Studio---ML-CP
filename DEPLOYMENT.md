# ActivePulse Studio – Production Deployment Guide

This guide describes how to run the completed Phases 1–5 backend stack using either local Python `.venv` development or full Docker Compose orchestration with **FastAPI**, **PostgreSQL**, **MLflow**, and **Label Studio**.

---

## 1. Environment Variables

Copy `.env.example` to `.env` and fill in values appropriate for your environment:

```bash
cp .env.example .env
```

Key settings:
- `DATABASE_URL`: Leave empty for isolated local per-experiment SQLite databases under `.activepulse_data/experiments/`. Set `postgresql://user:pass@host:5432/activepulse` (or `postgresql+pg8000://...` on Windows) for shared PostgreSQL persistence.
- `LABEL_STUDIO_URL`, `LABEL_STUDIO_API_KEY`: Connection settings for the Label Studio server.
- `LABEL_STUDIO_WEBHOOK_SECRET`: Optional shared secret checked on `POST /api/v1/webhooks/label-studio` via `X-Label-Studio-Secret`, `Authorization: Bearer <secret>`, or HMAC `X-Label-Studio-Signature`.
- `DEFAULT_DOUBLE_ANNOTATION_RATE`: Blind double-annotation fraction (default `0.10` = 10%).
- `BATCH_RETRAIN_THRESHOLD`, `AUTO_RETRAIN_ON_BATCH`: Batch completion ratio (`1.0` = 100%) and optional automatic HITL round advancement upon webhook completion.
- `MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT_NAME`, `MLFLOW_ENABLED`: MLflow experiment tracking configuration.

---

## 2. Database Initialization & Migrations

### Initialize Schema (SQLite or PostgreSQL)
```bash
python scripts/init_db.py
# Or explicitly:
python scripts/init_db.py --db-url postgresql+pg8000://activepulse:pass@localhost:5432/activepulse
```

### Run Alembic Migrations
```bash
alembic upgrade head
```

Schema revisions live in `migrations/versions/` and configure all core tables:
`experiments`, `samples`, `rounds`, `annotation_batches`, `annotations`, `quality_metrics`, `experiment_jobs`, and `model_versions`.

---

## 3. Docker Compose Orchestration

Bring up the full backend + PostgreSQL + MLflow + Label Studio stack:

```bash
docker compose up --build -d
```

Exposed service URLs:
- **FastAPI Backend & OpenAPI Docs:** `http://localhost:8000/docs`
- **Health Check:** `http://localhost:8000/api/v1/health`
- **MLflow UI:** `http://localhost:5000`
- **Label Studio UI:** `http://localhost:8080`
- **PostgreSQL:** `localhost:5432`

Check service status & health:
```bash
docker compose ps
docker compose logs -f backend
```

Stop services (preserving named volumes):
```bash
docker compose down
```

---

## 4. Label Studio Human-in-the-Loop Workflow

1. Create a `human_in_the_loop` experiment via `POST /api/v1/experiments` and wait for status `awaiting_annotations`.
2. Upload queued samples + calibrated model pre-annotations to Label Studio:
   `POST /api/v1/annotations/create-tasks` (with `double_annotation_rate=0.10`).
3. Configure Label Studio webhook target to `http://<backend-host>:8000/api/v1/webhooks/label-studio` with header `X-Label-Studio-Secret: <secret>`, OR pull annotations manually via `POST /api/v1/annotations/sync`.
4. Inspect Cohen's Kappa inter-annotator agreement and disagreement flags via `GET /api/v1/annotations/quality?experiment_id=<id>`.
5. Trigger model retraining and the next query round via `POST /api/v1/experiments/{id}/advance`.

---

## 5. Multi-Seed Benchmarking (Phase 5)

Run all 8 active-learning strategies across 5 paired seeds (`42 43 44 45 46`) on CIFAR-10 and AG News:

```bash
python scripts/run_benchmark.py --dataset both --synthetic --output-dir results
```

To benchmark on real cached/downloadable CIFAR-10 (when network access is available):
```bash
python scripts/run_benchmark.py --dataset cifar10 --output-dir results/cifar10_real
```

Artifacts produced in `results/`:
- `benchmark_savings.csv` & `benchmark_savings.md`
- `learning_curves.png`
- `accuracy_comparison.png`
- `label_efficiency.png`
- `calibration_comparison.png`
- `confusion_matrices/cm_<dataset>_<strategy>.png`
- `statistical_summary.json` & `statistical_summary.md`
- `experiment_config.yaml`

