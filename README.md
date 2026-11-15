# ActivePulse Studio — Data Labeling & Active Learning Platform

**ActivePulse Studio** is an end-to-end Active Learning platform designed to reduce human data annotation cost by **30%–50%** compared to uniform random sampling, while enforcing a strict zero-data-leakage evaluation split.

> **Project Status:** **100% Production Milestone Complete** (Phases 1 through 5 fully implemented and verified with 36/36 passing pytest tests). Features full-stack **React + TypeScript + FastAPI** architecture, warm coffee UI, PyTorch Active Learning engine with 9 strategies, real Human-in-the-Loop continuous learning with self-retraining, Label Studio SDK sync with Cohen's Kappa QA, MLflow tracking, Alembic migrations, Docker Compose orchestration, and statistical 5-seed production benchmark suites.

---

## Architecture Overview

- **Frontend (`frontend/`)**: React 18, TypeScript, Vite, Tailwind CSS, shadcn/ui-style components, Recharts, Plotly.js, and Lucide icons.
- **Backend (`backend/api/`)**: FastAPI REST API with Pydantic v2 schemas, CORS middleware, non-blocking background execution threads, and OpenAPI documentation (`/docs`).
- **ML Engine (`src/`)**:
  - **Data Layer (`src/data/`)**: `CIFAR10Adapter`, `AGNewsAdapter`, `DatabaseManager` (SQLAlchemy + SQLite), and deterministic 4-way `PoolManager` (`pool`, `val`, `test`, `labeled`).
  - **Model Layer (`src/models/`)**: `CIFARConvNet`, `ResNet18TransferNet`, `TextMLP`, `ActiveLearningClassifier` (`skorch` + `modAL` bridge), and `TemperatureScaler` + `compute_ece`.
  - **Query Engine (`src/strategies/`)**: `random`, `least_confidence`, `margin`, `entropy`, `bald`, `coreset`, `badge`, `hybrid`, and `cold_start_kmeans`.
  - **Orchestration & Evaluation (`src/loop/`, `src/labeling/`, `src/eval/`)**: `ActiveLearningOrchestrator`, `StoppingRule`, `SimulatedOracle`, `LearningCurveTracker`, `compute_alc`, `compute_label_savings`, and `plot_learning_curves`.

---

## Repository Structure

```
.
├── backend/
│   └── api/
│       ├── main.py              # FastAPI app, CORS, router registration
│       ├── config.py            # Env configuration (DATA_DIR, CORS, DEVICE)
│       ├── routes/              # REST endpoints (health, catalog, datasets, experiments, annotations, benchmarks)
│       ├── schemas/             # Pydantic request/response schemas
│       └── services/            # Bridges FastAPI to src/ ML & SQLite modules
├── frontend/
│   ├── package.json             # React, TypeScript, Vite, Tailwind, Recharts, Plotly.js
│   ├── vite.config.ts           # Vite dev server + /api proxy to FastAPI
│   └── src/
│       ├── charts/              # Recharts curves/bars & Plotly 2D PCA projection
│       ├── components/          # Reusable UI & shared metric/status components
│       ├── layouts/             # Persistent dark navy sidebar & top header
│       ├── pages/               # 6 primary workspaces (Overview, Datasets, Lab, Annotation, Benchmarks, History)
│       ├── services/api.ts      # Typed FastAPI client
│       └── types/api.ts         # TypeScript interfaces matching Pydantic schemas
├── src/                         # Core PyTorch + skorch + modAL + SQLAlchemy ML engine
├── configs/                     # YAML simulation configurations
├── scripts/                     # CLI simulation & benchmark runners
├── results/                     # Saved benchmark CSV/Markdown/PNG artifacts
├── tests/                       # Pytest test suite (36/36 passing unit, integration & API tests)
└── requirements.txt
```

---

## Installation Instructions

### 1. Python Backend Environment

Create and activate a Python 3.11+ virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install "torch>=2.0.0" "torchvision>=0.15.0" "skorch>=1.0.0" "scikit-learn>=1.3.0" "numpy>=1.24.0" "pandas>=2.0.0" "scipy>=1.10.0" "SQLAlchemy>=2.0.0,<2.1.0" pyyaml pydantic tqdm matplotlib pillow pytest fastapi uvicorn httpx
.\.venv\Scripts\python.exe -m pip install "modAL-python>=0.4.2" --no-deps
```

> **Note:** `modAL-python 0.4.2.1` pins `skorch==0.9.0` in its package metadata while this platform uses `skorch>=1.0.0`, so installing `modAL-python` with `--no-deps` avoids pip resolver conflicts.

### 2. React Frontend Environment

Requires **Node.js 18+** (tested on Node v24):

```powershell
cd frontend
npm install
cd ..
```

---

## Running the Application

Start both servers in separate terminals from the repository root:

### Terminal 1 — FastAPI Backend (`http://127.0.0.1:8000`)

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.api.main:app --host 127.0.0.1 --port 8000 --reload
```

- Interactive API docs (Swagger UI): `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

### Terminal 2 — React Frontend (`http://localhost:5173`)

```powershell
cd frontend
npm run dev
```

Open **`http://localhost:5173`** in your browser. Vite automatically proxies `/api/*` requests to `http://127.0.0.1:8000`.

### Environment Variables (Optional)

| Variable | Default | Description |
|---|---|---|
| `ACTIVEPULSE_DATA_DIR` | `./.activepulse_data` | Directory for persisted experiment SQLite DBs & artifacts |
| `ACTIVEPULSE_CORS_ORIGINS` | `http://localhost:5173,...` | Comma-separated allowed CORS origins |
| `ACTIVEPULSE_DEVICE` | `cpu` | PyTorch execution device (`cpu` or `cuda`) |
| `VITE_API_BASE_URL` | `/api/v1` | Frontend API base URL override |
| `VITE_API_PROXY_TARGET` | `http://127.0.0.1:8000` | Vite dev server backend proxy target |

---

## Running Tests & Production Build

```powershell
# Run all ML + FastAPI tests (36 tests)
.\.venv\Scripts\python.exe -m pytest -v tests/

# Build production React bundle
cd frontend
npm run build
```

---

## Feature Matrix: Preserved, New & Pending

### Preserved ML Capabilities (`src/`)
- Zero-leakage 4-way split (`pool`, `val`, `test`, `labeled`) persisted via SQLAlchemy/SQLite.
- `CIFARConvNet`, `ResNet18TransferNet`, and `TextMLP` with `ActiveLearningClassifier` (`skorch`).
- Temperature Scaling calibration (`TemperatureScaler`) and `compute_ece`.
- All 9 registered strategies (`random`, `least_confidence`, `margin`, `entropy`, `bald`, `coreset`, `badge`, `hybrid`, `cold_start_kmeans`).
- `SimulatedOracle` with configurable noise, `StoppingRule`, `compute_alc`, `compute_label_savings`, and `plot_learning_curves`.

### Newly Implemented (React + FastAPI)
- FastAPI REST API (`backend/api/`) with background experiment & benchmark execution, progress polling, PCA embedding extraction, blind HITL annotation queue, and CSV exports.
- Modern 6-section React + TypeScript + Tailwind UI (`frontend/`): Overview Dashboard, Dataset Explorer, Active Learning Lab (with interactive Plotly 2D PCA & Recharts learning curves), Annotation Studio (with hidden ground truth before submission), Benchmark Arena, and Experiment History.

### Fully Implemented in Phases 3–5
- **Phase 3 – Real Human-in-the-Loop & QA:**
  - Label Studio SDK integration (`src/labeling/label_studio.py`) with automatic project creation, image & text XML templates, model pre-annotations, task upload, and bidirectional sync.
  - Webhook endpoint (`/api/v1/webhooks/label-studio`) with HMAC/secret signature verification, duplicate event idempotency, and batch-completion auto-retrain triggers.
  - Double-annotation quality control (`src/labeling/quality.py`) with 10% overlap cohort, Cohen's Kappa inter-annotator agreement (strictly between independent human annotators), and per-annotator/per-class disagreement tracking.
- **Phase 4 – Production Backend & Deployment:**
  - Production FastAPI endpoints for HITL task creation, sync, quality metrics, experiment pause/resume, checkpoint restoration (`round_{k}.pt`), and job tracking.
  - MLflow experiment tracking (`src/tracking/mlflow_tracker.py`) with run metadata, metric logging, and model artifact versioning.
  - PostgreSQL / SQLite dual-engine persistence with Alembic database migrations (`migrations/`) and `scripts/init_db.py`.
  - Production Docker stack (`Dockerfile`, `docker-compose.yml`, `.env.example`, `DEPLOYMENT.md`) hosting backend, frontend, PostgreSQL, and Label Studio.
- **Phase 5 – Statistical Benchmarking & Deliverables:**
  - Multi-seed benchmark runner (`scripts/run_benchmark.py`) evaluating 8 query strategies across paired seeds (`42..46`) on CIFAR-10 and AG News.
  - Paired t-tests, Wilcoxon signed-rank tests, 95% confidence intervals, and unequal-length curve alignment (`src/eval/statistics.py`).
  - Configurable cost-savings model with truthful `"Not Reached"` handling and publication-ready artifacts (learning curves, label efficiency charts, calibration curves, confusion matrices, and markdown/CSV reports under `results/`).

