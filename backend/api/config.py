"""Environment-driven configuration for the ActivePulse Studio API."""

import os
from typing import List


def _parse_origins(raw: str) -> List[str]:
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


API_TITLE = "ActivePulse Studio API"
API_VERSION = "1.0.0"
API_PREFIX = os.getenv("ACTIVEPULSE_API_PREFIX", "/api/v1")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.abspath(
    os.getenv("ACTIVEPULSE_DATA_DIR", os.path.join(REPO_ROOT, ".activepulse_data"))
)
RESULTS_DIR = os.path.abspath(
    os.getenv("ACTIVEPULSE_RESULTS_DIR", os.path.join(REPO_ROOT, "results"))
)

CORS_ORIGINS: List[str] = _parse_origins(
    os.getenv(
        "ACTIVEPULSE_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000",
    )
)

DEFAULT_DEVICE = os.getenv("ACTIVEPULSE_DEVICE", "cpu")
DATABASE_URL = os.getenv("DATABASE_URL", "")

# Phase 3: Label Studio & Quality Control
LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL", "")
LABEL_STUDIO_API_KEY = os.getenv("LABEL_STUDIO_API_KEY", "")
LABEL_STUDIO_WEBHOOK_SECRET = os.getenv("LABEL_STUDIO_WEBHOOK_SECRET", "")
DEFAULT_DOUBLE_ANNOTATION_RATE = float(os.getenv("DEFAULT_DOUBLE_ANNOTATION_RATE", "0.10"))
BATCH_RETRAIN_THRESHOLD = float(os.getenv("BATCH_RETRAIN_THRESHOLD", "1.0"))
AUTO_RETRAIN_ON_BATCH = os.getenv("AUTO_RETRAIN_ON_BATCH", "false").lower() in ("1", "true", "yes")

# Phase 4: MLflow Experiment Tracking
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "ActivePulse-Studio")
MLFLOW_ENABLED = os.getenv("MLFLOW_ENABLED", "true").lower() in ("1", "true", "yes")

# Phase 5: Default Cost Assumptions (configurable per experiment/benchmark)
COST_PER_HOUR = float(os.getenv("ACTIVEPULSE_COST_PER_HOUR", "25.0"))
SECONDS_PER_ANNOTATION = float(os.getenv("ACTIVEPULSE_SECONDS_PER_ANNOTATION", "8.0"))

