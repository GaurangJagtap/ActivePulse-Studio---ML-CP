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
LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL", "")

