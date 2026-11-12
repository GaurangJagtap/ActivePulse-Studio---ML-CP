"""FastAPI entrypoint for ActivePulse Studio."""

import os
import sys

# Ensure repository root is importable when running `uvicorn backend.api.main:app`
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.config import API_PREFIX, API_TITLE, API_VERSION, CORS_ORIGINS, DATA_DIR
from backend.api.routes import (
    annotations,
    benchmarks,
    catalog,
    datasets,
    experiments,
    health,
)


def create_app() -> FastAPI:
    os.makedirs(DATA_DIR, exist_ok=True)

    app = FastAPI(
        title=API_TITLE,
        version=API_VERSION,
        description=(
            "REST API powering ActivePulse Studio: connecting React UI to PyTorch, skorch, "
            "modAL query strategies, Temperature Scaling calibration, and SQLite persistence."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router, prefix=API_PREFIX)
    app.include_router(catalog.router, prefix=API_PREFIX)
    app.include_router(datasets.router, prefix=API_PREFIX)
    app.include_router(experiments.router, prefix=API_PREFIX)
    app.include_router(annotations.router, prefix=API_PREFIX)
    app.include_router(benchmarks.router, prefix=API_PREFIX)

    @app.get("/", tags=["System"])
    def root() -> dict:
        return {
            "service": API_TITLE,
            "version": API_VERSION,
            "docs": "/docs",
            "api_prefix": API_PREFIX,
        }

    return app


app = create_app()

