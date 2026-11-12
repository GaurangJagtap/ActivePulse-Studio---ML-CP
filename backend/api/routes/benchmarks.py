"""Benchmark Arena routes for multi-strategy comparison and export."""

import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response

from backend.api.config import RESULTS_DIR
from backend.api.schemas.benchmarks import (
    BenchmarkJobStatus,
    BenchmarkResultsResponse,
    BenchmarkRunRequest,
)
from backend.api.services.benchmark_service import (
    get_benchmark_job_status,
    get_benchmark_results,
    start_benchmark_job,
)

router = APIRouter(prefix="/benchmarks", tags=["Benchmarks"])


@router.get("/results", response_model=BenchmarkResultsResponse, summary="Get latest multi-strategy benchmark results")
def fetch_benchmark_results() -> BenchmarkResultsResponse:
    return get_benchmark_results()


@router.post("/run", response_model=BenchmarkJobStatus, summary="Start a multi-seed active learning benchmark job")
def run_benchmark(req: BenchmarkRunRequest) -> BenchmarkJobStatus:
    try:
        return start_benchmark_job(req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/status/{job_id}", response_model=BenchmarkJobStatus, summary="Poll benchmark job progress")
def benchmark_status(job_id: str) -> BenchmarkJobStatus:
    try:
        return get_benchmark_job_status(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/export", summary="Export benchmark savings summary as CSV")
def export_benchmark_csv():
    csv_path = os.path.join(RESULTS_DIR, "benchmark_savings.csv")
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail="No benchmark CSV available yet.")
    with open(csv_path, "r", encoding="utf-8") as f:
        content = f.read()
    return Response(
        content=content,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=benchmark_savings.csv"},
    )


@router.get("/plot", summary="Fetch generated learning curves PNG artifact")
def get_benchmark_plot():
    img_path = os.path.join(RESULTS_DIR, "learning_curves.png")
    if not os.path.exists(img_path):
        raise HTTPException(status_code=404, detail="No benchmark learning curves plot generated yet.")
    return FileResponse(img_path, media_type="image/png")

