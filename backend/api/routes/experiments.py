"""Active Learning experiment lifecycle, training progress, PCA, and sample selection routes."""

import csv
import io
from typing import List
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from backend.api.schemas.experiments import (
    ExperimentCompareRequest,
    ExperimentCompareResponse,
    ExperimentCreateRequest,
    ExperimentDetailResponse,
    ExperimentEvaluationResponse,
    ExperimentJobsResponse,
    ExperimentProgressResponse,
    ExperimentSummary,
    PcaEmbeddingResponse,
    SelectedSamplesResponse,
)
from backend.api.services.experiment_service import (
    advance_experiment_round,
    compare_experiments,
    create_experiment,
    delete_experiment,
    get_experiment_detail,
    get_experiment_evaluation,
    get_experiment_jobs,
    get_experiment_pca,
    get_experiment_progress,
    get_experiment_selected_samples,
    list_experiments,
    resume_experiment,
    start_experiment,
    stop_experiment,
)

router = APIRouter(prefix="/experiments", tags=["Experiments"])


@router.get("", response_model=List[ExperimentSummary], summary="List all Active Learning experiments")
def get_all_experiments() -> List[ExperimentSummary]:
    return list_experiments()


@router.post("", response_model=ExperimentSummary, summary="Create (and optionally start) an experiment")
def create_new_experiment(req: ExperimentCreateRequest) -> ExperimentSummary:
    try:
        return create_experiment(req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{experiment_id}", response_model=ExperimentDetailResponse, summary="Get experiment details and round metrics")
def get_experiment(experiment_id: str) -> ExperimentDetailResponse:
    try:
        return get_experiment_detail(experiment_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{experiment_id}/start", response_model=ExperimentSummary, summary="Start an existing experiment")
def trigger_start(experiment_id: str) -> ExperimentSummary:
    try:
        return start_experiment(experiment_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{experiment_id}/stop", response_model=ExperimentSummary, summary="Cancel or pause an active experiment safely")
def trigger_stop(experiment_id: str) -> ExperimentSummary:
    try:
        return stop_experiment(experiment_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{experiment_id}/resume", response_model=ExperimentSummary, summary="Resume a paused, cancelled, or failed experiment from its latest checkpoint")
def trigger_resume(experiment_id: str) -> ExperimentSummary:
    try:
        return resume_experiment(experiment_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{experiment_id}/advance", response_model=ExperimentSummary, summary="Retrain and advance HITL experiment round")
def trigger_advance(experiment_id: str) -> ExperimentSummary:
    try:
        return advance_experiment_round(experiment_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{experiment_id}/progress", response_model=ExperimentProgressResponse, summary="Poll live experiment training progress")
def poll_progress(experiment_id: str) -> ExperimentProgressResponse:
    try:
        return get_experiment_progress(experiment_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{experiment_id}/jobs", response_model=ExperimentJobsResponse, summary="List background job executions and statuses for an experiment")
def list_experiment_jobs(experiment_id: str) -> ExperimentJobsResponse:
    try:
        return get_experiment_jobs(experiment_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{experiment_id}/evaluation", response_model=ExperimentEvaluationResponse, summary="Get comprehensive evaluation metrics (Accuracy, F1, ECE, NLL, ALC, Cost Savings)")
def evaluate_experiment(experiment_id: str) -> ExperimentEvaluationResponse:
    try:
        return get_experiment_evaluation(experiment_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/compare", response_model=ExperimentCompareResponse, summary="Compare multiple experiments and tracked MLflow runs side by side")
def compare_runs_endpoint(req: ExperimentCompareRequest) -> ExperimentCompareResponse:
    try:
        return compare_experiments(req)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{experiment_id}/pca", response_model=PcaEmbeddingResponse, summary="Get 2D PCA latent embedding coordinates")
def get_pca(experiment_id: str) -> PcaEmbeddingResponse:
    try:
        return get_experiment_pca(experiment_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{experiment_id}/selected-samples", response_model=SelectedSamplesResponse, summary="Get latest queried samples and uncertainty breakdown")
def get_selected(experiment_id: str) -> SelectedSamplesResponse:
    try:
        return get_experiment_selected_samples(experiment_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{experiment_id}/export", summary="Export experiment round metrics as CSV")
def export_experiment_csv(experiment_id: str):
    try:
        detail = get_experiment_detail(experiment_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["round_num", "strategy", "n_labeled", "n_pool", "test_acc", "test_f1", "ece", "temperature", "timestamp"])
    for r in detail.rounds:
        writer.writerow([r.round_num, r.strategy, r.n_labeled, r.n_pool, r.test_acc, r.test_f1, r.ece, r.temperature, r.timestamp])
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={experiment_id}_metrics.csv"},
    )


@router.delete("/{experiment_id}", summary="Delete an experiment and its SQLite DB")
def remove_experiment(experiment_id: str) -> dict:
    try:
        delete_experiment(experiment_id)
        return {"deleted": True, "experiment_id": experiment_id}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

