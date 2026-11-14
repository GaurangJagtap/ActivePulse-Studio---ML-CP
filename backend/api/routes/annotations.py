"""Human-in-the-loop Annotation Studio, Label Studio SDK sync, QA, and webhook routes."""

from fastapi import APIRouter, HTTPException, Query, Request

from backend.api.schemas.annotations import (
    AnnotationHistoryResponse,
    AnnotationQualityResponse,
    AnnotationSubmitRequest,
    AnnotationSubmitResponse,
    CompletedAnnotationsResponse,
    LabelStudioStatusResponse,
    LabelStudioSyncRequest,
    LabelStudioSyncResponse,
    LabelStudioTaskCreateRequest,
    LabelStudioTaskCreateResponse,
    LabelStudioWebhookPayload,
    LabelStudioWebhookResponse,
    PendingAnnotationsResponse,
)
from backend.api.services.annotation_service import (
    create_label_studio_tasks,
    get_annotation_history,
    get_annotation_quality,
    get_completed_annotations,
    get_label_studio_status,
    get_pending_annotations,
    process_label_studio_webhook,
    submit_annotation,
    sync_label_studio_annotations,
)

router = APIRouter(tags=["Annotations"])


@router.get("/annotations/pending", response_model=PendingAnnotationsResponse, summary="Get pending samples queued for human annotation (ground truth hidden)")
def list_pending_annotations(experiment_id: str = Query(...)) -> PendingAnnotationsResponse:
    try:
        return get_pending_annotations(experiment_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/annotations/submit", response_model=AnnotationSubmitResponse, summary="Submit a human annotation for a queued sample")
def submit_human_annotation(req: AnnotationSubmitRequest) -> AnnotationSubmitResponse:
    try:
        return submit_annotation(req)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/annotations/history", response_model=AnnotationHistoryResponse, summary="Get completed annotation history (distinguishing human vs oracle)")
def annotation_history(experiment_id: str = Query(...), limit: int = Query(default=200, ge=1, le=1000)) -> AnnotationHistoryResponse:
    try:
        return get_annotation_history(experiment_id, limit=limit)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/annotations/completed", response_model=CompletedAnnotationsResponse, summary="Get completed human/multi-annotator annotation records")
def completed_annotations(experiment_id: str = Query(...), limit: int = Query(default=200, ge=1, le=1000)) -> CompletedAnnotationsResponse:
    try:
        return get_completed_annotations(experiment_id, limit=limit)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/annotations/quality", response_model=AnnotationQualityResponse, summary="Compute inter-annotator Cohen's Kappa, class agreement, and disagreement flags")
def annotation_quality(experiment_id: str = Query(...)) -> AnnotationQualityResponse:
    try:
        return get_annotation_quality(experiment_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/annotations/create-tasks", response_model=LabelStudioTaskCreateResponse, summary="Upload queried samples to Label Studio with model pre-annotations")
@router.post("/annotations/label-studio/tasks", response_model=LabelStudioTaskCreateResponse, include_in_schema=False)
def create_ls_tasks(req: LabelStudioTaskCreateRequest) -> LabelStudioTaskCreateResponse:
    try:
        return create_label_studio_tasks(req)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/annotations/sync", response_model=LabelStudioSyncResponse, summary="Pull completed human annotations from Label Studio")
@router.post("/annotations/label-studio/sync", response_model=LabelStudioSyncResponse, include_in_schema=False)
def sync_ls_annotations(req: LabelStudioSyncRequest) -> LabelStudioSyncResponse:
    try:
        return sync_label_studio_annotations(req)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/webhooks/label-studio/status", response_model=LabelStudioStatusResponse, summary="Check Label Studio SDK configuration and live connectivity")
def label_studio_status() -> LabelStudioStatusResponse:
    return get_label_studio_status()


@router.post("/webhooks/label-studio", response_model=LabelStudioWebhookResponse, summary="Label Studio webhook receiver with secret validation and idempotent ingestion")
async def label_studio_webhook(payload: LabelStudioWebhookPayload, request: Request) -> LabelStudioWebhookResponse:
    raw_body = await request.body()
    headers = dict(request.headers)
    try:
        return process_label_studio_webhook(payload, headers=headers, raw_body=raw_body)
    except PermissionError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

