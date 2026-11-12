"""Human-in-the-loop Annotation Studio and Label Studio webhook routes."""

from fastapi import APIRouter, HTTPException, Query

from backend.api.schemas.annotations import (
    AnnotationHistoryResponse,
    AnnotationSubmitRequest,
    AnnotationSubmitResponse,
    LabelStudioStatusResponse,
    LabelStudioWebhookPayload,
    PendingAnnotationsResponse,
)
from backend.api.services.annotation_service import (
    get_annotation_history,
    get_label_studio_status,
    get_pending_annotations,
    process_label_studio_webhook,
    submit_annotation,
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


@router.get("/webhooks/label-studio/status", response_model=LabelStudioStatusResponse, summary="Check Phase 3 Label Studio integration status")
def label_studio_status() -> LabelStudioStatusResponse:
    return get_label_studio_status()


@router.post("/webhooks/label-studio", summary="Label Studio webhook receiver (Phase 3 integration hook)")
def label_studio_webhook(payload: LabelStudioWebhookPayload) -> dict:
    try:
        return process_label_studio_webhook(payload)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

