"""Pydantic schemas for Human-in-the-Loop Annotation Studio and Label Studio readiness."""

from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class PendingAnnotationSample(BaseModel):
    """Sample waiting for human annotation. Strictly omits ground-truth labels."""

    sample_id: str
    experiment_id: str
    dataset: str
    modality: Literal["image", "text_embedding"]
    round_num: int
    predicted_class_id: Optional[int] = None
    predicted_class_name: Optional[str] = None
    confidence: Optional[float] = None
    uncertainty: Optional[float] = None
    probabilities: Optional[List[float]] = None
    class_names: List[str]
    image_base64: Optional[str] = None
    feature_summary: Optional[Dict[str, float]] = None
    embedding_preview: Optional[List[float]] = None
    # Ground truth is explicitly excluded prior to annotation submission.


class PendingAnnotationsResponse(BaseModel):
    experiment_id: str
    experiment_name: str
    dataset: str
    round_num: int
    pending_count: int
    completed_human_count: int
    total_labeled_count: int
    samples: List[PendingAnnotationSample]


class AnnotationSubmitRequest(BaseModel):
    experiment_id: str
    sample_id: str
    label_id: int = Field(ge=0)
    annotator: str = Field(default="human_ui", min_length=1, max_length=64)


class AnnotationSubmitResponse(BaseModel):
    experiment_id: str
    sample_id: str
    assigned_label_id: int
    assigned_label_name: str
    annotator: str
    remaining_pending: int
    ready_to_retrain: bool


class AnnotationHistoryItem(BaseModel):
    sample_id: str
    experiment_id: str
    assigned_label_id: int
    assigned_label_name: str
    true_label_id: Optional[int] = None
    true_label_name: Optional[str] = None
    matches_ground_truth: Optional[bool] = None
    round_added: Optional[int] = None
    annotator: str
    source_type: Literal["human", "simulated_oracle", "cold_start", "other"]
    updated_at: Optional[str] = None


class AnnotationHistoryResponse(BaseModel):
    experiment_id: str
    total_count: int
    human_count: int
    oracle_count: int
    items: List[AnnotationHistoryItem]


class LabelStudioStatusResponse(BaseModel):
    integration_status: Literal["pending_phase_3"]
    label_studio_configured: bool
    webhook_endpoint: str
    supported_events: List[str]
    notes: str


class LabelStudioWebhookPayload(BaseModel):
    action: str = Field(..., description="Label Studio event type, e.g. ANNOTATION_CREATED")
    experiment_id: Optional[str] = None
    sample_id: Optional[str] = None
    label_id: Optional[int] = None
    annotator: Optional[str] = None
    raw_payload: Optional[Dict] = None

