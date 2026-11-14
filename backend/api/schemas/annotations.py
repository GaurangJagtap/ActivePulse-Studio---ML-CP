"""Pydantic schemas for Human-in-the-Loop Annotation Studio and Label Studio readiness."""

from typing import Any, Dict, List, Literal, Optional
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
    integration_status: Literal["connected", "configured", "unconfigured", "ready"]
    label_studio_configured: bool
    connected: bool = False
    base_url: Optional[str] = None
    default_project_id: Optional[int] = None
    project_count: int = 0
    webhook_endpoint: str
    supported_events: List[str]
    notes: str


class LabelStudioWebhookPayload(BaseModel):
    action: str = Field(..., description="Label Studio event type, e.g. ANNOTATION_CREATED")
    experiment_id: Optional[str] = None
    sample_id: Optional[str] = None
    label_id: Optional[int] = None
    annotator: Optional[str] = None
    webhook_event_id: Optional[str] = None
    task: Optional[Dict[str, Any]] = None
    annotation: Optional[Dict[str, Any]] = None
    project: Optional[Dict[str, Any]] = None
    raw_payload: Optional[Dict[str, Any]] = None


class LabelStudioWebhookResponse(BaseModel):
    status: str
    action: str
    experiment_id: str
    sample_id: Optional[str] = None
    assigned_label_id: Optional[int] = None
    duplicate_skipped: bool = False
    batch_id: Optional[str] = None
    batch_completed: int = 0
    batch_threshold: int = 0
    ready_to_retrain: bool = False
    retrain_triggered: bool = False
    message: str


class LabelStudioTaskCreateRequest(BaseModel):
    experiment_id: str
    sample_ids: Optional[List[str]] = None
    project_id: Optional[int] = None
    double_annotation_rate: float = Field(0.10, ge=0.0, le=1.0)
    include_pre_annotations: bool = True
    webhook_url: Optional[str] = None


class LabelStudioTaskCreateResponse(BaseModel):
    experiment_id: str
    project_id: int
    tasks_created: int
    double_annotation_count: int
    batch_id: str
    mappings: List[Dict[str, Any]]


class LabelStudioSyncRequest(BaseModel):
    experiment_id: str
    project_id: Optional[int] = None
    batch_id: Optional[str] = None


class LabelStudioSyncResponse(BaseModel):
    experiment_id: str
    project_id: int
    synced_count: int
    duplicate_skipped: int
    invalid_skipped: int
    synced_sample_ids: List[str]
    ready_to_retrain: bool


class AnnotationQualityResponse(BaseModel):
    experiment_id: str
    total_annotations: int
    total_annotated_samples: int
    overlapping_samples: int
    agreed_overlapping_samples: int
    disagreement_count: int
    raw_agreement_rate: Optional[float] = None
    cohens_kappa: Optional[float] = None
    kappa_status_note: str
    pairwise_kappa: List[Dict[str, Any]]
    class_wise_agreement: List[Dict[str, Any]]
    disagreements: List[Dict[str, Any]]
    annotator_stats: List[Dict[str, Any]]
    source: str = "independent_human_annotators"


class CompletedAnnotationItem(BaseModel):
    id: int
    sample_id: str
    label_id: int
    label_name: str
    annotator: str
    round_num: int
    source_type: str
    is_double_annotation: bool
    batch_id: Optional[str] = None
    created_at: Optional[str] = None


class CompletedAnnotationsResponse(BaseModel):
    experiment_id: str
    total_completed: int
    annotations: List[CompletedAnnotationItem]

