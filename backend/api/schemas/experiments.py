"""Pydantic schemas for Active Learning experiments, rounds, PCA, and selected samples."""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class ExperimentCreateRequest(BaseModel):
    name: Optional[str] = Field(default=None, max_length=120)
    dataset: str = "cifar10"
    model_type: str = "cifar_cnn"
    strategy: str = "hybrid"
    mode: Literal["simulated_oracle", "human_in_the_loop"] = "simulated_oracle"
    initial_size: int = Field(default=30, ge=2, le=200)
    batch_size: int = Field(default=15, ge=1, le=50)
    max_rounds: int = Field(default=3, ge=1, le=20)
    label_budget: Optional[int] = Field(default=300, ge=4, le=5000)
    target_accuracy: Optional[float] = Field(default=None, ge=0.05, le=1.0)
    max_samples: int = Field(default=450, ge=4, le=5000)
    use_synthetic: bool = True
    max_epochs: int = Field(default=6, ge=1, le=50)
    lr: float = Field(default=0.01, gt=0.0, le=1.0)
    seed: int = 42
    label_noise_rate: float = Field(default=0.0, ge=0.0, le=0.5)
    warm_start: bool = False
    auto_start: bool = True


class RoundMetricSchema(BaseModel):
    round_num: int
    strategy: str
    n_labeled: int
    n_pool: int
    test_acc: float
    test_f1: float
    ece: Optional[float] = None
    nll: Optional[float] = None
    train_time_sec: Optional[float] = None
    temperature: Optional[float] = None
    timestamp: Optional[str] = None


class CostSavingsSchema(BaseModel):
    total_samples: int
    n_labeled: int
    pct_dataset_labeled: float
    estimated_passive_labels: int
    estimated_labels_saved: int
    estimated_hours_saved: float
    estimated_dollars_saved: float
    methodology_note: str


class ExperimentSummary(BaseModel):
    id: str
    name: str
    dataset: str
    model_type: str
    strategy: str
    mode: Literal["simulated_oracle", "human_in_the_loop"]
    status: Literal[
        "created",
        "running",
        "awaiting_annotations",
        "completed",
        "failed",
        "cancelled",
        "paused",
    ]
    current_round: int
    max_rounds: int
    current_stage: str
    latest_accuracy: Optional[float] = None
    latest_f1: Optional[float] = None
    latest_ece: Optional[float] = None
    n_labeled: int = 0
    n_pool: int = 0
    n_queued: int = 0
    created_at: str
    updated_at: str
    error_message: Optional[str] = None


class ExperimentDetailResponse(BaseModel):
    summary: ExperimentSummary
    config: Dict
    rounds: List[RoundMetricSchema]
    cost_savings: Optional[CostSavingsSchema] = None
    stop_reason: Optional[str] = None
    logs: List[str] = []


class ExperimentProgressResponse(BaseModel):
    experiment_id: str
    status: str
    current_round: int
    max_rounds: int
    current_stage: str
    stage_index: int
    stages: List[str]
    n_labeled: int
    n_pool: int
    n_queued: int
    latest_accuracy: Optional[float] = None
    latest_f1: Optional[float] = None
    latest_ece: Optional[float] = None
    error_message: Optional[str] = None
    logs: List[str] = []


class PcaPointSchema(BaseModel):
    sample_id: str
    pc1: float
    pc2: float
    category: Literal[
        "Unlabeled Pool",
        "Already Labeled",
        "Newly Queried by Active Learning",
    ]


class PcaEmbeddingResponse(BaseModel):
    experiment_id: str
    round_num: int
    explained_variance_ratio: List[float]
    points: List[PcaPointSchema]


class SelectedSampleSchema(BaseModel):
    sample_id: str
    predicted_class_id: int
    predicted_class_name: str
    confidence: float
    uncertainty: float
    entropy: float
    probabilities: List[float]
    class_names: List[str]
    image_base64: Optional[str] = None
    feature_summary: Optional[Dict[str, float]] = None
    # Note: true_class_* is only populated for completed simulated_oracle evaluation views,
    # NEVER in the human annotation queue.
    true_class_id: Optional[int] = None
    true_class_name: Optional[str] = None


class SelectedSamplesResponse(BaseModel):
    experiment_id: str
    round_num: int
    strategy: str
    samples: List[SelectedSampleSchema]


class JobStatusSchema(BaseModel):
    job_id: str
    experiment_id: str
    job_type: str
    status: str
    progress_pct: float
    current_stage: Optional[str] = None
    error_message: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class ExperimentJobsResponse(BaseModel):
    experiment_id: str
    jobs: List[JobStatusSchema]


class ExperimentEvaluationResponse(BaseModel):
    experiment_id: str
    dataset: str
    strategy: str
    mode: str
    rounds_completed: int
    latest_accuracy: Optional[float] = None
    latest_f1: Optional[float] = None
    latest_ece: Optional[float] = None
    latest_nll: Optional[float] = None
    alc: float
    total_train_time_sec: float
    labels_to_target: Optional[int] = None
    target_accuracy: Optional[float] = None
    rounds: List[RoundMetricSchema]
    cost_savings: Optional[CostSavingsSchema] = None
    model_version: Optional[str] = None


class ExperimentCompareRequest(BaseModel):
    experiment_ids: List[str] = Field(..., min_length=1, max_length=20)


class ExperimentCompareEntry(BaseModel):
    experiment_id: str
    name: str
    dataset: str
    model_type: str
    strategy: str
    status: str
    rounds_completed: int
    n_labeled: int
    final_accuracy: Optional[float] = None
    final_f1: Optional[float] = None
    final_ece: Optional[float] = None
    final_nll: Optional[float] = None
    alc: float


class ExperimentCompareResponse(BaseModel):
    experiments: List[ExperimentCompareEntry]
    mlflow_runs: List[Dict[str, Any]] = []

