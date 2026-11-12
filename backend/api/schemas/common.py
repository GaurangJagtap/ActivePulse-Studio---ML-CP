"""Common Pydantic schemas for health, datasets, models, and strategies."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    pytorch_version: str
    device: str
    registered_strategies: int
    label_studio_configured: bool


class DatasetSummary(BaseModel):
    id: str
    name: str
    modality: str
    num_classes: int
    class_names: List[str]
    default_model: str
    supported_models: List[str]
    supports_real_download: bool
    supports_raw_text: bool
    description: str


class ClassCount(BaseModel):
    class_id: int
    class_name: str
    count: int


class SplitStats(BaseModel):
    n_total: int
    n_pool_unlabeled: int
    n_queued: int
    n_labeled: int
    n_val: int
    n_test: int


class DatasetStatsResponse(BaseModel):
    dataset_id: str
    name: str
    modality: str
    use_synthetic: bool
    max_samples: int
    num_classes: int
    class_names: List[str]
    split_stats: SplitStats
    class_distribution: List[ClassCount]
    feature_shape: List[int]
    upload_supported: bool = False
    upload_status_note: str = (
        "Custom dataset upload is planned for a future release; currently CIFAR-10 and AG News adapters are supported."
    )


class DatasetSamplePreview(BaseModel):
    sample_id: str
    split: str
    status: str
    true_class_id: Optional[int] = None
    true_class_name: Optional[str] = None
    assigned_class_name: Optional[str] = None
    annotator: Optional[str] = None
    image_base64: Optional[str] = None
    feature_summary: Optional[Dict[str, float]] = None
    embedding_preview: Optional[List[float]] = None


class DatasetSamplesResponse(BaseModel):
    dataset_id: str
    total: int
    offset: int
    limit: int
    samples: List[DatasetSamplePreview]


class ModelInfo(BaseModel):
    id: str
    name: str
    architecture: str
    supported_datasets: List[str]
    embedding_dim: int
    supports_mc_dropout: bool
    description: str


class StrategyInfo(BaseModel):
    id: str
    name: str
    category: str
    formula: str
    description: str
    requires_embeddings: bool
    requires_mc_dropout: bool

