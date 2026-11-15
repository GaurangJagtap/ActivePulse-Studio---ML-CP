"""Pydantic schemas for Benchmark Arena."""

from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class BenchmarkRunRequest(BaseModel):
    dataset: str = "cifar10"
    strategies: List[str] = Field(
        default=["random", "least_confidence", "margin", "entropy", "bald", "coreset", "badge", "hybrid"]
    )
    seeds: List[int] = Field(default=[42, 43], min_length=1, max_length=5)
    max_samples: int = Field(default=500, ge=10, le=2000)
    initial_size: int = Field(default=30, ge=2, le=200)
    batch_size: int = Field(default=15, ge=1, le=50)
    rounds: int = Field(default=3, ge=1, le=10)
    epochs: int = Field(default=4, ge=1, le=20)
    target_accuracy: float = Field(default=0.25, ge=0.05, le=1.0)
    use_synthetic: bool = True


class BenchmarkSavingsRow(BaseModel):
    strategy: str
    seeds: int
    alc_mean: float
    alc_ci95: float
    alc_display: str
    labels_to_target: Optional[float] = None
    labels_to_target_display: str
    label_reduction_pct: Optional[float] = None
    label_reduction_display: str
    final_accuracy: Optional[float] = None
    final_f1: Optional[float] = None
    final_ece: Optional[float] = None


class BenchmarkCurvePoint(BaseModel):
    strategy: str
    n_labeled: int
    mean_acc: float
    ci95_acc: float
    mean_f1: Optional[float] = None
    mean_ece: Optional[float] = None


class BenchmarkResultsResponse(BaseModel):
    source: str
    dataset: Optional[str] = None
    target_accuracy: Optional[float] = None
    seeds: List[int] = []
    has_learning_curve_image: bool = False
    summary_rows: List[BenchmarkSavingsRow]
    curve_points: List[BenchmarkCurvePoint] = []


class BenchmarkJobStatus(BaseModel):
    job_id: str
    status: Literal["running", "completed", "failed"]
    current_strategy: Optional[str] = None
    current_seed: Optional[int] = None
    completed_steps: int = 0
    total_steps: int = 0
    error_message: Optional[str] = None
    started_at: str
    completed_at: Optional[str] = None

