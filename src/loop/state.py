"""State and configuration data structures for Active Learning."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import yaml


@dataclass
class LoopConfig:
    """Experiment and loop configuration settings."""

    dataset: str = "cifar10"
    model_type: str = "cifar_cnn"
    strategy: str = "random"
    initial_size: int = 50
    batch_size: int = 25
    max_rounds: int = 10
    target_accuracy: Optional[float] = None
    label_budget: Optional[int] = 300
    patience: int = 3
    min_delta: float = 0.002
    test_ratio: float = 0.2
    val_ratio: float = 0.1
    seed: int = 42
    warm_start: bool = False
    max_epochs: int = 15
    lr: float = 0.001
    label_noise_rate: float = 0.0
    use_synthetic: bool = False
    max_samples: Optional[int] = 1000
    db_url: str = "sqlite:///al_platform.db"

    @classmethod
    def from_yaml(cls, path: str) -> "LoopConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    def to_yaml(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self.__dict__, f)
