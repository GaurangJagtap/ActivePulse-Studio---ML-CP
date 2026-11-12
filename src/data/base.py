"""Abstract base class for dataset adapters."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
import numpy as np


class DatasetAdapter(ABC):
    """Abstract adapter defining common data access methods for image and text datasets."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the dataset (e.g. 'cifar10', 'agnews')."""
        pass

    @property
    @abstractmethod
    def num_classes(self) -> int:
        """Number of target classes."""
        pass

    @property
    @abstractmethod
    def class_names(self) -> List[str]:
        """Names of the target classes."""
        pass

    @abstractmethod
    def get_all_sample_ids(self) -> List[str]:
        """Return full list of unique sample identifiers."""
        pass

    @abstractmethod
    def get_features(self, sample_ids: List[str]) -> np.ndarray:
        """Return feature representations (e.g. images as float32 arrays or embeddings)."""
        pass

    @abstractmethod
    def get_ground_truth(self, sample_ids: List[str]) -> np.ndarray:
        """Return integer true ground-truth labels for specified sample IDs."""
        pass
