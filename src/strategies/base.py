"""Base class and common interface for Active Learning query strategies."""

from abc import ABC, abstractmethod
from typing import List, Callable, Tuple, Any
import numpy as np


class BaseStrategy(ABC):
    """Abstract base class for all query strategies."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier of the strategy."""
        pass

    @abstractmethod
    def select(
        self,
        pool_sample_ids: List[str],
        pool_features: np.ndarray,
        model: Any,
        k: int,
    ) -> List[str]:
        """Select top-k most informative samples from the unlabeled pool.
        
        Args:
            pool_sample_ids: List of unique sample IDs in the unlabeled pool.
            pool_features: Feature array of shape (N, ...) corresponding to pool_sample_ids.
            model: Fitted ActiveLearningClassifier.
            k: Number of samples to query.
            
        Returns:
            List of selected sample IDs of length min(k, len(pool_sample_ids)).
        """
        pass

    def as_modal_query(self) -> Callable:
        """Returns a modAL-compatible query callable: query(classifier, X, n_instances)."""
        def _modal_query(classifier: Any, X: np.ndarray, n_instances: int = 1) -> Tuple[np.ndarray, np.ndarray]:
            dummy_ids = [str(i) for i in range(len(X))]
            selected_ids = self.select(dummy_ids, X, classifier, k=n_instances)
            selected_indices = np.array([int(sid) for sid in selected_ids], dtype=int)
            return selected_indices, X[selected_indices]

        return _modal_query
