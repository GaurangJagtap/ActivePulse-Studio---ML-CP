"""Random sampling baseline query strategy."""

from typing import List, Any
import numpy as np
from src.strategies.base import BaseStrategy


class RandomSamplingStrategy(BaseStrategy):
    """Uniform random selection baseline."""

    @property
    def name(self) -> str:
        return "random"

    def select(
        self,
        pool_sample_ids: List[str],
        pool_features: np.ndarray,
        model: Any,
        k: int,
    ) -> List[str]:
        n_available = len(pool_sample_ids)
        if n_available == 0:
            return []

        k = min(k, n_available)
        chosen_indices = self.rng.choice(n_available, size=k, replace=False)
        return [pool_sample_ids[i] for i in chosen_indices]
