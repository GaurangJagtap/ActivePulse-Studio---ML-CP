"""Uncertainty-based active learning query strategies.

Includes:
- Least Confidence: picks samples with lowest maximum class probability.
- Margin Sampling: picks samples with smallest margin between top two predicted classes.
- Entropy Sampling: picks samples with highest predictive Shannon entropy.
"""

from typing import List, Any
import numpy as np
from src.strategies.base import BaseStrategy


class LeastConfidenceStrategy(BaseStrategy):
    """Selects samples with the lowest maximum predicted class probability."""

    @property
    def name(self) -> str:
        return "least_confidence"

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

        # Get calibrated probabilities if available
        if hasattr(model, "predict_calibrated_proba"):
            probs = model.predict_calibrated_proba(pool_features)
        else:
            probs = model.predict_proba(pool_features)

        # Uncertainty score = 1.0 - max(probs)
        max_probs = np.max(probs, axis=1)
        uncertainties = 1.0 - max_probs

        # Top-k most uncertain (highest uncertainty score)
        top_indices = np.argsort(uncertainties)[::-1][:k]
        return [pool_sample_ids[i] for i in top_indices]


class MarginStrategy(BaseStrategy):
    """Selects samples with the smallest difference between top-1 and top-2 class probabilities."""

    @property
    def name(self) -> str:
        return "margin"

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

        if hasattr(model, "predict_calibrated_proba"):
            probs = model.predict_calibrated_proba(pool_features)
        else:
            probs = model.predict_proba(pool_features)

        # Sort class probabilities in descending order along axis 1
        sorted_probs = np.sort(probs, axis=1)[:, ::-1]

        if sorted_probs.shape[1] > 1:
            margin = sorted_probs[:, 0] - sorted_probs[:, 1]
        else:
            margin = sorted_probs[:, 0]

        # Smallest margin = highest ambiguity
        top_indices = np.argsort(margin)[:k]
        return [pool_sample_ids[i] for i in top_indices]


class EntropyStrategy(BaseStrategy):
    """Selects samples with the highest Shannon predictive entropy."""

    @property
    def name(self) -> str:
        return "entropy"

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

        if hasattr(model, "predict_calibrated_proba"):
            probs = model.predict_calibrated_proba(pool_features)
        else:
            probs = model.predict_proba(pool_features)

        # Prevent log(0)
        eps = 1e-12
        clipped_probs = np.clip(probs, eps, 1.0)
        entropy = -np.sum(clipped_probs * np.log2(clipped_probs), axis=1)

        # Highest entropy = highest uncertainty
        top_indices = np.argsort(entropy)[::-1][:k]
        return [pool_sample_ids[i] for i in top_indices]
