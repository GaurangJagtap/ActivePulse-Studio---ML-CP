"""Bayesian Active Learning by Disagreement (BALD) query strategy using MC-Dropout."""

from typing import List, Any
import numpy as np
from src.strategies.base import BaseStrategy


class BALDStrategy(BaseStrategy):
    """BALD: selects samples with the highest mutual information between predictions and model weights.
    
    Uses Monte Carlo Dropout passes to measure epistemic uncertainty (model disagreement).
    Score = H[p(y|x, D)] - E_w[H[p(y|x, w)]]
    """

    def __init__(self, seed: int = 42, n_passes: int = 10):
        super().__init__(seed=seed)
        self.n_passes = n_passes

    @property
    def name(self) -> str:
        return "bald"

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

        if not hasattr(model, "mc_dropout_predict"):
            raise ValueError("Model must implement mc_dropout_predict(X, n_passes) for BALD.")

        # mc_probs shape: (T, N, C)
        mc_probs = model.mc_dropout_predict(pool_features, n_passes=self.n_passes)

        eps = 1e-12
        clipped_mc_probs = np.clip(mc_probs, eps, 1.0)

        # 1. Mean predictive distribution across passes: shape (N, C)
        mean_probs = np.mean(clipped_mc_probs, axis=0)
        # Total entropy of the mean distribution: shape (N,)
        total_entropy = -np.sum(mean_probs * np.log2(mean_probs), axis=1)

        # 2. Average entropy across individual passes: shape (N,)
        pass_entropies = -np.sum(clipped_mc_probs * np.log2(clipped_mc_probs), axis=2)  # (T, N)
        expected_entropy = np.mean(pass_entropies, axis=0)  # (N,)

        # BALD mutual information
        bald_scores = total_entropy - expected_entropy

        # Pick top-k with highest mutual information
        top_indices = np.argsort(bald_scores)[::-1][:k]
        return [pool_sample_ids[i] for i in top_indices]
