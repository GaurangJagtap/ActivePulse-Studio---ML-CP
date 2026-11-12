"""BADGE: Batch Active Learning by Diverse Gradient Embeddings (Ash et al., 2020)."""

from typing import List, Any
import numpy as np
from src.strategies.base import BaseStrategy


class BADGEStrategy(BaseStrategy):
    """BADGE: Computes hypothetical gradient embeddings with respect to the output layer,
    then samples diverse, uncertain batches using k-means++ initialization.
    
    Gradient embedding captures:
    - Uncertainty via gradient magnitude (larger norm = higher uncertainty)
    - Diversity via embedding direction
    """

    @property
    def name(self) -> str:
        return "badge"

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

        # 1. Compute predicted probabilities
        if hasattr(model, "predict_calibrated_proba"):
            probs = model.predict_calibrated_proba(pool_features)
        else:
            probs = model.predict_proba(pool_features)

        # 2. Extract bottleneck feature embeddings: (N, D)
        if hasattr(model, "extract_embeddings"):
            embeddings = model.extract_embeddings(pool_features)
        else:
            embeddings = pool_features.reshape(n_available, -1)

        num_classes = probs.shape[1]
        embed_dim = embeddings.shape[1]

        # 3. Compute hypothetical gradient embeddings: (N, C * D)
        preds = np.argmax(probs, axis=1)  # Pseudo-label = argmax class
        grad_embeddings = np.zeros((n_available, num_classes * embed_dim), dtype=np.float32)

        for c in range(num_classes):
            # Scale factor: (p_c - 1) if c == pred else p_c
            scale = probs[:, c] - (preds == c).astype(np.float32)
            grad_embeddings[:, c * embed_dim : (c + 1) * embed_dim] = embeddings * scale[:, np.newaxis]

        # 4. K-Means++ Sampling on gradient embeddings
        selected_indices = self._kmeans_plus_plus(grad_embeddings, k)
        return [pool_sample_ids[i] for i in selected_indices]

    def _kmeans_plus_plus(self, X: np.ndarray, k: int) -> List[int]:
        """Perform k-means++ seeding to pick k diverse points based on distance squared."""
        n_samples = len(X)
        norms = np.linalg.norm(X, axis=1)

        # First center chosen proportional to gradient norm (uncertainty) or uniform if all zero
        sq_norms = norms ** 2
        sum_sq = np.sum(sq_norms)
        if sum_sq > 0:
            p_init = sq_norms / sum_sq
            first_idx = int(self.rng.choice(n_samples, p=p_init))
        else:
            first_idx = int(self.rng.choice(n_samples))

        selected = [first_idx]
        min_sq_dist = np.sum((X - X[first_idx]) ** 2, axis=1)

        for _ in range(1, k):
            min_sq_dist[selected] = 0.0
            sum_dist = np.sum(min_sq_dist)
            if sum_dist <= 1e-12:
                # Fallback: pick any remaining unselected point
                remaining = [i for i in range(n_samples) if i not in selected]
                if not remaining:
                    break
                selected.append(int(self.rng.choice(remaining)))
            else:
                probs = min_sq_dist / sum_dist
                next_idx = int(self.rng.choice(n_samples, p=probs))
                selected.append(next_idx)
                # Update distances
                new_sq_dist = np.sum((X - X[next_idx]) ** 2, axis=1)
                min_sq_dist = np.minimum(min_sq_dist, new_sq_dist)

        return selected
