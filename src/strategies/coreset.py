"""Core-Set (k-center greedy) active learning query strategy on feature embeddings."""

from typing import List, Any, Optional
import numpy as np
from src.strategies.base import BaseStrategy


class CoreSetStrategy(BaseStrategy):
    """Core-Set Selection (Sener & Savarese, 2018).
    
    Selects samples that maximize the minimum distance to already-selected samples
    in the bottleneck feature embedding space (k-center greedy).
    """

    @property
    def name(self) -> str:
        return "coreset"

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

        # 1. Extract feature embeddings
        if hasattr(model, "extract_embeddings"):
            embeddings = model.extract_embeddings(pool_features)
        else:
            embeddings = pool_features.reshape(n_available, -1)

        # 2. Greedy k-center initialization
        # Pick the first center uniformly at random or closest to boundary
        first_center_idx = int(self.rng.choice(n_available))
        selected_indices = [first_center_idx]

        # Initialize min distance array from all points to the first center
        min_distances = np.linalg.norm(embeddings - embeddings[first_center_idx], axis=1)

        # 3. Iteratively pick the point that is furthest from its nearest chosen center
        for _ in range(1, k):
            # Mask out already selected indices
            min_distances[selected_indices] = -1.0
            next_idx = int(np.argmax(min_distances))
            selected_indices.append(next_idx)

            # Update minimum distance to the new center
            new_distances = np.linalg.norm(embeddings - embeddings[next_idx], axis=1)
            min_distances = np.minimum(min_distances, new_distances)

        return [pool_sample_ids[i] for i in selected_indices]
