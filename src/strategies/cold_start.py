"""Cold-start representative selection using k-means clustering on embeddings."""

from typing import List, Any
import numpy as np
from sklearn.cluster import KMeans
from src.strategies.base import BaseStrategy


class ColdStartClusteringStrategy(BaseStrategy):
    """Cold-Start Selector: Clusters unlabeled pool embeddings and selects cluster medoids.
    
    Ensures maximum dataset coverage before the first model is ever trained.
    """

    @property
    def name(self) -> str:
        return "cold_start_kmeans"

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

        # Extract embeddings or flatten features
        if model is not None and hasattr(model, "extract_embeddings") and getattr(model, "initialized_", False):
            embeddings = model.extract_embeddings(pool_features)
        else:
            embeddings = pool_features.reshape(n_available, -1)

        kmeans = KMeans(n_clusters=k, random_state=self.seed, n_init=5)
        cluster_labels = kmeans.fit_predict(embeddings)
        centers = kmeans.cluster_centers_

        selected_indices = []
        for c in range(k):
            members = np.where(cluster_labels == c)[0]
            if len(members) == 0:
                continue
            dists = np.linalg.norm(embeddings[members] - centers[c], axis=1)
            best_idx = members[np.argmin(dists)]
            selected_indices.append(best_idx)

        if len(selected_indices) < k:
            remaining = [i for i in range(n_available) if i not in selected_indices]
            selected_indices.extend(remaining[: k - len(selected_indices)])

        return [pool_sample_ids[i] for i in selected_indices]
