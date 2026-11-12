"""Hybrid active learning strategy combining uncertainty filtering with k-means diversity."""

from typing import List, Any
import numpy as np
from sklearn.cluster import KMeans
from src.strategies.base import BaseStrategy


class HybridStrategy(BaseStrategy):
    """Hybrid Strategy (Uncertainty + Diversity).
    
    1. Filters pool down to top-N most uncertain samples (e.g. N = 3 * K).
    2. Runs k-means clustering on their feature embeddings into k clusters.
    3. Selects the sample closest to each cluster center.
    
    Prevents batch active learning from selecting redundant, near-identical uncertain samples.
    """

    def __init__(self, seed: int = 42, uncertainty_multiplier: int = 3):
        super().__init__(seed=seed)
        self.uncertainty_multiplier = uncertainty_multiplier

    @property
    def name(self) -> str:
        return "hybrid"

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
        if k == n_available:
            return list(pool_sample_ids)

        # 1. Compute entropy uncertainty
        if hasattr(model, "predict_calibrated_proba"):
            probs = model.predict_calibrated_proba(pool_features)
        else:
            probs = model.predict_proba(pool_features)

        eps = 1e-12
        clipped_probs = np.clip(probs, eps, 1.0)
        entropy = -np.sum(clipped_probs * np.log2(clipped_probs), axis=1)

        # 2. Filter to top-N most uncertain candidates
        n_candidates = min(k * self.uncertainty_multiplier, n_available)
        top_n_indices = np.argsort(entropy)[::-1][:n_candidates]

        candidate_features = pool_features[top_n_indices]

        # 3. Extract bottleneck embeddings
        if hasattr(model, "extract_embeddings"):
            candidate_embeddings = model.extract_embeddings(candidate_features)
        else:
            candidate_embeddings = candidate_features.reshape(n_candidates, -1)

        # 4. K-Means clustering into k clusters
        kmeans = KMeans(n_clusters=k, random_state=self.seed, n_init=5)
        cluster_labels = kmeans.fit_predict(candidate_embeddings)
        centers = kmeans.cluster_centers_

        # 5. Pick the sample closest to each cluster center
        selected_candidate_indices = []
        for c in range(k):
            cluster_member_indices = np.where(cluster_labels == c)[0]
            if len(cluster_member_indices) == 0:
                continue
            cluster_embeds = candidate_embeddings[cluster_member_indices]
            center = centers[c]
            dists = np.linalg.norm(cluster_embeds - center, axis=1)
            best_idx = cluster_member_indices[np.argmin(dists)]
            selected_candidate_indices.append(best_idx)

        # Handle any edge-case where fewer than k clusters were formed
        if len(selected_candidate_indices) < k:
            remaining = [i for i in range(n_candidates) if i not in selected_candidate_indices]
            needed = k - len(selected_candidate_indices)
            selected_candidate_indices.extend(remaining[:needed])

        # Map back to original pool sample IDs
        final_pool_indices = [top_n_indices[i] for i in selected_candidate_indices]
        return [pool_sample_ids[i] for i in final_pool_indices]
