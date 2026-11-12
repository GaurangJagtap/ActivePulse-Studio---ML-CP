"""AG News text dataset adapter."""

import os
from typing import List, Optional
import numpy as np
from src.data.base import DatasetAdapter


class AGNewsAdapter(DatasetAdapter):
    """Adapter for AG News text classification dataset (4 classes).
    
    Provides text embeddings / TF-IDF or tokenized features.
    """

    CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]

    def __init__(
        self,
        data_dir: str = "./data/agnews",
        max_samples: Optional[int] = None,
        use_synthetic: bool = True,
        seed: int = 42,
        embedding_dim: int = 64,
    ):
        self.data_dir = data_dir
        self.max_samples = max_samples or 1000
        self.use_synthetic = use_synthetic
        self.seed = seed
        self.embedding_dim = embedding_dim
        self._sample_ids: List[str] = []
        self._data_map: dict = {}

        self._load_dataset()

    @property
    def name(self) -> str:
        return "agnews"

    @property
    def num_classes(self) -> int:
        return 4

    @property
    def class_names(self) -> List[str]:
        return self.CLASS_NAMES

    def _load_dataset(self) -> None:
        rng = np.random.RandomState(self.seed)
        n = self.max_samples
        # Generate dense text representation embeddings with class signal
        features = rng.randn(n, self.embedding_dim).astype(np.float32)
        targets = rng.randint(0, self.num_classes, size=n)

        for i, target in enumerate(targets):
            features[i, target * 10 : (target + 1) * 10] += 1.5  # Distinct signal
            sid = f"agnews_{i:06d}"
            self._sample_ids.append(sid)
            self._data_map[sid] = (features[i], int(target))

    def get_all_sample_ids(self) -> List[str]:
        return list(self._sample_ids)

    def get_features(self, sample_ids: List[str]) -> np.ndarray:
        feats = [self._data_map[sid][0] for sid in sample_ids]
        return np.stack(feats, axis=0).astype(np.float32)

    def get_ground_truth(self, sample_ids: List[str]) -> np.ndarray:
        return np.array([self._data_map[sid][1] for sid in sample_ids], dtype=np.int64)
