"""Pool manager coordinating splits, active learning pool, and database state."""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from src.data.base import DatasetAdapter
from src.data.db import DatabaseManager


class PoolManager:
    """Manages active learning sample states, data splits, and access to features."""

    def __init__(
        self,
        dataset: DatasetAdapter,
        db: DatabaseManager,
        test_ratio: float = 0.2,
        val_ratio: float = 0.1,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.db = db
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.seed = seed

        self._initialize_splits_if_needed()

    def _initialize_splits_if_needed(self) -> None:
        """Create deterministic splits and populate DB if empty."""
        all_ids = self.dataset.get_all_sample_ids()
        ground_truth = self.dataset.get_ground_truth(all_ids)

        rng = np.random.RandomState(self.seed)
        n_total = len(all_ids)
        indices = np.arange(n_total)
        rng.shuffle(indices)

        n_test = int(n_total * self.test_ratio)
        n_val = int(n_total * self.val_ratio)

        test_idx = set(indices[:n_test])
        val_idx = set(indices[n_test : n_test + n_val])

        splits = []
        for i in range(n_total):
            if i in test_idx:
                splits.append("test")
            elif i in val_idx:
                splits.append("val")
            else:
                splits.append("pool")

        self.db.initialize_samples(
            sample_ids=all_ids,
            splits=splits,
            true_labels=ground_truth.tolist(),
        )

    def get_pool_data(self) -> Tuple[List[str], np.ndarray]:
        """Return (sample_ids, features) of all currently unlabeled pool samples."""
        pool_ids = self.db.get_pool_sample_ids()
        if not pool_ids:
            # Empty pool
            features = np.empty((0,), dtype=np.float32)
            return [], features
        features = self.dataset.get_features(pool_ids)
        return pool_ids, features

    def get_labeled_data(self) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """Return (sample_ids, X_labeled, y_labeled) of all labeled samples."""
        sample_ids, labels = self.db.get_labeled_samples()
        if not sample_ids:
            return [], np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int64)
        features = self.dataset.get_features(sample_ids)
        return sample_ids, features, np.array(labels, dtype=np.int64)

    def get_val_data(self) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """Return (sample_ids, X_val, y_val) for calibration."""
        sample_ids, labels = self.db.get_split_samples("val")
        if not sample_ids:
            return [], np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int64)
        features = self.dataset.get_features(sample_ids)
        return sample_ids, features, np.array(labels, dtype=np.int64)

    def get_test_data(self) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """Return (sample_ids, X_test, y_test) for fixed test evaluation (never queried)."""
        sample_ids, labels = self.db.get_split_samples("test")
        if not sample_ids:
            return [], np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int64)
        features = self.dataset.get_features(sample_ids)
        return sample_ids, features, np.array(labels, dtype=np.int64)

    def mark_queued(self, sample_ids: List[str]) -> None:
        """Mark samples as queued for labeling."""
        self.db.mark_queued(sample_ids)

    def label_samples(
        self,
        sample_ids: List[str],
        labels: List[int],
        round_num: int,
        annotator: str = "oracle",
    ) -> None:
        """Add newly labeled samples to the labeled pool in the DB."""
        self.db.mark_labeled(sample_ids, labels, round_num=round_num, annotator=annotator)

    def get_stats(self) -> Dict[str, int]:
        """Return current counts across all splits and pool states."""
        pool_ids = self.db.get_pool_sample_ids()
        labeled_ids, _ = self.db.get_labeled_samples()
        val_ids, _ = self.db.get_split_samples("val")
        test_ids, _ = self.db.get_split_samples("test")
        return {
            "n_pool": len(pool_ids),
            "n_labeled": len(labeled_ids),
            "n_val": len(val_ids),
            "n_test": len(test_ids),
        }
