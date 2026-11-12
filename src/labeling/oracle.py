"""Simulated oracle revealing hidden ground-truth labels with optional noise."""

from typing import List, Tuple, Optional
import numpy as np
from src.data.base import DatasetAdapter
from src.labeling.base import BaseLabeler


class SimulatedOracle(BaseLabeler):
    """Reveals hidden ground-truth labels for automated active learning simulation.
    
    Supports:
    - Ground-truth lookup from dataset adapter.
    - Configurable label noise rate (e.g. 0.0 to 0.2) to simulate human error.
    - Deterministic random state for reproducible runs.
    """

    def __init__(
        self,
        dataset: DatasetAdapter,
        noise_rate: float = 0.0,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.noise_rate = max(0.0, min(1.0, noise_rate))
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def label(self, sample_ids: List[str]) -> Tuple[List[str], List[int]]:
        """Reveal labels for the requested samples, injecting noise if configured."""
        if not sample_ids:
            return [], []

        true_labels = self.dataset.get_ground_truth(sample_ids)
        assigned_labels = []
        num_classes = self.dataset.num_classes

        for lbl in true_labels:
            if self.noise_rate > 0.0 and self.rng.rand() < self.noise_rate:
                # Corrupt label to a random distinct class
                noisy_lbl = int(self.rng.choice([c for c in range(num_classes) if c != lbl]))
                assigned_labels.append(noisy_lbl)
            else:
                assigned_labels.append(int(lbl))

        return list(sample_ids), assigned_labels
