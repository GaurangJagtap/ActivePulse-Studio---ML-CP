"""Abstract labeling interface."""

from abc import ABC, abstractmethod
from typing import List, Tuple


class BaseLabeler(ABC):
    """Base interface for sample labeling (Oracle or Human via Label Studio)."""

    @abstractmethod
    def label(self, sample_ids: List[str]) -> Tuple[List[str], List[int]]:
        """Given a list of sample_ids, return (sample_ids, assigned_labels)."""
        pass
