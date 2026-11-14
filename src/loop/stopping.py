"""Stopping rules for Active Learning loop."""

from typing import Optional, Tuple, List


class StoppingRule:
    """Evaluates whether the active learning loop should terminate."""

    def __init__(
        self,
        max_rounds: int = 10,
        target_accuracy: Optional[float] = None,
        label_budget: Optional[int] = None,
        patience: int = 3,
        min_delta: float = 0.002,
    ):
        self.max_rounds = max_rounds
        self.target_accuracy = target_accuracy
        self.label_budget = label_budget
        self.patience = patience
        self.min_delta = min_delta

        self._best_acc: float = -1.0
        self._plateau_counter: int = 0

    def restore_from_history(self, accuracies: List[float]) -> None:
        """Reconstruct plateau tracking state when resuming an interrupted run."""
        self._best_acc = -1.0
        self._plateau_counter = 0
        for acc in accuracies:
            if acc > self._best_acc + self.min_delta:
                self._best_acc = float(acc)
                self._plateau_counter = 0
            else:
                self._plateau_counter += 1

    def should_stop(
        self,
        current_round: int,
        n_labeled: int,
        n_pool: int,
        test_acc: float,
    ) -> Tuple[bool, str]:
        """Check stopping conditions.
        
        Returns:
            Tuple of (should_stop_bool, reason_string)
        """
        if n_pool == 0:
            return True, "Unlabeled pool is exhausted."

        if self.target_accuracy is not None and test_acc >= self.target_accuracy:
            return True, f"Target accuracy ({self.target_accuracy:.4f}) reached: {test_acc:.4f}"

        if self.label_budget is not None and n_labeled >= self.label_budget:
            return True, f"Label budget ({self.label_budget}) reached: {n_labeled}"

        if current_round > self.max_rounds:
            return True, f"Maximum rounds ({self.max_rounds}) reached."

        # Check plateau
        if test_acc > self._best_acc + self.min_delta:
            self._best_acc = test_acc
            self._plateau_counter = 0
        else:
            self._plateau_counter += 1
            if self._plateau_counter >= self.patience:
                return True, f"Accuracy plateaued for {self.patience} consecutive rounds (< {self.min_delta} gain)."

        return False, ""
