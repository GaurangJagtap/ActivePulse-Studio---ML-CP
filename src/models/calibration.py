"""Temperature scaling for probability calibration on validation split."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Compute Expected Calibration Error (ECE).
    
    ECE partitions predictions into M equally-spaced confidence bins and computes
    the weighted average of the absolute difference between accuracy and confidence.
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(np.float32)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_samples = len(labels)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


class TemperatureScaler(nn.Module):
    """Calibrates model logits via Temperature Scaling (Guo et al., 2017).
    
    Divides unnormalized logits by a single learnable scalar T > 0:
        p_i = exp(z_i / T) / sum_j exp(z_j / T)
    Optimized by minimizing CrossEntropyLoss on validation logits.
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by temperature."""
        return logits / self.temperature

    def fit(self, logits: np.ndarray, labels: np.ndarray, max_iter: int = 50, lr: float = 0.01) -> float:
        """Optimize temperature parameter on validation set using L-BFGS.
        
        Args:
            logits: Array of shape (N, C) containing unnormalized model logits.
            labels: Array of shape (N,) containing ground-truth class indices.
            
        Returns:
            The learned temperature scalar T.
        """
        if len(logits) == 0 or len(labels) == 0:
            return 1.0

        logits_tensor = torch.tensor(logits, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval_loss():
            optimizer.zero_grad()
            scaled_logits = self.forward(logits_tensor)
            loss = criterion(scaled_logits, labels_tensor)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        
        # Enforce positive temperature floor
        with torch.no_grad():
            self.temperature.clamp_(min=0.01, max=10.0)

        return float(self.temperature.item())

    def calibrate_probs(self, logits: np.ndarray) -> np.ndarray:
        """Convert unnormalized logits into calibrated softmax probabilities."""
        with torch.no_grad():
            logits_tensor = torch.tensor(logits, dtype=torch.float32)
            scaled = self.forward(logits_tensor)
            probs = torch.softmax(scaled, dim=-1).cpu().numpy()
        return probs
