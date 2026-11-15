"""Skorch wrapper integrating PyTorch models with modAL and Active Learning."""

from typing import Optional, Union, List, Tuple
import os
import random
import numpy as np
import torch
import torch.nn as nn
from skorch import NeuralNetClassifier
from skorch.dataset import ValidSplit
from src.models.calibration import TemperatureScaler, compute_ece


def seed_everything(seed: int) -> None:
    """Set Python, NumPy, and PyTorch/CUDA RNG seeds for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


class ActiveLearningClassifier(NeuralNetClassifier):
    """Skorch-based neural network classifier with Active Learning extensions.
    
    Provides:
    - Feature embedding extraction via `extract_embeddings`
    - Monte Carlo Dropout uncertainty sampling via `mc_dropout_predict`
    - Temperature scaling calibration on validation split
    - Retrain from scratch (warm_start=False) vs fine-tune (warm_start=True)
    - Full compatibility with modAL.models.ActiveLearner
    """

    def __init__(
        self,
        module,
        *args,
        criterion=nn.CrossEntropyLoss,
        warm_start: bool = False,
        device: str = "cpu",
        train_split=None,  # All provided X_labeled used for training
        batch_size: int = 64,
        max_epochs: int = 15,
        lr: float = 0.001,
        verbose: int = 0,
        **kwargs,
    ):
        self.warm_start = warm_start
        self.temperature_scaler = TemperatureScaler()
        self.is_calibrated = False
        self.current_temperature = 1.0

        super().__init__(
            module,
            *args,
            criterion=criterion,
            warm_start=warm_start,
            device=device,
            train_split=train_split,
            batch_size=batch_size,
            max_epochs=max_epochs,
            lr=lr,
            verbose=verbose,
            iterator_train__drop_last=True,
            **kwargs,
        )

    def fit(self, X, y=None, **fit_params):
        """Fit model. If warm_start is False, completely resets module weights."""
        if not self.warm_start and self.initialized_:
            self.initialize()
        self.is_calibrated = False
        return super().fit(X, y, **fit_params)

    def extract_logits(self, X: np.ndarray, batch_size: int = 128) -> np.ndarray:
        """Extract unnormalized output logits."""
        if not self.initialized_:
            raise RuntimeError("Model must be fitted before extracting logits.")

        self.module_.eval()
        logits_list = []
        n_samples = len(X)

        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                batch_x = torch.as_tensor(X[i : i + batch_size], device=self.device)
                logits = self.module_(batch_x)
                logits_list.append(logits.detach().cpu().numpy())

        return np.concatenate(logits_list, axis=0)

    def extract_embeddings(self, X: np.ndarray, batch_size: int = 128) -> np.ndarray:
        """Extract bottleneck embeddings from the module's feature extractor.
        
        Returns:
            np.ndarray of shape (N, embedding_dim)
        """
        if not self.initialized_:
            raise RuntimeError("Model must be fitted before extracting embeddings.")

        self.module_.eval()
        embeddings_list = []
        n_samples = len(X)

        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                batch_x = torch.as_tensor(X[i : i + batch_size], device=self.device)
                if hasattr(self.module_, "forward_features"):
                    features = self.module_.forward_features(batch_x)
                else:
                    # Fallback to output logits if no feature extractor defined
                    features = self.module_(batch_x)
                embeddings_list.append(features.detach().cpu().numpy())

        return np.concatenate(embeddings_list, axis=0)

    def mc_dropout_predict(self, X: np.ndarray, n_passes: int = 10, batch_size: int = 128) -> np.ndarray:
        """Perform Monte Carlo Dropout forward passes to estimate predictive uncertainty.
        
        Args:
            X: Input samples array of shape (N, ...)
            n_passes: Number of stochastic forward passes (T)
            batch_size: Batch size for forward evaluation
            
        Returns:
            np.ndarray of shape (T, N, num_classes) containing predicted probabilities.
        """
        if not self.initialized_:
            raise RuntimeError("Model must be fitted before MC-dropout passes.")

        # Keep module in train mode so Dropout layers remain active
        self.module_.train()
        n_samples = len(X)
        all_passes = []

        with torch.no_grad():
            for _ in range(n_passes):
                pass_probs = []
                for i in range(0, n_samples, batch_size):
                    batch_x = torch.as_tensor(X[i : i + batch_size], device=self.device)
                    logits = self.module_(batch_x)
                    probs = torch.softmax(logits, dim=-1)
                    pass_probs.append(probs.detach().cpu().numpy())
                all_passes.append(np.concatenate(pass_probs, axis=0))

        # Reset module back to eval mode
        self.module_.eval()
        return np.stack(all_passes, axis=0)  # Shape: (T, N, C)

    def calibrate(self, X_val: np.ndarray, y_val: np.ndarray) -> Tuple[float, float]:
        """Fit TemperatureScaler on validation set.
        
        Returns:
            Tuple of (learned_temperature, val_ece)
        """
        if len(X_val) == 0:
            self.current_temperature = 1.0
            return 1.0, 0.0

        logits_val = self.extract_logits(X_val)
        self.current_temperature = self.temperature_scaler.fit(logits_val, y_val)
        self.is_calibrated = True

        calibrated_probs = self.temperature_scaler.calibrate_probs(logits_val)
        val_ece = compute_ece(calibrated_probs, y_val)
        return self.current_temperature, val_ece

    def predict_calibrated_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities scaled by temperature."""
        logits = self.extract_logits(X)
        if self.is_calibrated:
            return self.temperature_scaler.calibrate_probs(logits)
        # Fallback to standard softmax
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    def save_checkpoint(self, path: str) -> None:
        """Persist PyTorch module weights and calibration state for checkpoint resumption."""
        if not self.initialized_:
            raise RuntimeError("Cannot save checkpoint before model initialization.")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {
            "module_state_dict": {
                k: v.detach().cpu() for k, v in self.module_.state_dict().items()
            },
            "temperature": float(self.current_temperature),
            "is_calibrated": bool(self.is_calibrated),
        }
        torch.save(payload, path)

    def load_checkpoint(self, path: str) -> bool:
        """Restore PyTorch module weights and temperature scaling from disk."""
        if not os.path.exists(path):
            return False
        if not self.initialized_:
            self.initialize()
        payload = torch.load(path, map_location=self.device, weights_only=True)
        self.module_.load_state_dict(payload["module_state_dict"])
        self.module_.to(self.device)
        temp = float(payload.get("temperature", 1.0))
        self.current_temperature = temp
        self.is_calibrated = bool(payload.get("is_calibrated", False))
        self.temperature_scaler.temperature = temp
        self.module_.eval()
        return True
