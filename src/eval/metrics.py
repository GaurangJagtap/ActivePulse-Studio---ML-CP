"""Evaluation metrics for Active Learning rounds."""

from typing import Dict, Any
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss
from src.models.calibration import compute_ece


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    use_calibration: bool = True,
) -> Dict[str, float]:
    """Evaluate model performance on the fixed test set.
    
    Returns:
        Dictionary containing accuracy, macro_f1, ece, and nll.
    """
    if len(X_test) == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0, "ece": 0.0, "nll": 0.0}

    # Extract probabilities (calibrated if available)
    if use_calibration and hasattr(model, "predict_calibrated_proba"):
        probs = model.predict_calibrated_proba(X_test)
    else:
        probs = model.predict_proba(X_test)

    preds = np.argmax(probs, axis=1)

    acc = float(accuracy_score(y_test, preds))
    f1 = float(f1_score(y_test, preds, average="macro", zero_division=0))
    ece = float(compute_ece(probs, y_test))

    try:
        # Clip probabilities to prevent log(0)
        eps = 1e-12
        clipped_probs = np.clip(probs, eps, 1.0 - eps)
        clipped_probs /= np.sum(clipped_probs, axis=1, keepdims=True)
        nll = float(log_loss(y_test, clipped_probs, labels=list(range(probs.shape[1]))))
    except Exception:
        nll = 0.0

    return {
        "accuracy": acc,
        "macro_f1": f1,
        "ece": ece,
        "nll": nll,
    }
