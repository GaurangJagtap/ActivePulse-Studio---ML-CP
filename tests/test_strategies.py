"""Unit tests for all Active Learning query strategies and evaluation calculations."""

import os
import tempfile
import pytest
import numpy as np
import torch

from src.models.cnn import CIFARConvNet
from src.models.wrapper import ActiveLearningClassifier
from src.strategies.registry import list_registered_strategies, get_strategy
from src.eval.savings import compute_alc, labels_to_target_accuracy, compute_label_savings, plot_learning_curves


@pytest.fixture(scope="module")
def fitted_mock_model():
    """Create and fit a lightweight model on synthetic data."""
    X = np.random.randn(40, 3, 32, 32).astype(np.float32)
    y = np.random.randint(0, 5, size=40).astype(np.int64)

    model = ActiveLearningClassifier(
        module=lambda: CIFARConvNet(num_classes=5),
        max_epochs=2,
        lr=0.01,
        device="cpu",
        batch_size=16,
        verbose=0,
    )
    model.fit(X, y)
    return model


@pytest.mark.parametrize(
    "strategy_name",
    [
        "random",
        "least_confidence",
        "margin",
        "entropy",
        "bald",
        "coreset",
        "badge",
        "hybrid",
        "cold_start_kmeans",
    ],
)
def test_all_strategies_selection(strategy_name, fitted_mock_model):
    """Verify that every strategy selects k unique valid IDs from the pool."""
    strategy = get_strategy(strategy_name, seed=42)
    pool_ids = [f"sample_{i:04d}" for i in range(50)]
    pool_features = np.random.randn(50, 3, 32, 32).astype(np.float32)

    k = 12
    selected = strategy.select(pool_ids, pool_features, model=fitted_mock_model, k=k)

    # 1. Correct length
    assert len(selected) == k, f"Strategy {strategy_name} did not return {k} items."

    # 2. No duplicates
    assert len(set(selected)) == k, f"Strategy {strategy_name} returned duplicate sample IDs."

    # 3. All selected are members of the original pool
    assert all(sid in pool_ids for sid in selected), f"Strategy {strategy_name} returned unknown sample ID."


def test_modal_callable_adapter(fitted_mock_model):
    """Verify modAL-compatible callable query adapter."""
    strategy = get_strategy("entropy", seed=42)
    modal_query = strategy.as_modal_query()

    X_pool = np.random.randn(20, 3, 32, 32).astype(np.float32)
    query_idx, query_inst = modal_query(fitted_mock_model, X_pool, n_instances=5)

    assert len(query_idx) == 5
    assert len(set(query_idx)) == 5
    assert query_inst.shape == (5, 3, 32, 32)


def test_compute_alc_and_savings():
    """Verify ALC calculation, target accuracy interpolation, and savings table."""
    labeled_counts = [50, 100, 150, 200]
    accuracies = [0.50, 0.70, 0.85, 0.90]

    # ALC
    alc = compute_alc(labeled_counts, accuracies)
    assert 0.50 <= alc <= 0.90

    # Labels to target (exact and interpolated)
    lbl_exact = labels_to_target_accuracy(labeled_counts, accuracies, target_acc=0.70)
    assert lbl_exact == 100

    lbl_interpolated = labels_to_target_accuracy(labeled_counts, accuracies, target_acc=0.80)
    assert 100 < lbl_interpolated < 150

    # Label savings computation
    benchmark_results = {
        "random": [
            [{"n_labeled": 50, "test_acc": 0.5}, {"n_labeled": 100, "test_acc": 0.7}, {"n_labeled": 150, "test_acc": 0.8}],
            [{"n_labeled": 50, "test_acc": 0.52}, {"n_labeled": 100, "test_acc": 0.68}, {"n_labeled": 150, "test_acc": 0.81}],
        ],
        "hybrid": [
            [{"n_labeled": 50, "test_acc": 0.55}, {"n_labeled": 100, "test_acc": 0.82}, {"n_labeled": 150, "test_acc": 0.88}],
            [{"n_labeled": 50, "test_acc": 0.58}, {"n_labeled": 100, "test_acc": 0.80}, {"n_labeled": 150, "test_acc": 0.87}],
        ],
    }

    df = compute_label_savings(benchmark_results, target_accuracy=0.80, baseline_strategy="random")
    assert len(df) == 2
    assert "Strategy" in df.columns
    assert "% Label Reduction vs Random" in df.columns


def test_plot_learning_curves_generation():
    """Verify learning curves plot is generated without error."""
    benchmark_results = {
        "random": [[{"n_labeled": 20, "test_acc": 0.3}, {"n_labeled": 40, "test_acc": 0.5}]],
        "entropy": [[{"n_labeled": 20, "test_acc": 0.35}, {"n_labeled": 40, "test_acc": 0.6}]],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        out_png = os.path.join(tmpdir, "test_plot.png")
        plot_learning_curves(benchmark_results, output_path=out_png, target_accuracy=0.5)
        assert os.path.exists(out_png)
        assert os.path.getsize(out_png) > 1000
