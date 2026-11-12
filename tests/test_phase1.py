"""Comprehensive unit and integration tests for Phase 1 of the Active Learning Platform."""

import os
import tempfile
import pytest
import numpy as np
import torch

from src.data.cifar10 import CIFAR10Adapter
from src.data.agnews import AGNewsAdapter
from src.data.db import DatabaseManager
from src.data.pool import PoolManager
from src.models.cnn import CIFARConvNet
from src.models.wrapper import ActiveLearningClassifier
from src.models.calibration import TemperatureScaler, compute_ece
from src.strategies.random import RandomSamplingStrategy
from src.labeling.oracle import SimulatedOracle
from src.loop.state import LoopConfig
from src.loop.stopping import StoppingRule
from src.loop.orchestrator import ActiveLearningOrchestrator


@pytest.fixture
def temp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = DatabaseManager(f"sqlite:///{path}")
    yield db
    db.close()
    if os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


def test_data_splits_and_no_leakage(temp_db):
    """Verify that test, validation, and pool splits are strictly disjoint."""
    dataset = CIFAR10Adapter(max_samples=100, use_synthetic=True, seed=42)
    pool_mgr = PoolManager(dataset=dataset, db=temp_db, test_ratio=0.2, val_ratio=0.1, seed=42)

    stats = pool_mgr.get_stats()
    assert stats["n_test"] == 20
    assert stats["n_val"] == 10
    assert stats["n_pool"] == 70

    test_ids, _, _ = pool_mgr.get_test_data()
    val_ids, _, _ = pool_mgr.get_val_data()
    pool_ids, _ = pool_mgr.get_pool_data()

    set_test = set(test_ids)
    set_val = set(val_ids)
    set_pool = set(pool_ids)

    # Strictly disjoint
    assert len(set_test.intersection(set_val)) == 0
    assert len(set_test.intersection(set_pool)) == 0
    assert len(set_val.intersection(set_pool)) == 0


def test_pool_state_machine(temp_db):
    """Verify state transitions: unlabeled -> queued -> labeled."""
    dataset = CIFAR10Adapter(max_samples=50, use_synthetic=True, seed=42)
    pool_mgr = PoolManager(dataset=dataset, db=temp_db, test_ratio=0.2, val_ratio=0.2, seed=42)

    pool_ids, _ = pool_mgr.get_pool_data()
    assert len(pool_ids) == 30

    query_ids = pool_ids[:5]
    pool_mgr.mark_queued(query_ids)

    # After queuing, remaining unlabeled pool should decrease
    rem_pool_ids, _ = pool_mgr.get_pool_data()
    assert len(rem_pool_ids) == 25
    assert not any(qid in rem_pool_ids for qid in query_ids)

    # Label the queued samples
    pool_mgr.label_samples(query_ids, labels=[1, 2, 3, 4, 5], round_num=1, annotator="test_annotator")
    labeled_ids, X_lab, y_lab = pool_mgr.get_labeled_data()
    assert len(labeled_ids) == 5
    assert list(y_lab) == [1, 2, 3, 4, 5]


def test_oracle_labeling():
    """Verify SimulatedOracle label retrieval and noise behavior."""
    dataset = CIFAR10Adapter(max_samples=20, use_synthetic=True, seed=42)
    all_ids = dataset.get_all_sample_ids()[:10]
    ground_truth = dataset.get_ground_truth(all_ids)

    # Noise-free oracle
    oracle_clean = SimulatedOracle(dataset, noise_rate=0.0, seed=42)
    sids, labels = oracle_clean.label(all_ids)
    assert sids == all_ids
    assert np.array_equal(labels, ground_truth)

    # Noisy oracle
    oracle_noisy = SimulatedOracle(dataset, noise_rate=1.0, seed=42)
    _, noisy_labels = oracle_noisy.label(all_ids)
    assert not np.array_equal(noisy_labels, ground_truth)


def test_model_wrapper_capabilities():
    """Verify fit, predict_proba, embeddings, mc_dropout, and calibration."""
    n_samples = 30
    X = np.random.randn(n_samples, 3, 32, 32).astype(np.float32)
    y = np.random.randint(0, 10, size=n_samples).astype(np.int64)

    model = ActiveLearningClassifier(
        module=lambda: CIFARConvNet(num_classes=10),
        max_epochs=2,
        lr=0.01,
        device="cpu",
        batch_size=16,
        verbose=0,
    )

    model.fit(X, y)

    # Probabilities
    probs = model.predict_proba(X[:5])
    assert probs.shape == (5, 10)
    assert np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-5)

    # Embeddings
    embeds = model.extract_embeddings(X[:5])
    assert embeds.shape == (5, 128)

    # MC Dropout
    mc_probs = model.mc_dropout_predict(X[:5], n_passes=4)
    assert mc_probs.shape == (4, 5, 10)

    # Calibration
    X_val = np.random.randn(10, 3, 32, 32).astype(np.float32)
    y_val = np.random.randint(0, 10, size=10).astype(np.int64)
    temp, ece = model.calibrate(X_val, y_val)
    assert temp > 0.0
    cal_probs = model.predict_calibrated_proba(X[:5])
    assert cal_probs.shape == (5, 10)


def test_random_strategy():
    """Verify RandomSamplingStrategy returns k unique valid pool IDs."""
    strategy = RandomSamplingStrategy(seed=42)
    pool_ids = [f"id_{i}" for i in range(50)]
    X_dummy = np.zeros((50, 10))

    selected = strategy.select(pool_ids, X_dummy, model=None, k=10)
    assert len(selected) == 10
    assert len(set(selected)) == 10
    assert all(sid in pool_ids for sid in selected)


def test_end_to_end_smoke_simulation(temp_db):
    """End-to-end smoke test running 2 rounds of active learning simulation."""
    config = LoopConfig(
        dataset="cifar10",
        initial_size=16,
        batch_size=8,
        max_rounds=2,
        max_epochs=2,
        use_synthetic=True,
        max_samples=60,
        db_url=temp_db.db_url,
    )

    dataset = CIFAR10Adapter(max_samples=config.max_samples, use_synthetic=True, seed=config.seed)
    model = ActiveLearningClassifier(
        module=lambda: CIFARConvNet(num_classes=dataset.num_classes),
        max_epochs=config.max_epochs,
        lr=0.01,
        device="cpu",
        batch_size=16,
        verbose=0,
    )
    strategy = RandomSamplingStrategy(seed=config.seed)
    oracle = SimulatedOracle(dataset=dataset, noise_rate=0.0, seed=config.seed)

    orchestrator = ActiveLearningOrchestrator(
        config=config,
        dataset=dataset,
        model=model,
        strategy=strategy,
        labeler=oracle,
        db=temp_db,
    )

    tracker = orchestrator.run_simulation()

    # Verify rounds were logged
    assert len(tracker.history) == 3  # Round 0 (cold start), Round 1, Round 2
    round_nums = [m.round_num for m in tracker.history]
    assert round_nums == [0, 1, 2]

    # Verify labeled samples grew
    n_labeled = [m.n_labeled for m in tracker.history]
    assert n_labeled[0] == 16
    assert n_labeled[1] == 24
    assert n_labeled[2] == 32

    # Verify metrics were stored in DB
    db_history = temp_db.get_all_rounds_history()
    assert len(db_history) == 3
    assert db_history[0]["round_num"] == 0
    assert db_history[2]["round_num"] == 2
