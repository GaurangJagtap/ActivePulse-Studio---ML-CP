"""Active Learning Orchestrator managing round loop, training, evaluation, and stopping."""

import time
from typing import Optional, Dict, Any, List
import numpy as np
import torch

from src.data.base import DatasetAdapter
from src.data.db import DatabaseManager
from src.data.pool import PoolManager
from src.models.wrapper import ActiveLearningClassifier
from src.strategies.base import BaseStrategy
from src.labeling.base import BaseLabeler
from src.loop.state import LoopConfig
from src.loop.stopping import StoppingRule
from src.eval.metrics import evaluate_model
from src.eval.learning_curve import LearningCurveTracker


class ActiveLearningOrchestrator:
    """Coordinates the end-to-end active learning loop."""

    def __init__(
        self,
        config: LoopConfig,
        dataset: DatasetAdapter,
        model: ActiveLearningClassifier,
        strategy: BaseStrategy,
        labeler: BaseLabeler,
        db: Optional[DatabaseManager] = None,
    ):
        self.config = config
        self.dataset = dataset
        self.model = model
        self.strategy = strategy
        self.labeler = labeler

        self.db = db or DatabaseManager(db_url=config.db_url)
        self.pool_manager = PoolManager(
            dataset=self.dataset,
            db=self.db,
            test_ratio=config.test_ratio,
            val_ratio=config.val_ratio,
            seed=config.seed,
        )

        self.stopping_rule = StoppingRule(
            max_rounds=config.max_rounds,
            target_accuracy=config.target_accuracy,
            label_budget=config.label_budget,
            patience=config.patience,
            min_delta=config.min_delta,
        )

        self.tracker = LearningCurveTracker(strategy_name=strategy.name)

    def cold_start(self) -> None:
        """Execute Round 0 cold-start: select initial random batch and train base model."""
        labeled_ids, _, _ = self.pool_manager.get_labeled_data()
        if len(labeled_ids) > 0:
            print(f"[Orchestrator] Labeled set already has {len(labeled_ids)} samples. Skipping cold-start.")
            return

        print(f"[Orchestrator] Executing Cold-Start (Round 0): selecting {self.config.initial_size} initial samples...")
        pool_ids, _ = self.pool_manager.get_pool_data()
        rng = np.random.RandomState(self.config.seed)
        initial_k = min(self.config.initial_size, len(pool_ids))
        chosen_indices = rng.choice(len(pool_ids), size=initial_k, replace=False)
        selected_ids = [pool_ids[i] for i in chosen_indices]

        # Label via oracle / labeler
        sids, labels = self.labeler.label(selected_ids)
        self.pool_manager.label_samples(sids, labels, round_num=0, annotator="cold_start")

        # Train initial model
        self._train_and_evaluate(round_num=0)

    def _train_and_evaluate(self, round_num: int) -> Dict[str, float]:
        """Train model on current labeled set, calibrate on val set, and evaluate on test set."""
        labeled_ids, X_labeled, y_labeled = self.pool_manager.get_labeled_data()
        val_ids, X_val, y_val = self.pool_manager.get_val_data()
        test_ids, X_test, y_test = self.pool_manager.get_test_data()

        print(f"[Round {round_num}] Training model on {len(labeled_ids)} labeled samples...")
        t0 = time.time()
        self.model.fit(X_labeled, y_labeled)
        train_time = time.time() - t0

        # Calibrate on validation split
        temp, val_ece = self.model.calibrate(X_val, y_val)

        # Evaluate on strictly holdout fixed test set
        metrics = evaluate_model(self.model, X_test, y_test, use_calibration=True)
        stats = self.pool_manager.get_stats()

        print(
            f"[Round {round_num}] Test Acc: {metrics['accuracy']:.4f} | "
            f"Macro-F1: {metrics['macro_f1']:.4f} | ECE: {metrics['ece']:.4f} | "
            f"Temp: {temp:.2f} | Labeled: {stats['n_labeled']} | Pool: {stats['n_pool']} "
            f"({train_time:.1f}s)"
        )

        # Persist metrics to database and tracker
        self.db.record_round_metrics(
            round_num=round_num,
            strategy=self.strategy.name,
            n_labeled=stats["n_labeled"],
            n_pool=stats["n_pool"],
            test_acc=metrics["accuracy"],
            test_f1=metrics["macro_f1"],
            ece=metrics["ece"],
            temperature=temp,
        )

        self.tracker.log_round(
            round_num=round_num,
            n_labeled=stats["n_labeled"],
            n_pool=stats["n_pool"],
            test_acc=metrics["accuracy"],
            test_f1=metrics["macro_f1"],
            ece=metrics["ece"],
            temperature=temp,
        )

        return metrics

    def run_simulation(self) -> LearningCurveTracker:
        """Run the full active learning simulation loop until stopping condition is met."""
        print("=" * 70)
        print(f"Starting Active Learning Simulation: Strategy='{self.strategy.name}'")
        print(f"Dataset='{self.dataset.name}' | Batch Size={self.config.batch_size} | Seed={self.config.seed}")
        print("=" * 70)

        # Step 1: Cold start
        self.cold_start()

        # Step 2: Round loop
        latest_round = self.db.get_latest_round()
        current_round = latest_round if latest_round >= 0 else 0

        while current_round < self.config.max_rounds:
            current_round += 1
            stats = self.pool_manager.get_stats()
            latest_metrics = self.tracker.history[-1] if self.tracker.history else None
            curr_acc = latest_metrics.test_acc if latest_metrics else 0.0

            # Check stopping rule
            should_stop, reason = self.stopping_rule.should_stop(
                current_round=current_round,
                n_labeled=stats["n_labeled"],
                n_pool=stats["n_pool"],
                test_acc=curr_acc,
            )
            if should_stop:
                print(f"[Orchestrator] Stopping condition triggered: {reason}")
                break

            # Query strategy selects top-k most informative samples
            pool_ids, X_pool = self.pool_manager.get_pool_data()
            if len(pool_ids) == 0:
                print("[Orchestrator] Pool is empty. Stopping.")
                break

            k = min(self.config.batch_size, len(pool_ids))
            print(f"[Round {current_round}] Scoring pool ({len(pool_ids)} items) and selecting {k} samples...")
            selected_ids = self.strategy.select(pool_ids, X_pool, self.model, k=k)

            # Mark queued then label via oracle
            self.pool_manager.mark_queued(selected_ids)
            sids, labels = self.labeler.label(selected_ids)
            self.pool_manager.label_samples(sids, labels, round_num=current_round, annotator="oracle")

            # Retrain and evaluate
            self._train_and_evaluate(round_num=current_round)

        print("=" * 70)
        print("Active Learning Simulation Completed!")
        print("=" * 70)
        return self.tracker
