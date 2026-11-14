"""Active Learning Orchestrator managing round loop, checkpointing, resumption, evaluation, and stopping."""

import json
import os
import random
import threading
import time
from typing import Any, Callable, Dict, List, Optional
import numpy as np
import torch

from src.data.base import DatasetAdapter
from src.data.db import DatabaseManager
from src.data.pool import PoolManager
from src.eval.learning_curve import LearningCurveTracker
from src.eval.metrics import evaluate_model
from src.labeling.base import BaseLabeler
from src.loop.state import LoopConfig
from src.loop.stopping import StoppingRule
from src.models.wrapper import ActiveLearningClassifier, seed_everything
from src.strategies.base import BaseStrategy


class ActiveLearningOrchestrator:
    """Coordinates the end-to-end active learning loop with persistent resumption."""

    def __init__(
        self,
        config: LoopConfig,
        dataset: DatasetAdapter,
        model: ActiveLearningClassifier,
        strategy: BaseStrategy,
        labeler: BaseLabeler,
        db: Optional[DatabaseManager] = None,
        checkpoint_dir: Optional[str] = None,
        cancel_event: Optional[threading.Event] = None,
        on_round_complete: Optional[Callable[[int, Dict[str, float], float], None]] = None,
    ):
        self.config = config
        self.dataset = dataset
        self.model = model
        self.strategy = strategy
        self.labeler = labeler
        self.checkpoint_dir = checkpoint_dir
        self.cancel_event = cancel_event
        self.on_round_complete = on_round_complete
        self.last_stop_reason: Optional[str] = None
        self.cancelled: bool = False

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
        self._restore_from_persistence()

    def _rng_state_path(self) -> Optional[str]:
        if not self.checkpoint_dir:
            return None
        return os.path.join(self.checkpoint_dir, "rng_state.json")

    def _save_rng_state(self) -> None:
        path = self._rng_state_path()
        if not path:
            return
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        np_state = np.random.get_state()
        py_state = random.getstate()
        payload = {
            "python_random_version": py_state[0],
            "python_random_state": list(py_state[1]),
            "python_random_gauss": py_state[2],
            "numpy_state": [np_state[0], np_state[1].tolist(), int(np_state[2]), int(np_state[3]), float(np_state[4])],
            "torch_rng_state": torch.get_rng_state().tolist(),
        }
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp, path)

    def _restore_rng_state(self) -> None:
        path = self._rng_state_path()
        if not path or not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            random.setstate(
                (
                    payload["python_random_version"],
                    tuple(payload["python_random_state"]),
                    payload["python_random_gauss"],
                )
            )
            np_raw = payload["numpy_state"]
            np.random.set_state(
                (
                    np_raw[0],
                    np.array(np_raw[1], dtype=np.uint32),
                    np_raw[2],
                    np_raw[3],
                    np_raw[4],
                )
            )
            torch.set_rng_state(torch.tensor(payload["torch_rng_state"], dtype=torch.uint8))
        except Exception:
            seed_everything(self.config.seed)

    def _restore_from_persistence(self) -> None:
        """Restore round metrics, stopping state, latest model checkpoint, and RNG state on resume."""
        history = self.db.get_all_rounds_history()
        if history:
            for row in history:
                self.tracker.log_round(
                    round_num=row["round_num"],
                    n_labeled=row["n_labeled"],
                    n_pool=row["n_pool"],
                    test_acc=row["test_acc"],
                    test_f1=row["test_f1"],
                    ece=row.get("ece"),
                    temperature=row.get("temperature"),
                    nll=row.get("nll"),
                    train_time=row.get("train_time_sec"),
                )
            self.stopping_rule.restore_from_history([r["test_acc"] for r in history])

            # Restore latest saved checkpoint if available
            ckpt_candidates: List[str] = []
            latest_ver = self.db.get_latest_model_version()
            if latest_ver and latest_ver.get("checkpoint_path"):
                ckpt_candidates.append(latest_ver["checkpoint_path"])
            if self.checkpoint_dir:
                ckpt_candidates.append(os.path.join(self.checkpoint_dir, "latest.pt"))

            for ckpt_path in ckpt_candidates:
                if ckpt_path and os.path.exists(ckpt_path):
                    try:
                        # Ensure module shape is initialized if skorch needs input dimensions
                        if not getattr(self.model, "initialized_", False):
                            self.model.initialize()
                        self.model.load_checkpoint(ckpt_path)
                        break
                    except Exception:
                        pass

            self._restore_rng_state()

    def _check_cancelled(self) -> bool:
        if self.cancel_event is not None and self.cancel_event.is_set():
            self.cancelled = True
            self.last_stop_reason = "Experiment cancelled safely by user."
            return True
        return False

    def cold_start(self) -> None:
        """Execute Round 0 cold-start: select initial random batch and train base model."""
        if self._check_cancelled():
            return

        labeled_ids, _, _ = self.pool_manager.get_labeled_data()
        latest_round = self.db.get_latest_round()
        if len(labeled_ids) > 0 and latest_round >= 0:
            print(
                f"[Orchestrator] Labeled set already has {len(labeled_ids)} samples "
                f"(Round {latest_round} recorded). Resuming without overwriting Round 0."
            )
            return

        if len(labeled_ids) == 0:
            print(
                f"[Orchestrator] Executing Cold-Start (Round 0): selecting "
                f"{self.config.initial_size} initial samples..."
            )
            pool_ids, _ = self.pool_manager.get_pool_data()
            rng = np.random.RandomState(self.config.seed)
            initial_k = min(self.config.initial_size, len(pool_ids))
            chosen_indices = rng.choice(len(pool_ids), size=initial_k, replace=False)
            selected_ids = [pool_ids[i] for i in chosen_indices]

            sids, labels = self.labeler.label(selected_ids)
            self.pool_manager.label_samples(sids, labels, round_num=0, annotator="cold_start")

        if latest_round < 0:
            self._train_and_evaluate(round_num=0)

    def _train_and_evaluate(self, round_num: int) -> Dict[str, float]:
        """Train model on current labeled set, calibrate on val set, evaluate, and checkpoint."""
        labeled_ids, X_labeled, y_labeled = self.pool_manager.get_labeled_data()
        val_ids, X_val, y_val = self.pool_manager.get_val_data()
        test_ids, X_test, y_test = self.pool_manager.get_test_data()

        print(f"[Round {round_num}] Training model on {len(labeled_ids)} labeled samples...")
        t0 = time.time()
        self.model.fit(X_labeled, y_labeled)
        train_time = time.time() - t0

        temp, val_ece = self.model.calibrate(X_val, y_val)
        metrics = evaluate_model(self.model, X_test, y_test, use_calibration=True)
        stats = self.pool_manager.get_stats()

        print(
            f"[Round {round_num}] Test Acc: {metrics['accuracy']:.4f} | "
            f"Macro-F1: {metrics['macro_f1']:.4f} | ECE: {metrics['ece']:.4f} | "
            f"NLL: {metrics['nll']:.4f} | Temp: {temp:.2f} | "
            f"Labeled: {stats['n_labeled']} | Pool: {stats['n_pool']} ({train_time:.1f}s)"
        )

        self.db.record_round_metrics(
            round_num=round_num,
            strategy=self.strategy.name,
            n_labeled=stats["n_labeled"],
            n_pool=stats["n_pool"],
            test_acc=metrics["accuracy"],
            test_f1=metrics["macro_f1"],
            ece=metrics["ece"],
            temperature=temp,
            nll=metrics["nll"],
            train_time_sec=train_time,
        )

        self.tracker.log_round(
            round_num=round_num,
            n_labeled=stats["n_labeled"],
            n_pool=stats["n_pool"],
            test_acc=metrics["accuracy"],
            test_f1=metrics["macro_f1"],
            ece=metrics["ece"],
            temperature=temp,
            nll=metrics["nll"],
            train_time=train_time,
        )

        if self.checkpoint_dir:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            round_ckpt = os.path.join(self.checkpoint_dir, f"round_{round_num}.pt")
            latest_ckpt = os.path.join(self.checkpoint_dir, "latest.pt")
            self.model.save_checkpoint(round_ckpt)
            self.model.save_checkpoint(latest_ckpt)
            self.db.save_model_version(
                round_num=round_num,
                model_type=self.config.model_type,
                checkpoint_path=round_ckpt,
                temperature=temp,
                val_ece=val_ece,
                test_acc=metrics["accuracy"],
                test_f1=metrics["macro_f1"],
                test_ece=metrics["ece"],
                test_nll=metrics["nll"],
                train_time_sec=train_time,
            )
            self._save_rng_state()

        if self.on_round_complete is not None:
            self.on_round_complete(round_num, metrics, train_time)

        return metrics

    def run_simulation(self) -> LearningCurveTracker:
        """Run or resume the full active learning simulation loop until stopping condition is met."""
        print("=" * 70)
        print(f"Starting Active Learning Simulation: Strategy='{self.strategy.name}'")
        print(f"Dataset='{self.dataset.name}' | Batch Size={self.config.batch_size} | Seed={self.config.seed}")
        print("=" * 70)

        self.cold_start()
        if self._check_cancelled():
            return self.tracker

        latest_round = self.db.get_latest_round()
        current_round = latest_round if latest_round >= 0 else 0

        # Recover any interrupted queued batch left mid-round before crash
        leftover_queued = self.db.get_queued_sample_ids()
        if leftover_queued and current_round < self.config.max_rounds:
            next_round = current_round + 1
            sids, labels = self.labeler.label(leftover_queued)
            self.pool_manager.label_samples(sids, labels, round_num=next_round, annotator="oracle")
            self._train_and_evaluate(round_num=next_round)
            current_round = next_round

        while current_round < self.config.max_rounds:
            if self._check_cancelled():
                break

            next_round = current_round + 1
            stats = self.pool_manager.get_stats()
            latest_metrics = self.tracker.history[-1] if self.tracker.history else None
            curr_acc = latest_metrics.test_acc if latest_metrics else 0.0

            should_stop, reason = self.stopping_rule.should_stop(
                current_round=next_round,
                n_labeled=stats["n_labeled"],
                n_pool=stats["n_pool"],
                test_acc=curr_acc,
            )
            if should_stop:
                self.last_stop_reason = reason
                print(f"[Orchestrator] Stopping condition triggered: {reason}")
                break

            pool_ids, X_pool = self.pool_manager.get_pool_data()
            if len(pool_ids) == 0:
                self.last_stop_reason = "Unlabeled pool is exhausted."
                print("[Orchestrator] Pool is empty. Stopping.")
                break

            current_round = next_round
            k = min(self.config.batch_size, len(pool_ids))
            print(f"[Round {current_round}] Scoring pool ({len(pool_ids)} items) and selecting {k} samples...")
            selected_ids = self.strategy.select(pool_ids, X_pool, self.model, k=k)

            self.pool_manager.mark_queued(selected_ids)
            sids, labels = self.labeler.label(selected_ids)
            self.pool_manager.label_samples(sids, labels, round_num=current_round, annotator="oracle")

            if self._check_cancelled():
                break

            self._train_and_evaluate(round_num=current_round)

        print("=" * 70)
        print("Active Learning Simulation Completed!")
        print("=" * 70)
        return self.tracker
