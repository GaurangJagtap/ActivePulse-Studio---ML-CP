"""Continuous Learning Engine: automatic retraining from human feedback, model versioning & safe updating."""

import os
import threading
import time
import uuid
from typing import Any, Dict, List, Optional
import numpy as np
import torch

from backend.api.schemas.annotations import (
    ContinuousLearningStatusResponse,
    ContinuousModelMetrics,
    ContinuousRetrainEventItem,
)
from backend.api.services.dataset_service import get_cached_dataset
from backend.api.services.experiment_service import (
    get_experiment_db,
    get_experiment_detail,
    get_experiment_runtime,
    _append_log,
    _checkpoints_dir,
    _get_db_counts,
    _read_json,
    _update_meta,
    _write_json,
)
from src.eval.metrics import evaluate_model
from src.models.wrapper import ActiveLearningClassifier

# Thread-safe lock per experiment to prevent concurrent retraining overlaps
_EXP_RETRAIN_LOCKS: Dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _get_exp_lock(exp_id: str) -> threading.Lock:
    with _LOCKS_GUARD:
        if exp_id not in _EXP_RETRAIN_LOCKS:
            _EXP_RETRAIN_LOCKS[exp_id] = threading.Lock()
        return _EXP_RETRAIN_LOCKS[exp_id]


def trigger_continuous_retraining_async(
    exp_id: str,
    trigger_type: str = "auto",
    trigger_sample_id: Optional[str] = None,
    force: bool = False,
) -> str:
    """Launch background continuous retraining job if not already retraining."""
    job_id = f"cl_{uuid.uuid4().hex[:8]}"
    t = threading.Thread(
        target=_run_continuous_retrain_job,
        args=(exp_id, job_id, trigger_type, trigger_sample_id, force),
        daemon=True,
    )
    t.start()
    return job_id


def _run_continuous_retrain_job(
    exp_id: str,
    job_id: str,
    trigger_type: str,
    trigger_sample_id: Optional[str],
    force: bool,
) -> None:
    exp_lock = _get_exp_lock(exp_id)
    acquired = exp_lock.acquire(blocking=False)
    if not acquired:
        _append_log(exp_id, f"[Continuous Learning] Retraining already active; queued feedback will be absorbed in ongoing cycle.")
        return

    try:
        db = get_experiment_db(exp_id)
        detail = get_experiment_detail(exp_id)
        summary = detail.summary
        cfg = detail.config

        pending_feedback = db.get_pending_continuous_feedback()
        if not pending_feedback and not force:
            return

        feedback_ids = [f["id"] for f in pending_feedback]
        feedback_count = len(feedback_ids)

        # Baseline evaluation
        ctx = get_experiment_runtime(exp_id)
        dataset = get_cached_dataset(
            dataset_id=summary.dataset,
            max_samples=int(cfg.get("max_samples", 450)),
            use_synthetic=bool(cfg.get("use_synthetic", True)),
            seed=int(cfg.get("seed", 42)),
        )

        orch = ctx.orchestrator if ctx else None
        latest_version = db.get_latest_model_version()
        prev_acc = latest_version.get("test_acc") if latest_version else None
        prev_f1 = latest_version.get("test_f1") if latest_version else None
        prev_ece = latest_version.get("test_ece") if latest_version else None
        prev_ver_str = f"v{latest_version.get('round_num', 0)}" if latest_version else "v0_baseline"

        candidate_round = (latest_version.get("round_num", 0) if latest_version else 0) + 1
        candidate_ver_str = f"v{candidate_round}_candidate"

        # Record event in DB as training
        db.record_retrain_event(
            event_id=job_id,
            trigger_type=trigger_type,
            trigger_sample_id=trigger_sample_id,
            samples_incorporated=feedback_count,
            total_training_samples=0,
            prev_model_version=prev_ver_str,
            candidate_model_version=candidate_ver_str,
            prev_accuracy=prev_acc,
            prev_macro_f1=prev_f1,
            prev_ece=prev_ece,
            status="training",
            decision_reason=f"Learning from {feedback_count} verified human sample(s).",
        )
        db.record_job(
            job_id=job_id,
            job_type="continuous_retrain",
            status="running",
            progress_pct=25.0,
            current_stage="Preparing accumulated training data",
        )
        _append_log(
            exp_id,
            f"[Continuous Learning] Retraining triggered ({trigger_type}): incorporating {feedback_count} verified human sample(s)...",
        )

        # Retrieve all labeled data from pool manager (which stores authoritative human labels)
        if orch is not None:
            pool_mgr = orch.pool_manager
        else:
            from src.data.pool import PoolManager
            pool_mgr = PoolManager(
                dataset=dataset,
                db=db,
                test_ratio=float(cfg.get("test_ratio", 0.2)),
                val_ratio=float(cfg.get("val_ratio", 0.1)),
                seed=int(cfg.get("seed", 42)),
            )

        labeled_ids, X_labeled, y_labeled = pool_mgr.get_labeled_data()
        val_ids, X_val, y_val = pool_mgr.get_val_data()
        test_ids, X_test, y_test = pool_mgr.get_test_data()

        total_samples = len(labeled_ids)
        if total_samples == 0:
            db.update_retrain_event(event_id=job_id, status="failed", decision_reason="No labeled training data available.")
            return

        # Train a candidate model
        db.record_job(job_id=job_id, job_type="continuous_retrain", status="running", progress_pct=50.0, current_stage="Fitting candidate neural net")
        
        # Instantiate candidate model wrapper
        if orch is not None:
            candidate_model = ActiveLearningClassifier(
                module=orch.model.module,
                batch_size=orch.model.batch_size,
                max_epochs=orch.model.max_epochs,
                lr=orch.model.lr,
                device=orch.model.device,
                verbose=0,
            )
        else:
            # Fallback based on dataset type
            if dataset.name == "cifar10":
                from src.models.cnn import CIFARConvNet
                candidate_model = ActiveLearningClassifier(
                    module=CIFARConvNet,
                    batch_size=int(cfg.get("batch_size", 32)),
                    max_epochs=int(cfg.get("max_epochs", 15)),
                    lr=float(cfg.get("lr", 0.001)),
                    verbose=0,
                )
            else:
                from src.models.text_net import TextMLP
                candidate_model = ActiveLearningClassifier(
                    module=TextMLP,
                    batch_size=int(cfg.get("batch_size", 32)),
                    max_epochs=int(cfg.get("max_epochs", 15)),
                    lr=float(cfg.get("lr", 0.001)),
                    verbose=0,
                )

        t0 = time.time()
        candidate_model.fit(X_labeled, y_labeled)
        train_time = time.time() - t0

        # Calibrate & evaluate candidate model
        db.record_job(job_id=job_id, job_type="continuous_retrain", status="running", progress_pct=75.0, current_stage="Evaluating candidate holdout accuracy")
        temp, val_ece = candidate_model.calibrate(X_val, y_val)
        candidate_metrics = evaluate_model(candidate_model, X_test, y_test, use_calibration=True)
        
        cand_acc = float(candidate_metrics["accuracy"])
        cand_f1 = float(candidate_metrics["macro_f1"])
        cand_ece = float(candidate_metrics["ece"])
        cand_nll = float(candidate_metrics.get("nll", 0.0))

        # Decision rule: Safe model promotion
        # If no previous accuracy existed, accept baseline.
        # Otherwise, accept if accuracy is >= (prev_acc - allowed_margin), e.g. within 2.5% or improved
        is_promoted = False
        decision_reason = ""
        if prev_acc is None:
            is_promoted = True
            decision_reason = f"Initial candidate promoted as active model (Acc: {cand_acc*100:.1f}%)."
        else:
            # Acceptance tolerance of 0.025 to allow stochasticity while protecting against degradation
            if cand_acc >= (prev_acc - 0.025):
                is_promoted = True
                delta = cand_acc - prev_acc
                sign = "+" if delta >= 0 else ""
                decision_reason = f"Candidate promoted: holdout accuracy {cand_acc*100:.1f}% ({sign}{delta*100:.1f}%), F1 {cand_f1*100:.1f}%."
            else:
                is_promoted = False
                decision_reason = f"Candidate rejected: holdout accuracy dropped from {prev_acc*100:.1f}% to {cand_acc*100:.1f}%. Active model preserved."

        ckpt_dir = _checkpoints_dir(exp_id)
        os.makedirs(ckpt_dir, exist_ok=True)
        cand_ckpt_path = os.path.join(ckpt_dir, f"{candidate_ver_str}.pt")
        candidate_model.save_checkpoint(cand_ckpt_path)

        if is_promoted:
            # Adopt candidate into active orchestrator
            if orch is not None:
                orch.model = candidate_model
            # Save latest checkpoint
            latest_ckpt = os.path.join(ckpt_dir, "latest.pt")
            candidate_model.save_checkpoint(latest_ckpt)

            db.save_model_version(
                round_num=candidate_round,
                model_type=summary.model_type,
                checkpoint_path=cand_ckpt_path,
                temperature=temp,
                val_ece=val_ece,
                test_acc=cand_acc,
                test_f1=cand_f1,
                test_ece=cand_ece,
                test_nll=cand_nll,
                train_time_sec=train_time,
            )
            final_status = "promoted"
        else:
            final_status = "rejected"

        # Mark feedback rows as used
        db.mark_feedback_used_in_training(feedback_ids, job_id)

        # Update event record
        db.update_retrain_event(
            event_id=job_id,
            candidate_accuracy=cand_acc,
            candidate_macro_f1=cand_f1,
            candidate_ece=cand_ece,
            status=final_status,
            decision_reason=decision_reason,
        )
        db.record_job(
            job_id=job_id,
            job_type="continuous_retrain",
            status="completed",
            progress_pct=100.0,
            current_stage="Model update finalized",
        )
        _append_log(exp_id, f"[Continuous Learning] {decision_reason}")

        # Update experiment summary metrics
        if is_promoted:
            _update_meta(
                exp_id,
                latest_accuracy=cand_acc,
                latest_f1=cand_f1,
                latest_ece=cand_ece,
            )

    except Exception as exc:
        _append_log(exp_id, f"[Continuous Learning] Retraining failed: {exc}")
        try:
            db = get_experiment_db(exp_id)
            db.record_job(job_id=job_id, job_type="continuous_retrain", status="failed", progress_pct=0.0, error_message=str(exc))
            db.update_retrain_event(event_id=job_id, status="failed", decision_reason=f"Error: {exc}")
        except Exception:
            pass
    finally:
        exp_lock.release()


def get_continuous_learning_status(exp_id: str) -> ContinuousLearningStatusResponse:
    detail = get_experiment_detail(exp_id)
    summary = detail.summary
    db = get_experiment_db(exp_id)

    feedback_stats = db.get_continuous_feedback_stats()
    raw_events = db.list_retrain_events(limit=30)
    latest_version = db.get_latest_model_version()
    
    # Check if a retrain job is actively running
    jobs = db.list_jobs()
    cl_running_job = next((j for j in jobs if j["job_type"] == "continuous_retrain" and j["status"] == "running"), None)

    retraining_status = "waiting_for_feedback"
    status_msg = "Continuous model is idle, waiting for human annotations or corrections."

    if cl_running_job:
        stage = (cl_running_job.get("current_stage") or "").lower()
        if "evaluating" in stage:
            retraining_status = "evaluating_candidate"
            status_msg = "Evaluating candidate model against holdout test set..."
        elif "preparing" in stage:
            retraining_status = "preparing_data"
            status_msg = "Preparing accumulated training data with newly verified labels..."
        else:
            retraining_status = "retraining_in_progress"
            status_msg = "Retraining candidate PyTorch model on accumulated knowledge..."
    elif feedback_stats["pending_retrain"] > 0:
        retraining_status = "feedback_received"
        status_msg = f"{feedback_stats['pending_retrain']} verified sample(s) received and ready for model training."
    elif raw_events:
        latest_event = raw_events[0]
        if latest_event["status"] == "promoted":
            retraining_status = "model_updated_successfully"
            status_msg = latest_event.get("decision_reason") or "Candidate model passed holdout criteria and was promoted."
        elif latest_event["status"] == "rejected":
            retraining_status = "candidate_rejected"
            status_msg = latest_event.get("decision_reason") or "Candidate failed validation criteria; active model preserved."
        elif latest_event["status"] == "failed":
            retraining_status = "retraining_failed"
            status_msg = latest_event.get("decision_reason") or "Continuous retraining encountered an error."

    # Parse versions & metrics
    active_version_str = f"v{latest_version.get('round_num', 0)}" if latest_version else "v0_baseline"
    active_metrics = ContinuousModelMetrics(
        accuracy=latest_version.get("test_acc") if latest_version else None,
        macro_f1=latest_version.get("test_f1") if latest_version else None,
        ece=latest_version.get("test_ece") if latest_version else None,
        nll=latest_version.get("test_nll") if latest_version else None,
    )

    prev_metrics: Optional[ContinuousModelMetrics] = None
    delta_metrics: Optional[Dict[str, float]] = None
    prev_ver_str: Optional[str] = None
    latest_cand_ver: Optional[str] = None
    last_update_ts: Optional[str] = None

    if raw_events:
        latest_event = raw_events[0]
        prev_ver_str = latest_event.get("prev_model_version")
        latest_cand_ver = latest_event.get("candidate_model_version")
        last_update_ts = latest_event.get("completed_at") or latest_event.get("created_at")

        if latest_event.get("prev_accuracy") is not None:
            prev_metrics = ContinuousModelMetrics(
                accuracy=latest_event.get("prev_accuracy"),
                macro_f1=latest_event.get("prev_macro_f1"),
                ece=latest_event.get("prev_ece"),
            )
            if active_metrics.accuracy is not None and prev_metrics.accuracy is not None:
                delta_metrics = {
                    "accuracy": round(active_metrics.accuracy - prev_metrics.accuracy, 4),
                    "macro_f1": round((active_metrics.macro_f1 or 0.0) - (prev_metrics.macro_f1 or 0.0), 4),
                    "ece": round((active_metrics.ece or 0.0) - (prev_metrics.ece or 0.0), 4),
                }

    counts = _get_db_counts(db)
    total_training_samples = counts.get("n_labeled", 0)

    # Format event items
    history_items: List[ContinuousRetrainEventItem] = []
    for ev in raw_events:
        history_items.append(
            ContinuousRetrainEventItem(
                event_id=ev["event_id"],
                trigger_type=ev["trigger_type"],
                trigger_sample_id=ev["trigger_sample_id"],
                samples_incorporated=ev["samples_incorporated"],
                total_training_samples=ev["total_training_samples"],
                prev_model_version=ev["prev_model_version"],
                candidate_model_version=ev["candidate_model_version"],
                prev_metrics=ContinuousModelMetrics(
                    accuracy=ev["prev_accuracy"],
                    macro_f1=ev["prev_macro_f1"],
                    ece=ev["prev_ece"],
                ),
                candidate_metrics=ContinuousModelMetrics(
                    accuracy=ev["candidate_accuracy"],
                    macro_f1=ev["candidate_macro_f1"],
                    ece=ev["candidate_ece"],
                ),
                status=ev["status"],
                decision_reason=ev["decision_reason"],
                created_at=ev["created_at"],
                completed_at=ev["completed_at"],
            )
        )

    return ContinuousLearningStatusResponse(
        experiment_id=exp_id,
        experiment_name=summary.name,
        dataset=summary.dataset,
        model_type=summary.model_type,
        total_human_verified=feedback_stats["total_verified"],
        newly_corrected=feedback_stats["corrections"],
        confirmed_predictions=feedback_stats["confirmed"],
        samples_awaiting_training=feedback_stats["pending_retrain"],
        total_training_samples=total_training_samples,
        retraining_status=retraining_status,
        status_message=status_msg,
        active_model_version=active_version_str,
        previous_model_version=prev_ver_str,
        latest_candidate_version=latest_cand_ver,
        last_successful_update=last_update_ts,
        active_metrics=active_metrics,
        previous_metrics=prev_metrics,
        delta_metrics=delta_metrics,
        history=history_items,
    )

