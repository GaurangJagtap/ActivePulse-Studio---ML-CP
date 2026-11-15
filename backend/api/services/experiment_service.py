"""Experiment management service bridging FastAPI with ActiveLearningOrchestrator and SQLite."""

from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
import shutil
import threading
import uuid
from typing import Any, Dict, List, Optional
import numpy as np
import torch

from backend.api.config import DATA_DIR, DEFAULT_DEVICE
from backend.api.schemas.experiments import (
    CostSavingsSchema,
    ExperimentCompareEntry,
    ExperimentCompareRequest,
    ExperimentCompareResponse,
    ExperimentCreateRequest,
    ExperimentDetailResponse,
    ExperimentEvaluationResponse,
    ExperimentJobsResponse,
    ExperimentProgressResponse,
    ExperimentSummary,
    JobStatusSchema,
    PcaEmbeddingResponse,
    PcaPointSchema,
    RoundMetricSchema,
    SelectedSampleSchema,
    SelectedSamplesResponse,
)
from backend.api.services.dataset_service import get_cached_dataset
from backend.api.services.utils import (
    build_classifier,
    compute_pca_projection,
    estimate_cost_savings,
    feature_to_base64_png,
    summarize_text_embedding,
)
from src.data.db import DatabaseManager, SampleRecord
from src.eval.savings import compute_alc, labels_to_target_accuracy
from src.labeling.oracle import SimulatedOracle
from src.loop.orchestrator import ActiveLearningOrchestrator
from src.loop.state import LoopConfig
from src.models.wrapper import seed_everything
from src.strategies.registry import get_strategy, list_registered_strategies
from src.tracking.mlflow_tracker import MLflowTracker


PIPELINE_STAGES = [
    "Unlabeled Data Intake",
    "PyTorch Retraining",
    "Temperature Calibration",
    "Uncertainty & Diversity Scoring",
    "Holdout Evaluation (Zero Leakage)",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _experiments_root() -> str:
    root = os.path.join(DATA_DIR, "experiments")
    os.makedirs(root, exist_ok=True)
    return root


def _exp_dir(exp_id: str) -> str:
    return os.path.join(_experiments_root(), exp_id)


def _meta_path(exp_id: str) -> str:
    return os.path.join(_exp_dir(exp_id), "meta.json")


def _db_path(exp_id: str) -> str:
    return os.path.join(_exp_dir(exp_id), "al.db")


def _pca_path(exp_id: str) -> str:
    return os.path.join(_exp_dir(exp_id), "pca.json")


def _selected_path(exp_id: str) -> str:
    return os.path.join(_exp_dir(exp_id), "selected.json")


def _checkpoints_dir(exp_id: str) -> str:
    return os.path.join(_exp_dir(exp_id), "checkpoints")


class _RuntimeContext:
    def __init__(self, orchestrator: ActiveLearningOrchestrator, db: DatabaseManager, dataset: Any):
        self.orchestrator = orchestrator
        self.db = db
        self.dataset = dataset


_LOCK = threading.RLock()
_RUNTIMES: Dict[str, _RuntimeContext] = {}
_CANCEL_EVENTS: Dict[str, threading.Event] = {}


def _configured_db_url(exp_id: str) -> str:
    env_url = os.environ.get("DATABASE_URL", "").strip()
    if env_url:
        return env_url
    return f"sqlite:///{_db_path(exp_id)}"


def _get_cancel_event(exp_id: str, reset: bool = False) -> threading.Event:
    with _LOCK:
        ev = _CANCEL_EVENTS.get(exp_id)
        if ev is None or reset:
            ev = threading.Event()
            _CANCEL_EVENTS[exp_id] = ev
        return ev


def _read_json(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, data: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    # Retry loop for Windows atomic file replacement in concurrent threads
    for attempt in range(5):
        try:
            os.replace(tmp_path, path)
            break
        except (PermissionError, OSError):
            if attempt == 4:
                # Fallback directly to direct write if rename fails
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
                break
            time.sleep(0.05)



def _update_meta(exp_id: str, **updates) -> Dict:
    with _LOCK:
        meta = _read_json(_meta_path(exp_id))
        if meta is None:
            raise KeyError(f"Experiment '{exp_id}' not found.")
        meta.update(updates)
        meta["updated_at"] = _utc_now()
        _write_json(_meta_path(exp_id), meta)
        return meta


def _append_log(exp_id: str, message: str) -> None:
    with _LOCK:
        meta = _read_json(_meta_path(exp_id))
        if meta is None:
            return
        logs = meta.get("logs", [])
        timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        logs.append(f"[{timestamp}] {message}")
        meta["logs"] = logs[-100:]
        meta["updated_at"] = _utc_now()
        _write_json(_meta_path(exp_id), meta)


def _get_db_counts(db: DatabaseManager) -> Dict[str, int]:
    with db.get_session() as session:
        rows = (
            session.query(SampleRecord.split, SampleRecord.status)
            .filter(SampleRecord.experiment_id == db.experiment_id)
            .all()
        )
        return {
            "n_total": len(rows),
            "n_pool": sum(1 for sp, st in rows if sp == "pool" and st == "unlabeled"),
            "n_queued": sum(1 for sp, st in rows if sp == "pool" and st == "queued"),
            "n_labeled": sum(1 for sp, st in rows if sp == "pool" and st == "labeled"),
        }


def get_experiment_db(exp_id: str) -> DatabaseManager:
    with _LOCK:
        if exp_id in _RUNTIMES:
            return _RUNTIMES[exp_id].db
    env_url = os.environ.get("DATABASE_URL", "").strip()
    if env_url:
        return DatabaseManager(env_url, experiment_id=exp_id)
    db_file = _db_path(exp_id)
    if not os.path.exists(db_file):
        raise KeyError(f"Database for experiment '{exp_id}' not found.")
    return DatabaseManager(f"sqlite:///{db_file}", experiment_id=exp_id)


def get_experiment_runtime(exp_id: str) -> Optional[_RuntimeContext]:
    with _LOCK:
        return _RUNTIMES.get(exp_id)


def _build_runtime(exp_id: str, meta: Dict, cancel_event: Optional[threading.Event] = None) -> _RuntimeContext:
    cfg_dict = meta["config"]
    seed_everything(int(cfg_dict["seed"]))

    dataset = get_cached_dataset(
        dataset_id=cfg_dict["dataset"],
        max_samples=int(cfg_dict["max_samples"]),
        use_synthetic=bool(cfg_dict["use_synthetic"]),
        seed=int(cfg_dict["seed"]),
    )
    db_url = _configured_db_url(exp_id)
    db = DatabaseManager(db_url, experiment_id=exp_id)
    db.save_experiment_meta(
        name=meta["name"],
        dataset=meta["dataset"],
        model_type=meta["model_type"],
        strategy=meta["strategy"],
        mode=meta["mode"],
        status=meta.get("status", "created"),
        config=cfg_dict,
    )
    model = build_classifier(
        dataset=dataset,
        model_type=cfg_dict["model_type"],
        max_epochs=int(cfg_dict["max_epochs"]),
        lr=float(cfg_dict["lr"]),
        warm_start=bool(cfg_dict["warm_start"]),
        device=DEFAULT_DEVICE,
        batch_size=16,
    )
    strategy = get_strategy(cfg_dict["strategy"], seed=int(cfg_dict["seed"]))
    oracle = SimulatedOracle(
        dataset=dataset,
        noise_rate=float(cfg_dict["label_noise_rate"]),
        seed=int(cfg_dict["seed"]),
    )
    loop_cfg = LoopConfig(
        dataset=cfg_dict["dataset"],
        model_type=cfg_dict["model_type"],
        strategy=cfg_dict["strategy"],
        initial_size=int(cfg_dict["initial_size"]),
        batch_size=int(cfg_dict["batch_size"]),
        max_rounds=int(cfg_dict["max_rounds"]),
        target_accuracy=cfg_dict.get("target_accuracy"),
        label_budget=cfg_dict.get("label_budget"),
        seed=int(cfg_dict["seed"]),
        warm_start=bool(cfg_dict["warm_start"]),
        max_epochs=int(cfg_dict["max_epochs"]),
        lr=float(cfg_dict["lr"]),
        label_noise_rate=float(cfg_dict["label_noise_rate"]),
        use_synthetic=bool(cfg_dict["use_synthetic"]),
        max_samples=int(cfg_dict["max_samples"]),
        db_url=db_url,
    )
    orchestrator = ActiveLearningOrchestrator(
        config=loop_cfg,
        dataset=dataset,
        model=model,
        strategy=strategy,
        labeler=oracle,
        db=db,
        checkpoint_dir=_checkpoints_dir(exp_id),
        cancel_event=cancel_event or _get_cancel_event(exp_id),
    )
    ctx = _RuntimeContext(orchestrator=orchestrator, db=db, dataset=dataset)
    with _LOCK:
        _RUNTIMES[exp_id] = ctx
    return ctx


def _capture_round_artifacts(
    exp_id: str,
    round_num: int,
    strategy_name: str,
    ctx: _RuntimeContext,
    selected_ids: List[str],
    include_ground_truth: bool,
) -> None:
    """Compute and persist PCA projection and detailed selected-sample diagnostics."""
    labeled_ids, _, _ = ctx.orchestrator.pool_manager.get_labeled_data()
    points, explained = compute_pca_projection(
        dataset=ctx.dataset,
        model=ctx.orchestrator.model,
        labeled_ids=labeled_ids,
        newly_queried_ids=selected_ids,
        max_points=200,
    )
    _write_json(
        _pca_path(exp_id),
        {
            "experiment_id": exp_id,
            "round_num": round_num,
            "explained_variance_ratio": explained,
            "points": points,
        },
    )

    samples_payload: List[Dict] = []
    if selected_ids and getattr(ctx.orchestrator.model, "initialized_", False):
        feats = ctx.dataset.get_features(selected_ids)
        probs = ctx.orchestrator.model.predict_calibrated_proba(feats)
        preds = np.argmax(probs, axis=1)
        gt = ctx.dataset.get_ground_truth(selected_ids) if include_ground_truth else None

        for i, sid in enumerate(selected_ids):
            p_vec = probs[i]
            clipped = np.clip(p_vec, 1e-12, 1.0)
            entropy = float(-np.sum(clipped * np.log2(clipped)))
            conf = float(np.max(p_vec))
            item: Dict[str, Any] = {
                "sample_id": sid,
                "predicted_class_id": int(preds[i]),
                "predicted_class_name": ctx.dataset.class_names[int(preds[i])],
                "confidence": round(conf, 4),
                "uncertainty": round(1.0 - conf, 4),
                "entropy": round(entropy, 4),
                "probabilities": [round(float(v), 5) for v in p_vec],
                "class_names": ctx.dataset.class_names,
                "image_base64": feature_to_base64_png(feats[i]) if ctx.dataset.name == "cifar10" else None,
                "feature_summary": summarize_text_embedding(feats[i]) if ctx.dataset.name == "agnews" else None,
                "true_class_id": int(gt[i]) if gt is not None else None,
                "true_class_name": ctx.dataset.class_names[int(gt[i])] if gt is not None else None,
            }
            samples_payload.append(item)

    _write_json(
        _selected_path(exp_id),
        {
            "experiment_id": exp_id,
            "round_num": round_num,
            "strategy": strategy_name,
            "samples": samples_payload,
        },
    )


def _sync_summary_metrics(exp_id: str, ctx: _RuntimeContext, current_round: int, stage: str, stage_idx: int) -> None:
    counts = _get_db_counts(ctx.db)
    rounds = ctx.db.get_all_rounds_history()
    latest = rounds[-1] if rounds else None
    _update_meta(
        exp_id,
        current_round=current_round,
        current_stage=stage,
        stage_index=stage_idx,
        n_labeled=counts["n_labeled"],
        n_pool=counts["n_pool"],
        n_queued=counts["n_queued"],
        n_total=counts["n_total"],
        latest_accuracy=latest["test_acc"] if latest else None,
        latest_f1=latest["test_f1"] if latest else None,
        latest_ece=latest["ece"] if latest else None,
    )


def _run_simulated_oracle_worker(exp_id: str, job_id: Optional[str] = None) -> None:
    ctx: Optional[_RuntimeContext] = None
    jid = job_id or f"job_{uuid.uuid4().hex[:8]}"
    try:
        meta = _read_json(_meta_path(exp_id))
        if meta is None:
            return
        cancel_ev = _get_cancel_event(exp_id)
        ctx = _build_runtime(exp_id, meta, cancel_event=cancel_ev)
        orch = ctx.orchestrator

        ctx.db.record_job(
            job_id=jid,
            job_type="simulated_oracle",
            status="running",
            progress_pct=0.0,
            current_stage=PIPELINE_STAGES[0],
        )
        _update_meta(exp_id, status="running", error_message=None, current_job_id=jid)

        # Resume existing checkpoints/rounds cleanly if present
        orch._restore_from_persistence()
        resumed_round = ctx.db.get_latest_round()
        if resumed_round >= 0:
            _append_log(
                exp_id,
                f"Resuming {meta['dataset']} ({meta['strategy']}) from round {resumed_round} checkpoint.",
            )
        else:
            _append_log(exp_id, f"Initialized {meta['dataset']} with strategy={meta['strategy']} (Simulated Oracle mode).")

        ml_tracker = MLflowTracker()
        with ml_tracker.start_run(
            run_name=f"{meta['dataset']}_{meta['strategy']}_{exp_id}",
            tags={
                "activepulse_experiment_id": exp_id,
                "mode": meta["mode"],
                "dataset": meta["dataset"],
                "strategy": meta["strategy"],
            },
        ):
            ml_tracker.log_params(meta["config"])

            # Stage 1 & Round 0 Cold Start
            if resumed_round < 0:
                _sync_summary_metrics(exp_id, ctx, 0, PIPELINE_STAGES[0], 0)
                _append_log(exp_id, f"Round 0 cold-start: sampling {orch.config.initial_size} initial instances...")
                _sync_summary_metrics(exp_id, ctx, 0, PIPELINE_STAGES[1], 1)
                orch.cold_start()
                _sync_summary_metrics(exp_id, ctx, 0, PIPELINE_STAGES[4], 4)
                rounds_hist = ctx.db.get_all_rounds_history()
                if rounds_hist:
                    r0_metrics = rounds_hist[-1]
                    ml_tracker.log_round_metrics(
                        round_num=0,
                        accuracy=r0_metrics["test_acc"],
                        macro_f1=r0_metrics["test_f1"],
                        ece=r0_metrics["ece"],
                        n_labeled=r0_metrics["n_labeled"],
                        n_pool=r0_metrics["n_pool"],
                        nll=r0_metrics.get("nll"),
                        train_time_sec=r0_metrics.get("train_time_sec"),
                    )
                    ctx.db.save_model_version(
                        round_num=0,
                        model_type=meta["model_type"],
                        checkpoint_path=os.path.join(_checkpoints_dir(exp_id), "round_0.pt"),
                        temperature=float(orch.model.current_temperature),
                        test_acc=r0_metrics["test_acc"],
                        test_f1=r0_metrics["test_f1"],
                        test_ece=r0_metrics["ece"],
                        test_nll=r0_metrics.get("nll"),
                        train_time_sec=r0_metrics.get("train_time_sec"),
                    )
                    _append_log(
                        exp_id,
                        f"Round 0 complete: Acc={r0_metrics['test_acc']*100:.1f}%, "
                        f"F1={r0_metrics['test_f1']*100:.1f}%, ECE={r0_metrics['ece']:.4f}",
                    )

            current_round = ctx.db.get_latest_round()
            if current_round < 0:
                current_round = 0

            last_selected: List[str] = []
            stop_reason: Optional[str] = None

            while current_round < orch.config.max_rounds:
                if cancel_ev.is_set():
                    stop_reason = "Cancelled by user request."
                    ctx.db.record_job(
                        job_id=jid,
                        job_type="simulated_oracle",
                        status="cancelled",
                        progress_pct=round((current_round / max(1, orch.config.max_rounds)) * 100.0, 2),
                        current_stage="Cancelled",
                    )
                    _update_meta(exp_id, status="cancelled", current_stage="Cancelled", stop_reason=stop_reason)
                    _append_log(exp_id, stop_reason)
                    return

                next_round = current_round + 1
                stats = orch.pool_manager.get_stats()
                rounds_hist = ctx.db.get_all_rounds_history()
                curr_acc = rounds_hist[-1]["test_acc"] if rounds_hist else 0.0

                should_stop, reason = orch.stopping_rule.should_stop(
                    current_round=next_round,
                    n_labeled=stats["n_labeled"],
                    n_pool=stats["n_pool"],
                    test_acc=curr_acc,
                )
                if should_stop:
                    stop_reason = reason
                    _append_log(exp_id, f"Stopping rule triggered: {reason}")
                    break

                pool_ids, X_pool = orch.pool_manager.get_pool_data()
                if len(pool_ids) == 0:
                    stop_reason = "Unlabeled pool exhausted."
                    _append_log(exp_id, stop_reason)
                    break

                current_round = next_round
                _sync_summary_metrics(exp_id, ctx, current_round, PIPELINE_STAGES[3], 3)
                k = min(orch.config.batch_size, len(pool_ids))
                _append_log(exp_id, f"Round {current_round}: scoring {len(pool_ids)} pool samples and querying {k}...")
                selected_ids = orch.strategy.select(pool_ids, X_pool, orch.model, k=k)
                last_selected = selected_ids

                _capture_round_artifacts(
                    exp_id=exp_id,
                    round_num=current_round,
                    strategy_name=orch.strategy.name,
                    ctx=ctx,
                    selected_ids=selected_ids,
                    include_ground_truth=True,
                )

                orch.pool_manager.mark_queued(selected_ids)
                sids, labels = orch.labeler.label(selected_ids)
                orch.pool_manager.label_samples(sids, labels, round_num=current_round, annotator="oracle")

                _sync_summary_metrics(exp_id, ctx, current_round, PIPELINE_STAGES[1], 1)
                metrics = orch._train_and_evaluate(round_num=current_round)
                _sync_summary_metrics(exp_id, ctx, current_round, PIPELINE_STAGES[4], 4)

                pct = min(99.0, round((current_round / max(1, orch.config.max_rounds)) * 100.0, 2))
                ctx.db.record_job(
                    job_id=jid,
                    job_type="simulated_oracle",
                    status="running",
                    progress_pct=pct,
                    current_stage=f"Round {current_round}/{orch.config.max_rounds}",
                )
                ctx.db.save_model_version(
                    round_num=current_round,
                    model_type=meta["model_type"],
                    checkpoint_path=os.path.join(_checkpoints_dir(exp_id), f"round_{current_round}.pt"),
                    temperature=float(orch.model.current_temperature),
                    test_acc=metrics["accuracy"],
                    test_f1=metrics["macro_f1"],
                    test_ece=metrics["ece"],
                    test_nll=metrics.get("nll"),
                    train_time_sec=metrics.get("train_time"),
                )
                rounds_so_far = ctx.db.get_all_rounds_history()
                latest_r = rounds_so_far[-1] if rounds_so_far else {}
                post_stats = orch.pool_manager.get_stats()
                alc_so_far = (
                    compute_alc(
                        [r["n_labeled"] for r in rounds_so_far],
                        [r["test_acc"] for r in rounds_so_far],
                    )
                    if rounds_so_far
                    else None
                )
                ml_tracker.log_round_metrics(
                    round_num=current_round,
                    accuracy=metrics["accuracy"],
                    macro_f1=metrics["macro_f1"],
                    ece=metrics["ece"],
                    n_labeled=int(latest_r.get("n_labeled", post_stats["n_labeled"])),
                    n_pool=int(latest_r.get("n_pool", post_stats["n_pool"])),
                    nll=metrics.get("nll"),
                    alc=alc_so_far,
                    train_time_sec=latest_r.get("train_time_sec"),
                )
                _append_log(
                    exp_id,
                    f"Round {current_round} complete: Acc={metrics['accuracy']*100:.1f}%, "
                    f"F1={metrics['macro_f1']*100:.1f}%, ECE={metrics['ece']:.4f}",
                )

            _capture_round_artifacts(
                exp_id=exp_id,
                round_num=current_round,
                strategy_name=orch.strategy.name,
                ctx=ctx,
                selected_ids=last_selected,
                include_ground_truth=True,
            )
            rounds_all = ctx.db.get_all_rounds_history()
            if rounds_all:
                alc_val = compute_alc([r["n_labeled"] for r in rounds_all], [r["test_acc"] for r in rounds_all])
                ml_tracker.log_summary_metrics(
                    {
                        "final_accuracy": rounds_all[-1]["test_acc"],
                        "final_f1": rounds_all[-1]["test_f1"],
                        "final_ece": rounds_all[-1]["ece"],
                        "alc": alc_val,
                    }
                )
            latest_ckpt = os.path.join(_checkpoints_dir(exp_id), "latest.pt")
            if os.path.exists(latest_ckpt):
                ml_tracker.log_artifact(latest_ckpt, artifact_path="checkpoints")

        ctx.db.record_job(
            job_id=jid,
            job_type="simulated_oracle",
            status="completed",
            progress_pct=100.0,
            current_stage="Completed",
        )
        _update_meta(
            exp_id,
            status="completed",
            current_stage="Completed",
            stage_index=4,
            stop_reason=stop_reason or f"Completed {current_round} active learning rounds.",
        )
        _append_log(exp_id, "Simulation completed successfully.")
    except Exception as exc:
        if ctx is not None:
            ctx.db.record_job(
                job_id=jid,
                job_type="simulated_oracle",
                status="failed",
                progress_pct=0.0,
                current_stage="Failed",
                error_message=str(exc),
            )
        _update_meta(
            exp_id,
            status="failed",
            current_stage="Failed",
            error_message=str(exc),
        )
        _append_log(exp_id, f"Experiment failed: {exc}")


def _run_hitl_bootstrap_worker(exp_id: str) -> None:
    """Initialize Round 0 base model and queue Round 1 samples for human annotation."""
    try:
        meta = _read_json(_meta_path(exp_id))
        if meta is None:
            return
        ctx = _build_runtime(exp_id, meta)
        orch = ctx.orchestrator

        _update_meta(exp_id, status="running", error_message=None)
        _append_log(exp_id, f"Starting Human-in-the-Loop workflow on {meta['dataset']} ({meta['strategy']}).")

        _sync_summary_metrics(exp_id, ctx, 0, PIPELINE_STAGES[0], 0)
        _append_log(exp_id, f"Round 0 cold-start: bootstrapping initial {orch.config.initial_size} samples...")
        _sync_summary_metrics(exp_id, ctx, 0, PIPELINE_STAGES[1], 1)
        orch.cold_start()

        pool_ids, X_pool = orch.pool_manager.get_pool_data()
        if not pool_ids:
            _update_meta(exp_id, status="completed", current_stage="Completed", stop_reason="Pool exhausted.")
            return

        next_round = 1
        k = min(orch.config.batch_size, len(pool_ids))
        _sync_summary_metrics(exp_id, ctx, next_round, PIPELINE_STAGES[3], 3)
        _append_log(exp_id, f"Round {next_round}: querying {k} samples and placing into Annotation Studio queue...")
        selected_ids = orch.strategy.select(pool_ids, X_pool, orch.model, k=k)
        orch.pool_manager.mark_queued(selected_ids)

        _capture_round_artifacts(
            exp_id=exp_id,
            round_num=next_round,
            strategy_name=orch.strategy.name,
            ctx=ctx,
            selected_ids=selected_ids,
            include_ground_truth=False,  # Never store ground truth during HITL annotation
        )
        _sync_summary_metrics(exp_id, ctx, next_round, "Awaiting Human Annotations", 3)
        _update_meta(exp_id, status="awaiting_annotations")
        _append_log(exp_id, f"{len(selected_ids)} samples queued for human labeling in Annotation Studio.")
    except Exception as exc:
        _update_meta(exp_id, status="failed", current_stage="Failed", error_message=str(exc))
        _append_log(exp_id, f"HITL bootstrap failed: {exc}")


def _run_hitl_advance_worker(exp_id: str) -> None:
    """Retrain after human annotations and either complete or queue the next round."""
    try:
        meta = _read_json(_meta_path(exp_id))
        if meta is None:
            return
        ctx = get_experiment_runtime(exp_id) or _build_runtime(exp_id, meta)
        orch = ctx.orchestrator
        current_round = int(meta.get("current_round", 1))

        _update_meta(exp_id, status="running", error_message=None)
        _sync_summary_metrics(exp_id, ctx, current_round, PIPELINE_STAGES[1], 1)
        _append_log(exp_id, f"Round {current_round}: retraining model with newly submitted human labels...")
        metrics = orch._train_and_evaluate(round_num=current_round)
        _sync_summary_metrics(exp_id, ctx, current_round, PIPELINE_STAGES[4], 4)
        _append_log(
            exp_id,
            f"Round {current_round} evaluated: Acc={metrics['accuracy']*100:.1f}%, "
            f"F1={metrics['macro_f1']*100:.1f}%, ECE={metrics['ece']:.4f}",
        )

        if current_round >= orch.config.max_rounds:
            _capture_round_artifacts(exp_id, current_round, orch.strategy.name, ctx, [], include_ground_truth=False)
            _update_meta(
                exp_id,
                status="completed",
                current_stage="Completed",
                stage_index=4,
                stop_reason=f"Maximum rounds ({orch.config.max_rounds}) reached.",
            )
            _append_log(exp_id, "HITL active learning workflow completed.")
            return

        stats = orch.pool_manager.get_stats()
        should_stop, reason = orch.stopping_rule.should_stop(
            current_round=current_round + 1,
            n_labeled=stats["n_labeled"],
            n_pool=stats["n_pool"],
            test_acc=metrics["accuracy"],
        )
        if should_stop:
            _capture_round_artifacts(exp_id, current_round, orch.strategy.name, ctx, [], include_ground_truth=False)
            _update_meta(
                exp_id,
                status="completed",
                current_stage="Completed",
                stage_index=4,
                stop_reason=reason,
            )
            _append_log(exp_id, f"Stopping rule triggered: {reason}")
            return

        pool_ids, X_pool = orch.pool_manager.get_pool_data()
        if not pool_ids:
            _update_meta(exp_id, status="completed", current_stage="Completed", stop_reason="Unlabeled pool exhausted.")
            return

        next_round = current_round + 1
        k = min(orch.config.batch_size, len(pool_ids))
        _sync_summary_metrics(exp_id, ctx, next_round, PIPELINE_STAGES[3], 3)
        selected_ids = orch.strategy.select(pool_ids, X_pool, orch.model, k=k)
        orch.pool_manager.mark_queued(selected_ids)

        _capture_round_artifacts(
            exp_id=exp_id,
            round_num=next_round,
            strategy_name=orch.strategy.name,
            ctx=ctx,
            selected_ids=selected_ids,
            include_ground_truth=False,
        )
        _sync_summary_metrics(exp_id, ctx, next_round, "Awaiting Human Annotations", 3)
        _update_meta(exp_id, status="awaiting_annotations")
        _append_log(exp_id, f"Round {next_round}: {len(selected_ids)} new samples queued for annotation.")
    except Exception as exc:
        _update_meta(exp_id, status="failed", current_stage="Failed", error_message=str(exc))
        _append_log(exp_id, f"Round advance failed: {exc}")


def _validate_request(req: ExperimentCreateRequest) -> None:
    registered = list_registered_strategies()
    if req.strategy.lower() not in registered:
        raise ValueError(f"Unknown strategy '{req.strategy}'. Registered: {registered}")
    if req.dataset == "cifar10" and req.model_type not in ("cifar_cnn", "resnet18"):
        raise ValueError("Dataset 'cifar10' supports model_type 'cifar_cnn' or 'resnet18'.")
    if req.dataset == "agnews" and req.model_type != "text_mlp":
        raise ValueError("Dataset 'agnews' requires model_type 'text_mlp'.")


def create_experiment(req: ExperimentCreateRequest) -> ExperimentSummary:
    _validate_request(req)
    exp_id = f"exp_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    exp_name = (req.name or "").strip() or f"{req.dataset.upper()} · {req.strategy} ({req.mode})"
    os.makedirs(_exp_dir(exp_id), exist_ok=True)

    now = _utc_now()
    meta = {
        "id": exp_id,
        "name": exp_name,
        "dataset": req.dataset,
        "model_type": req.model_type,
        "strategy": req.strategy.lower(),
        "mode": req.mode,
        "status": "created",
        "current_round": 0,
        "max_rounds": req.max_rounds,
        "current_stage": "Initialized",
        "stage_index": 0,
        "latest_accuracy": None,
        "latest_f1": None,
        "latest_ece": None,
        "n_labeled": 0,
        "n_pool": 0,
        "n_queued": 0,
        "n_total": req.max_samples,
        "created_at": now,
        "updated_at": now,
        "error_message": None,
        "stop_reason": None,
        "logs": [f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] Experiment created."],
        "config": req.model_dump(),
    }
    _write_json(_meta_path(exp_id), meta)

    if req.auto_start:
        return start_experiment(exp_id)
    return _meta_to_summary(meta)


def start_experiment(exp_id: str) -> ExperimentSummary:
    meta = _read_json(_meta_path(exp_id))
    if meta is None:
        raise KeyError(f"Experiment '{exp_id}' not found.")
    if meta["status"] == "running":
        return _meta_to_summary(meta)

    cancel_ev = _get_cancel_event(exp_id, reset=True)
    cancel_ev.clear()
    job_id = f"job_{uuid.uuid4().hex[:8]}"
    db = DatabaseManager(_configured_db_url(exp_id), experiment_id=exp_id)
    db.record_job(
        job_id=job_id,
        job_type=meta["mode"],
        status="queued",
        progress_pct=0.0,
        current_stage="Queued",
    )

    if meta["mode"] == "simulated_oracle":
        _update_meta(exp_id, status="running", current_stage=PIPELINE_STAGES[0], stage_index=0, current_job_id=job_id)
        t = threading.Thread(target=_run_simulated_oracle_worker, args=(exp_id, job_id), daemon=True)
    else:
        _update_meta(exp_id, status="running", current_stage=PIPELINE_STAGES[0], stage_index=0, current_job_id=job_id)
        t = threading.Thread(target=_run_hitl_bootstrap_worker, args=(exp_id,), daemon=True)
    t.start()
    return _meta_to_summary(_read_json(_meta_path(exp_id)))


def stop_experiment(exp_id: str) -> ExperimentSummary:
    """Safely request cancellation of a running experiment or mark queued/awaiting runs paused."""
    meta = _read_json(_meta_path(exp_id))
    if meta is None:
        raise KeyError(f"Experiment '{exp_id}' not found.")
    cancel_ev = _get_cancel_event(exp_id)
    cancel_ev.set()

    current_status = meta["status"]
    if current_status in ("completed", "failed", "cancelled"):
        return _meta_to_summary(meta)

    if os.path.exists(_db_path(exp_id)) or os.environ.get("DATABASE_URL", "").strip():
        db = get_experiment_db(exp_id)
        if meta.get("current_job_id"):
            db.record_job(
                job_id=meta["current_job_id"],
                job_type=meta["mode"],
                status="cancelled",
                progress_pct=round((meta.get("current_round", 0) / max(1, meta.get("max_rounds", 1))) * 100.0, 2),
                current_stage="Cancellation requested",
            )

    new_status = "cancelled" if current_status == "running" else "paused"
    updated = _update_meta(
        exp_id,
        status=new_status,
        current_stage="Cancelled" if new_status == "cancelled" else "Paused",
        stop_reason="Stopped by user request.",
    )
    _append_log(exp_id, "Stop/cancellation requested by user.")
    return _meta_to_summary(updated)


def resume_experiment(exp_id: str) -> ExperimentSummary:
    """Resume a cancelled, paused, or failed experiment from its latest checkpoint."""
    meta = _read_json(_meta_path(exp_id))
    if meta is None:
        raise KeyError(f"Experiment '{exp_id}' not found.")
    if meta["status"] == "running":
        return _meta_to_summary(meta)
    if meta["status"] == "completed":
        return _meta_to_summary(meta)

    cancel_ev = _get_cancel_event(exp_id, reset=True)
    cancel_ev.clear()
    job_id = f"job_{uuid.uuid4().hex[:8]}"
    db = DatabaseManager(_configured_db_url(exp_id), experiment_id=exp_id)
    db.record_job(
        job_id=job_id,
        job_type=f"resume_{meta['mode']}",
        status="queued",
        progress_pct=round((meta.get("current_round", 0) / max(1, meta.get("max_rounds", 1))) * 100.0, 2),
        current_stage="Resuming from checkpoint",
    )

    _update_meta(exp_id, status="running", current_stage="Resuming", error_message=None, current_job_id=job_id)
    if meta["mode"] == "simulated_oracle":
        t = threading.Thread(target=_run_simulated_oracle_worker, args=(exp_id, job_id), daemon=True)
    else:
        t = threading.Thread(target=_run_hitl_advance_worker, args=(exp_id,), daemon=True)
    t.start()
    return _meta_to_summary(_read_json(_meta_path(exp_id)))


def advance_experiment_round(exp_id: str) -> ExperimentSummary:
    meta = _read_json(_meta_path(exp_id))
    if meta is None:
        raise KeyError(f"Experiment '{exp_id}' not found.")
    if meta["mode"] != "human_in_the_loop":
        raise ValueError("Round advancement is only applicable to 'human_in_the_loop' experiments.")
    if meta["status"] == "running":
        raise ValueError("Experiment is currently training.")
    if meta["status"] == "completed":
        raise ValueError("Experiment is already completed.")

    db = get_experiment_db(exp_id)
    counts = _get_db_counts(db)
    if counts["n_queued"] > 0:
        raise ValueError(
            f"{counts['n_queued']} sample(s) are still pending in Annotation Studio. Label all queued samples before retraining."
        )

    _update_meta(exp_id, status="running", current_stage=PIPELINE_STAGES[1], stage_index=1)
    t = threading.Thread(target=_run_hitl_advance_worker, args=(exp_id,), daemon=True)
    t.start()
    return _meta_to_summary(_read_json(_meta_path(exp_id)))


def _meta_to_summary(meta: Dict) -> ExperimentSummary:
    return ExperimentSummary(
        id=meta["id"],
        name=meta["name"],
        dataset=meta["dataset"],
        model_type=meta["model_type"],
        strategy=meta["strategy"],
        mode=meta["mode"],
        status=meta["status"],
        current_round=meta.get("current_round", 0),
        max_rounds=meta.get("max_rounds", 3),
        current_stage=meta.get("current_stage", "Initialized"),
        latest_accuracy=meta.get("latest_accuracy"),
        latest_f1=meta.get("latest_f1"),
        latest_ece=meta.get("latest_ece"),
        n_labeled=meta.get("n_labeled", 0),
        n_pool=meta.get("n_pool", 0),
        n_queued=meta.get("n_queued", 0),
        created_at=meta["created_at"],
        updated_at=meta["updated_at"],
        error_message=meta.get("error_message"),
    )


def list_experiments() -> List[ExperimentSummary]:
    root = _experiments_root()
    summaries: List[ExperimentSummary] = []
    for entry in os.listdir(root):
        meta = _read_json(os.path.join(root, entry, "meta.json"))
        if meta:
            summaries.append(_meta_to_summary(meta))
    summaries.sort(key=lambda s: s.created_at, reverse=True)
    return summaries


def get_experiment_detail(exp_id: str) -> ExperimentDetailResponse:
    meta = _read_json(_meta_path(exp_id))
    if meta is None:
        raise KeyError(f"Experiment '{exp_id}' not found.")

    rounds: List[RoundMetricSchema] = []
    if os.path.exists(_db_path(exp_id)):
        db = get_experiment_db(exp_id)
        for row in db.get_all_rounds_history():
            rounds.append(RoundMetricSchema(**row))

    total_samples = int(meta.get("n_total", meta["config"].get("max_samples", 0)))
    n_labeled = int(meta.get("n_labeled", 0))
    savings = None
    if rounds:
        savings = CostSavingsSchema(**estimate_cost_savings(total_samples, n_labeled))

    return ExperimentDetailResponse(
        summary=_meta_to_summary(meta),
        config=meta["config"],
        rounds=rounds,
        cost_savings=savings,
        stop_reason=meta.get("stop_reason"),
        logs=meta.get("logs", []),
    )


def get_experiment_progress(exp_id: str) -> ExperimentProgressResponse:
    meta = _read_json(_meta_path(exp_id))
    if meta is None:
        raise KeyError(f"Experiment '{exp_id}' not found.")
    return ExperimentProgressResponse(
        experiment_id=exp_id,
        status=meta["status"],
        current_round=meta.get("current_round", 0),
        max_rounds=meta.get("max_rounds", 3),
        current_stage=meta.get("current_stage", "Initialized"),
        stage_index=meta.get("stage_index", 0),
        stages=PIPELINE_STAGES,
        n_labeled=meta.get("n_labeled", 0),
        n_pool=meta.get("n_pool", 0),
        n_queued=meta.get("n_queued", 0),
        latest_accuracy=meta.get("latest_accuracy"),
        latest_f1=meta.get("latest_f1"),
        latest_ece=meta.get("latest_ece"),
        error_message=meta.get("error_message"),
        logs=meta.get("logs", []),
    )


def get_experiment_pca(exp_id: str) -> PcaEmbeddingResponse:
    meta = _read_json(_meta_path(exp_id))
    if meta is None:
        raise KeyError(f"Experiment '{exp_id}' not found.")
    pca_data = _read_json(_pca_path(exp_id))
    if pca_data is None:
        return PcaEmbeddingResponse(
            experiment_id=exp_id,
            round_num=meta.get("current_round", 0),
            explained_variance_ratio=[0.0, 0.0],
            points=[],
        )
    return PcaEmbeddingResponse(
        experiment_id=exp_id,
        round_num=pca_data["round_num"],
        explained_variance_ratio=pca_data["explained_variance_ratio"],
        points=[PcaPointSchema(**p) for p in pca_data["points"]],
    )


def get_experiment_selected_samples(exp_id: str) -> SelectedSamplesResponse:
    meta = _read_json(_meta_path(exp_id))
    if meta is None:
        raise KeyError(f"Experiment '{exp_id}' not found.")
    sel_data = _read_json(_selected_path(exp_id))
    if sel_data is None:
        return SelectedSamplesResponse(
            experiment_id=exp_id,
            round_num=meta.get("current_round", 0),
            strategy=meta["strategy"],
            samples=[],
        )
    return SelectedSamplesResponse(
        experiment_id=exp_id,
        round_num=sel_data["round_num"],
        strategy=sel_data["strategy"],
        samples=[SelectedSampleSchema(**s) for s in sel_data["samples"]],
    )


def delete_experiment(exp_id: str) -> None:
    meta = _read_json(_meta_path(exp_id))
    if meta is None:
        raise KeyError(f"Experiment '{exp_id}' not found.")
    with _LOCK:
        ctx = _RUNTIMES.pop(exp_id, None)
        _CANCEL_EVENTS.pop(exp_id, None)
        if ctx is not None:
            ctx.db.close()
    shutil.rmtree(_exp_dir(exp_id), ignore_errors=True)


def get_experiment_jobs(exp_id: str) -> ExperimentJobsResponse:
    meta = _read_json(_meta_path(exp_id))
    if meta is None:
        raise KeyError(f"Experiment '{exp_id}' not found.")
    if not os.path.exists(_db_path(exp_id)) and not os.environ.get("DATABASE_URL", "").strip():
        return ExperimentJobsResponse(experiment_id=exp_id, jobs=[])
    db = get_experiment_db(exp_id)
    rows = db.list_jobs()
    return ExperimentJobsResponse(
        experiment_id=exp_id,
        jobs=[JobStatusSchema(**r) for r in rows],
    )


def get_experiment_evaluation(exp_id: str) -> ExperimentEvaluationResponse:
    detail = get_experiment_detail(exp_id)
    summary = detail.summary
    cfg = detail.config
    rounds = detail.rounds

    if rounds:
        x_vals = [r.n_labeled for r in rounds]
        y_vals = [r.test_acc for r in rounds]
        alc = compute_alc(x_vals, y_vals)
        latest = rounds[-1]
        total_train = round(sum(float(r.train_time_sec or 0.0) for r in rounds), 4)
        target_acc = cfg.get("target_accuracy")
        ltt = labels_to_target_accuracy(x_vals, y_vals, float(target_acc)) if target_acc else None
    else:
        alc = 0.0
        latest = None
        total_train = 0.0
        target_acc = cfg.get("target_accuracy")
        ltt = None

    latest_model_ver = None
    if os.path.exists(_db_path(exp_id)) or os.environ.get("DATABASE_URL", "").strip():
        db = get_experiment_db(exp_id)
        mv = db.get_latest_model_version()
        if mv:
            latest_model_ver = f"{exp_id}_r{mv['round_num']}"

    return ExperimentEvaluationResponse(
        experiment_id=exp_id,
        dataset=summary.dataset,
        strategy=summary.strategy,
        mode=summary.mode,
        rounds_completed=len(rounds),
        latest_accuracy=latest.test_acc if latest else None,
        latest_f1=latest.test_f1 if latest else None,
        latest_ece=latest.ece if latest else None,
        latest_nll=latest.nll if latest else None,
        alc=round(float(alc), 6),
        total_train_time_sec=total_train,
        labels_to_target=ltt,
        target_accuracy=target_acc,
        rounds=rounds,
        cost_savings=detail.cost_savings,
        model_version=latest_model_ver,
    )


def compare_experiments(req: ExperimentCompareRequest) -> ExperimentCompareResponse:
    entries: List[ExperimentCompareEntry] = []
    for exp_id in req.experiment_ids:
        ev = get_experiment_evaluation(exp_id)
        detail = get_experiment_detail(exp_id)
        entries.append(
            ExperimentCompareEntry(
                experiment_id=exp_id,
                name=detail.summary.name,
                dataset=ev.dataset,
                model_type=detail.summary.model_type,
                strategy=ev.strategy,
                status=detail.summary.status,
                rounds_completed=ev.rounds_completed,
                n_labeled=detail.summary.n_labeled,
                final_accuracy=ev.latest_accuracy,
                final_f1=ev.latest_f1,
                final_ece=ev.latest_ece,
                final_nll=ev.latest_nll,
                alc=ev.alc,
            )
        )
    tracker = MLflowTracker()
    ml_runs = tracker.compare_runs(experiment_ids=req.experiment_ids)
    return ExperimentCompareResponse(experiments=entries, mlflow_runs=ml_runs)

