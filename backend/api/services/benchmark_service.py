"""Benchmark Arena service reusing src/eval/savings.py and ActiveLearningOrchestrator."""

from datetime import datetime, timezone
import json
import os
import tempfile
import threading
import uuid
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import torch

from backend.api.config import DATA_DIR, DEFAULT_DEVICE, RESULTS_DIR
from backend.api.schemas.benchmarks import (
    BenchmarkCurvePoint,
    BenchmarkJobStatus,
    BenchmarkResultsResponse,
    BenchmarkRunRequest,
    BenchmarkSavingsRow,
)
from backend.api.services.dataset_service import get_cached_dataset
from backend.api.services.utils import build_classifier
from src.data.db import DatabaseManager
from src.eval.savings import compute_label_savings, plot_learning_curves
from src.labeling.oracle import SimulatedOracle
from src.loop.orchestrator import ActiveLearningOrchestrator
from src.loop.state import LoopConfig
from src.strategies.registry import get_strategy, list_registered_strategies


_BENCHMARK_JSON = os.path.join(DATA_DIR, "latest_benchmark.json")
_JOBS: Dict[str, Dict] = {}
_JOB_LOCK = threading.RLock()


def _parse_val_ci(raw: str) -> tuple:
    text = str(raw).strip()
    if "+/-" in text:
        parts = text.split("+/-")
        try:
            return float(parts[0].strip()), float(parts[1].strip())
        except ValueError:
            return 0.0, 0.0
    try:
        return float(text), 0.0
    except ValueError:
        return 0.0, 0.0


def _load_csv_fallback() -> BenchmarkResultsResponse:
    csv_path = os.path.join(RESULTS_DIR, "benchmark_savings.csv")
    img_path = os.path.join(RESULTS_DIR, "learning_curves.png")
    if not os.path.exists(csv_path):
        return BenchmarkResultsResponse(
            source="none",
            has_learning_curve_image=os.path.exists(img_path),
            summary_rows=[],
            curve_points=[],
        )

    df = pd.read_csv(csv_path)
    rows: List[BenchmarkSavingsRow] = []
    for _, r in df.iterrows():
        alc_mean, alc_ci = _parse_val_ci(r.get("ALC (Mean +/- 95% CI)", "0 +/- 0"))
        lbl_raw = str(r.get("Labels to Target", "Not Reached"))
        lbl_mean = None if "Not Reached" in lbl_raw else _parse_val_ci(lbl_raw)[0]
        red_raw = str(r.get("% Label Reduction vs Random", "0.0%"))
        red_val = 0.0 if "Baseline" in red_raw else _parse_val_ci(red_raw.replace("%", ""))[0]

        rows.append(
            BenchmarkSavingsRow(
                strategy=str(r["Strategy"]),
                seeds=int(r["Seeds"]),
                alc_mean=alc_mean,
                alc_ci95=alc_ci,
                alc_display=str(r.get("ALC (Mean +/- 95% CI)", "")),
                labels_to_target=lbl_mean,
                labels_to_target_display=lbl_raw,
                label_reduction_pct=red_val,
                label_reduction_display=red_raw,
            )
        )

    return BenchmarkResultsResponse(
        source="results/benchmark_savings.csv",
        has_learning_curve_image=os.path.exists(img_path),
        summary_rows=rows,
        curve_points=[],
    )


def get_benchmark_results() -> BenchmarkResultsResponse:
    img_path = os.path.join(RESULTS_DIR, "learning_curves.png")
    if os.path.exists(_BENCHMARK_JSON):
        with open(_BENCHMARK_JSON, "r", encoding="utf-8") as f:
            payload = json.load(f)
        payload["has_learning_curve_image"] = os.path.exists(img_path)
        return BenchmarkResultsResponse(**payload)
    return _load_csv_fallback()


def _benchmark_worker(job_id: str, req: BenchmarkRunRequest) -> None:
    try:
        model_type = "cifar_cnn" if req.dataset == "cifar10" else "text_mlp"
        benchmark_raw: Dict[str, List[List[Dict]]] = {}
        total_steps = len(req.strategies) * len(req.seeds)
        completed = 0

        for strat_name in req.strategies:
            strategy_runs: List[List[Dict]] = []
            for seed in req.seeds:
                with _JOB_LOCK:
                    _JOBS[job_id]["current_strategy"] = strat_name
                    _JOBS[job_id]["current_seed"] = seed

                torch.manual_seed(seed)
                np.random.seed(seed)

                fd, db_path = tempfile.mkstemp(suffix=f"_bench_{strat_name}_{seed}.db")
                os.close(fd)
                db = DatabaseManager(f"sqlite:///{db_path}")
                try:
                    dataset = get_cached_dataset(
                        dataset_id=req.dataset,
                        max_samples=req.max_samples,
                        use_synthetic=req.use_synthetic,
                        seed=seed,
                    )
                    model = build_classifier(
                        dataset=dataset,
                        model_type=model_type,
                        max_epochs=req.epochs,
                        lr=0.005 if req.dataset == "cifar10" else 0.01,
                        warm_start=False,
                        device=DEFAULT_DEVICE,
                        batch_size=32,
                    )
                    strategy = get_strategy(strat_name, seed=seed)
                    oracle = SimulatedOracle(dataset=dataset, noise_rate=0.0, seed=seed)
                    config = LoopConfig(
                        dataset=req.dataset,
                        model_type=model_type,
                        strategy=strat_name,
                        initial_size=req.initial_size,
                        batch_size=req.batch_size,
                        max_rounds=req.rounds,
                        seed=seed,
                        db_url=f"sqlite:///{db_path}",
                    )
                    orchestrator = ActiveLearningOrchestrator(
                        config=config,
                        dataset=dataset,
                        model=model,
                        strategy=strategy,
                        labeler=oracle,
                        db=db,
                    )
                    tracker = orchestrator.run_simulation()
                    strategy_runs.append([m.__dict__ for m in tracker.history])
                finally:
                    db.close()
                    if os.path.exists(db_path):
                        try:
                            os.remove(db_path)
                        except OSError:
                            pass

                completed += 1
                with _JOB_LOCK:
                    _JOBS[job_id]["completed_steps"] = completed

            benchmark_raw[strat_name] = strategy_runs

        savings_df = compute_label_savings(
            benchmark_results=benchmark_raw,
            target_accuracy=req.target_accuracy,
            baseline_strategy="random",
        )

        os.makedirs(RESULTS_DIR, exist_ok=True)
        clean_table = savings_df[
            ["Strategy", "Seeds", "ALC (Mean +/- 95% CI)", "Labels to Target", "% Label Reduction vs Random"]
        ]
        clean_table.to_csv(os.path.join(RESULTS_DIR, "benchmark_savings.csv"), index=False)
        with open(os.path.join(RESULTS_DIR, "benchmark_savings.md"), "w", encoding="utf-8") as f:
            f.write("# Active Learning Benchmark Results\n\n")
            f.write(clean_table.to_markdown(index=False))
            f.write("\n")

        plot_learning_curves(
            benchmark_results=benchmark_raw,
            output_path=os.path.join(RESULTS_DIR, "learning_curves.png"),
            target_accuracy=req.target_accuracy,
            title=f"Active Learning Curves ({req.dataset.upper()} - {len(req.seeds)} Seeds)",
        )

        # Build structured response payload for interactive Recharts visualizations
        summary_rows: List[Dict] = []
        curve_points: List[Dict] = []

        for _, row in savings_df.iterrows():
            strat = str(row["Strategy"])
            runs = benchmark_raw.get(strat, [])
            final_accs = [r[-1]["test_acc"] for r in runs if r]
            final_f1s = [r[-1]["test_f1"] for r in runs if r]
            final_eces = [r[-1]["ece"] for r in runs if r and r[-1].get("ece") is not None]

            _, alc_ci = _parse_val_ci(row["ALC (Mean +/- 95% CI)"])
            lbl_mean = row["_raw_labels_mean"]
            summary_rows.append(
                {
                    "strategy": strat,
                    "seeds": int(row["Seeds"]),
                    "alc_mean": round(float(row["_raw_alc_mean"]), 4),
                    "alc_ci95": round(float(alc_ci), 4),
                    "alc_display": str(row["ALC (Mean +/- 95% CI)"]),
                    "labels_to_target": None if pd.isna(lbl_mean) else round(float(lbl_mean), 1),
                    "labels_to_target_display": str(row["Labels to Target"]),
                    "label_reduction_pct": round(float(row["_raw_reduction_mean"]), 2),
                    "label_reduction_display": str(row["% Label Reduction vs Random"]),
                    "final_accuracy": round(float(np.mean(final_accs)), 4) if final_accs else None,
                    "final_f1": round(float(np.mean(final_f1s)), 4) if final_f1s else None,
                    "final_ece": round(float(np.mean(final_eces)), 4) if final_eces else None,
                }
            )

            if runs:
                min_len = min(len(r) for r in runs)
                for step_idx in range(min_len):
                    n_lbl = int(runs[0][step_idx]["n_labeled"])
                    accs = [r[step_idx]["test_acc"] for r in runs]
                    f1s = [r[step_idx]["test_f1"] for r in runs]
                    eces = [r[step_idx]["ece"] for r in runs if r[step_idx].get("ece") is not None]
                    n_s = len(accs)
                    ci = 1.96 * (float(np.std(accs)) / np.sqrt(n_s)) if n_s > 1 else 0.0
                    curve_points.append(
                        {
                            "strategy": strat,
                            "n_labeled": n_lbl,
                            "mean_acc": round(float(np.mean(accs)), 4),
                            "ci95_acc": round(ci, 4),
                            "mean_f1": round(float(np.mean(f1s)), 4) if f1s else None,
                            "mean_ece": round(float(np.mean(eces)), 4) if eces else None,
                        }
                    )

        payload = {
            "source": "live_benchmark",
            "dataset": req.dataset,
            "target_accuracy": req.target_accuracy,
            "seeds": req.seeds,
            "has_learning_curve_image": True,
            "summary_rows": summary_rows,
            "curve_points": curve_points,
        }
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(_BENCHMARK_JSON, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        with _JOB_LOCK:
            _JOBS[job_id]["status"] = "completed"
            _JOBS[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
    except Exception as exc:
        with _JOB_LOCK:
            _JOBS[job_id]["status"] = "failed"
            _JOBS[job_id]["error_message"] = str(exc)
            _JOBS[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()


def start_benchmark_job(req: BenchmarkRunRequest) -> BenchmarkJobStatus:
    registered = set(list_registered_strategies())
    for s in req.strategies:
        if s.lower() not in registered:
            raise ValueError(f"Unknown strategy '{s}'.")
    if "random" not in [s.lower() for s in req.strategies]:
        req.strategies = ["random"] + req.strategies

    job_id = f"bench_{uuid.uuid4().hex[:8]}"
    now = datetime.now(timezone.utc).isoformat()
    status_obj = {
        "job_id": job_id,
        "status": "running",
        "current_strategy": req.strategies[0],
        "current_seed": req.seeds[0],
        "completed_steps": 0,
        "total_steps": len(req.strategies) * len(req.seeds),
        "error_message": None,
        "started_at": now,
        "completed_at": None,
    }
    with _JOB_LOCK:
        _JOBS[job_id] = status_obj

    t = threading.Thread(target=_benchmark_worker, args=(job_id, req), daemon=True)
    t.start()
    return BenchmarkJobStatus(**status_obj)


def get_benchmark_job_status(job_id: str) -> BenchmarkJobStatus:
    with _JOB_LOCK:
        if job_id not in _JOBS:
            raise KeyError(f"Benchmark job '{job_id}' not found.")
        return BenchmarkJobStatus(**_JOBS[job_id])

