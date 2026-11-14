"""Phase 5 Multi-Seed Active Learning Benchmark & Statistical Evaluation Runner."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data.agnews import AGNewsAdapter
from src.data.cifar10 import CIFAR10Adapter
from src.data.db import DatabaseManager
from src.eval.savings import compute_alc, compute_label_savings, labels_to_target_accuracy, plot_learning_curves
from src.eval.statistics import (
    compute_configurable_cost_savings,
    paired_strategy_comparison,
    plot_confusion_matrix,
    plot_label_efficiency_chart,
    plot_metric_bars_with_error,
    write_statistical_report,
)
from src.labeling.oracle import SimulatedOracle
from src.loop.orchestrator import ActiveLearningOrchestrator
from src.loop.state import LoopConfig
from src.models.cnn import CIFARConvNet
from src.models.text_net import TextMLP
from src.models.wrapper import ActiveLearningClassifier, seed_everything
from src.strategies.registry import get_strategy
from src.tracking.mlflow_tracker import MLflowTracker

ALL_STRATEGIES = [
    "random",
    "least_confidence",
    "margin",
    "entropy",
    "bald",
    "coreset",
    "badge",
    "hybrid",
]
DEFAULT_SEEDS = [42, 43, 44, 45, 46]


def _build_dataset_and_model(
    dataset_name: str,
    use_synthetic: bool,
    max_samples: int,
    seed: int,
    epochs: int,
    lr: float,
    device: str,
):
    if dataset_name == "cifar10":
        dataset = CIFAR10Adapter(
            max_samples=max_samples,
            use_synthetic=use_synthetic,
            seed=seed,
        )
        net = CIFARConvNet(num_classes=dataset.num_classes)
        model_type = "cifar_cnn"
    elif dataset_name == "agnews":
        dataset = AGNewsAdapter(
            max_samples=max_samples,
            use_synthetic=use_synthetic,
            seed=seed,
        )
        net = TextMLP(in_features=64, num_classes=dataset.num_classes)
        model_type = "text_mlp"
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    classifier = ActiveLearningClassifier(
        module=net,
        max_epochs=epochs,
        lr=lr,
        batch_size=32,
        warm_start=False,
        device=device,
        verbose=0,
    )
    return dataset, classifier, model_type


def _detect_data_source(dataset: Any) -> str:
    sample_ids = dataset.get_all_sample_ids()
    if sample_ids and "_syn_" in sample_ids[0]:
        return "synthetic"
    if getattr(dataset, "use_synthetic", False):
        return "synthetic"
    return "real"


def run_dataset_benchmark(
    dataset_name: str,
    strategies: List[str],
    seeds: List[int],
    use_synthetic: bool,
    max_samples: int,
    initial_size: int,
    batch_size: int,
    rounds: int,
    epochs: int,
    lr: float,
    target_accuracy: float,
    label_budget: Optional[int],
    device: str,
    output_dir: str,
    cost_per_hour: float,
    seconds_per_annotation: float,
    enable_mlflow: bool,
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    cm_dir = os.path.join(output_dir, "confusion_matrices")
    os.makedirs(cm_dir, exist_ok=True)

    tracker = MLflowTracker(enabled=enable_mlflow)

    benchmark_histories: Dict[str, List[List[Dict[str, Any]]]] = {s: [] for s in strategies}
    seed_final_records: Dict[str, List[Dict[str, Any]]] = {s: [] for s in strategies}
    data_source_type: Optional[str] = None
    effective_budget = label_budget if label_budget is not None else (initial_size + rounds * batch_size)

    with tempfile.TemporaryDirectory() as tmp_dir:
        for strat_name in strategies:
            print(f"\n=== [{dataset_name.upper()}] Strategy: {strat_name} ({len(seeds)} seeds) ===")
            for seed in seeds:
                seed_everything(seed)
                dataset, model, model_type = _build_dataset_and_model(
                    dataset_name=dataset_name,
                    use_synthetic=use_synthetic,
                    max_samples=max_samples,
                    seed=seed,
                    epochs=epochs,
                    lr=lr,
                    device=device,
                )
                if data_source_type is None:
                    data_source_type = _detect_data_source(dataset)

                db_path = os.path.join(tmp_dir, f"{dataset_name}_{strat_name}_seed_{seed}.db")
                exp_id = f"bench_{dataset_name}_{strat_name}_s{seed}"
                db = DatabaseManager(f"sqlite:///{db_path}", experiment_id=exp_id)
                strategy = get_strategy(strat_name, seed=seed)
                oracle = SimulatedOracle(dataset=dataset, seed=seed)

                cfg = LoopConfig(
                    dataset=dataset_name,
                    model_type=model_type,
                    strategy=strat_name,
                    initial_size=initial_size,
                    batch_size=batch_size,
                    max_rounds=rounds,
                    label_budget=effective_budget,
                    target_accuracy=None,  # Run full curve cleanly across seeds for paired statistics
                    seed=seed,
                    max_epochs=epochs,
                    lr=lr,
                    use_synthetic=use_synthetic,
                    max_samples=max_samples,
                    db_url=f"sqlite:///{db_path}",
                )
                orch = ActiveLearningOrchestrator(
                    config=cfg,
                    dataset=dataset,
                    model=model,
                    strategy=strategy,
                    labeler=oracle,
                    db=db,
                )

                t_start = time.perf_counter()
                with tracker.start_run(
                    run_name=f"bench_{dataset_name}_{strat_name}_s{seed}",
                    tags={
                        "dataset": dataset_name,
                        "data_source_type": data_source_type,
                        "strategy": strat_name,
                        "seed": str(seed),
                    },
                ):
                    tracker.log_params(
                        {
                            "dataset": dataset_name,
                            "data_source_type": data_source_type,
                            "strategy": strat_name,
                            "seed": seed,
                            "initial_size": initial_size,
                            "batch_size": batch_size,
                            "rounds": rounds,
                            "epochs": epochs,
                            "max_samples": max_samples,
                        }
                    )
                    orch.run_simulation()
                    history = db.get_all_rounds_history()
                    wall_time = time.perf_counter() - t_start

                    for h in history:
                        tracker.log_round_metrics(
                            round_num=h["round_num"],
                            accuracy=h["test_acc"],
                            macro_f1=h["test_f1"],
                            ece=h["ece"],
                            n_labeled=h["n_labeled"],
                            n_pool=h["n_pool"],
                            nll=h.get("nll"),
                            train_time_sec=h.get("train_time_sec"),
                        )

                    # Confusion matrix on first seed per strategy
                    if seed == seeds[0]:
                        test_ids, _ = db.get_split_samples("test")
                        if test_ids:
                            X_test = dataset.get_features(test_ids)
                            y_true = dataset.get_ground_truth(test_ids)
                            y_pred = np.argmax(model.predict_calibrated_proba(X_test), axis=1).tolist()
                            cm_path = os.path.join(cm_dir, f"cm_{dataset_name}_{strat_name}.png")
                            plot_confusion_matrix(
                                y_true=y_true,
                                y_pred=y_pred,
                                class_names=dataset.class_names,
                                save_path=cm_path,
                                title=f"{dataset_name.upper()} ({data_source_type}) - {strat_name} (seed {seed})",
                            )
                            tracker.log_artifact(cm_path, artifact_path="confusion_matrices")

                db.close()
                benchmark_histories[strat_name].append(history)

                n_lbls = [h["n_labeled"] for h in history]
                accs = [h["test_acc"] for h in history]
                last = history[-1]
                ltt = labels_to_target_accuracy(n_lbls, accs, target_accuracy)
                alc = compute_alc(n_lbls, accs)
                total_train_time = sum(float(h.get("train_time_sec") or 0.0) for h in history) or wall_time
                budget_util = (last["n_labeled"] / max(1, effective_budget)) * 100.0

                seed_final_records[strat_name].append(
                    {
                        "seed": seed,
                        "final_accuracy": float(last["test_acc"]),
                        "final_f1": float(last["test_f1"]),
                        "final_ece": float(last["ece"]),
                        "final_nll": float(last.get("nll") or 0.0),
                        "alc": float(alc),
                        "labels_to_target": ltt,
                        "n_labeled_final": int(last["n_labeled"]),
                        "train_time_sec": float(total_train_time),
                        "budget_utilization_pct": float(min(100.0, budget_util)),
                    }
                )
                print(
                    f"  seed={seed}: acc={last['test_acc']:.4f}, f1={last['test_f1']:.4f}, "
                    f"ece={last['ece']:.4f}, nll={last.get('nll', 0.0):.4f}, alc={alc:.4f}, "
                    f"ltt={ltt if ltt is not None else 'Not Reached'}"
                )

    data_source_label = data_source_type or ("synthetic" if use_synthetic else "real")

    # Core CSV compatible with existing API + extended columns
    savings_df = compute_label_savings(benchmark_histories, target_accuracy=target_accuracy)

    baseline_recs = seed_final_records.get("random", [])
    baseline_by_seed = {r["seed"]: r for r in baseline_recs}
    baseline_ltts = [r["labels_to_target"] for r in baseline_recs if r["labels_to_target"] is not None]
    baseline_ltt_mean = float(np.mean(baseline_ltts)) if baseline_ltts else None

    strategy_stats: List[Dict[str, Any]] = []
    extra_cols: List[Dict[str, Any]] = []

    for strat_name in strategies:
        recs = seed_final_records[strat_name]
        accs = [r["final_accuracy"] for r in recs]
        f1s = [r["final_f1"] for r in recs]
        eces = [r["final_ece"] for r in recs]
        nlls = [r["final_nll"] for r in recs]
        alcs = [r["alc"] for r in recs]
        times = [r["train_time_sec"] for r in recs]
        utils = [r["budget_utilization_pct"] for r in recs]
        ltts = [r["labels_to_target"] for r in recs if r["labels_to_target"] is not None]
        ltt_mean = float(np.mean(ltts)) if ltts else None

        cost_info = compute_configurable_cost_savings(
            strategy_labels_to_target=ltt_mean,
            random_labels_to_target=baseline_ltt_mean,
            cost_per_hour=cost_per_hour,
            seconds_per_annotation=seconds_per_annotation,
        )

        paired_acc = {}
        paired_alc = {}
        if strat_name != "random" and baseline_by_seed:
            common_seeds = [r["seed"] for r in recs if r["seed"] in baseline_by_seed]
            if common_seeds:
                cand_map = {r["seed"]: r for r in recs}
                paired_acc = paired_strategy_comparison(
                    [baseline_by_seed[s]["final_accuracy"] for s in common_seeds],
                    [cand_map[s]["final_accuracy"] for s in common_seeds],
                )
                paired_alc = paired_strategy_comparison(
                    [baseline_by_seed[s]["alc"] for s in common_seeds],
                    [cand_map[s]["alc"] for s in common_seeds],
                )

        acc_m, acc_s = float(np.mean(accs)), (float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0)
        f1_m, f1_s = float(np.mean(f1s)), (float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0)
        ece_m, ece_s = float(np.mean(eces)), (float(np.std(eces, ddof=1)) if len(eces) > 1 else 0.0)
        nll_m, nll_s = float(np.mean(nlls)), (float(np.std(nlls, ddof=1)) if len(nlls) > 1 else 0.0)
        alc_m, alc_s = float(np.mean(alcs)), (float(np.std(alcs, ddof=1)) if len(alcs) > 1 else 0.0)
        time_m = float(np.mean(times))
        util_m = float(np.mean(utils))

        strategy_stats.append(
            {
                "strategy": strat_name,
                "n_seeds": len(recs),
                "final_accuracy_mean": round(acc_m, 6),
                "final_accuracy_std": round(acc_s, 6),
                "final_f1_mean": round(f1_m, 6),
                "final_f1_std": round(f1_s, 6),
                "final_ece_mean": round(ece_m, 6),
                "final_ece_std": round(ece_s, 6),
                "final_nll_mean": round(nll_m, 6),
                "final_nll_std": round(nll_s, 6),
                "alc_mean": round(alc_m, 6),
                "alc_std": round(alc_s, 6),
                "labels_to_target_mean": round(ltt_mean, 2) if ltt_mean is not None else None,
                "train_time_mean_sec": round(time_m, 3),
                "budget_utilization_pct": round(util_m, 2),
                "cost_savings": cost_info,
                "paired_vs_random_acc": paired_acc,
                "paired_vs_random_alc": paired_alc,
                "per_seed": recs,
            }
        )
        extra_cols.append(
            {
                "Strategy": strat_name,
                "Data Source": data_source_label,
                "Acc_Mean": acc_m,
                "Acc_Std": acc_s,
                "F1_Mean": f1_m,
                "F1_Std": f1_s,
                "ECE_Mean": ece_m,
                "ECE_Std": ece_s,
                "NLL_Mean": nll_m,
                "NLL_Std": nll_s,
                "ALC_Mean": alc_m,
                "ALC_Std": alc_s,
                "Labels_to_Target_Mean": ltt_mean if ltt_mean is not None else np.nan,
                "Train Time (s)": round(time_m, 2),
                "Budget Used (%)": round(util_m, 1),
                "Est. Dollars Saved ($)": (
                    cost_info["estimated_dollars_saved"] if cost_info["target_reached_both"] else "Not Reached"
                ),
            }
        )

    extra_df = pd.DataFrame(extra_cols)
    merged_df = savings_df.merge(extra_df, on="Strategy", how="left")

    csv_df = merged_df[
        [
            "Strategy",
            "Data Source",
            "Seeds",
            "Acc_Mean",
            "Acc_Std",
            "F1_Mean",
            "ECE_Mean",
            "NLL_Mean",
            "ALC (Mean +/- 95% CI)",
            "Labels to Target",
            "% Label Reduction vs Random",
            "Train Time (s)",
            "Budget Used (%)",
            "Est. Dollars Saved ($)",
        ]
    ]
    csv_path = os.path.join(output_dir, "benchmark_savings.csv")
    csv_df.to_csv(csv_path, index=False)

    md_path = os.path.join(output_dir, "benchmark_savings.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(
            f"# ActivePulse Studio Benchmark Results - {dataset_name.upper()} ({data_source_label} dataset)\n\n"
        )
        f.write(
            f"- **Seeds:** {seeds}\n"
            f"- **Target Accuracy:** {target_accuracy * 100:.1f}%\n"
            f"- **Budget:** {initial_size} initial + {rounds} rounds x {batch_size}/round = {effective_budget} max labels\n"
            f"- **Cost Assumptions:** ${cost_per_hour:.2f}/hr, {seconds_per_annotation:.1f}s/label\n\n"
        )
        f.write(csv_df.to_markdown(index=False) + "\n")

    # Save visual plots
    curves_path = os.path.join(output_dir, "learning_curves.png")
    plot_learning_curves(
        benchmark_histories,
        output_path=curves_path,
        target_accuracy=target_accuracy,
        title=f"Active Learning Benchmark ({dataset_name.upper()} - {data_source_label})",
    )

    acc_plot_path = os.path.join(output_dir, "accuracy_comparison.png")
    plot_metric_bars_with_error(
        extra_df,
        save_path=acc_plot_path,
        title=f"Accuracy & Macro-F1 Across Strategies ({dataset_name.upper()} - {data_source_label})",
        metric_specs=[
            ("Acc_Mean", "Acc_Std", "Test Accuracy", "Final Test Accuracy (+/- 1 SD)"),
            ("F1_Mean", "F1_Std", "Macro-F1", "Final Macro-F1 (+/- 1 SD)"),
        ],
    )

    eff_plot_path = os.path.join(output_dir, "label_efficiency.png")
    plot_label_efficiency_chart(
        extra_df,
        target_accuracy=target_accuracy,
        save_path=eff_plot_path,
        title=f"Label Efficiency & ALC ({dataset_name.upper()} - {data_source_label})",
    )

    cal_plot_path = os.path.join(output_dir, "calibration_comparison.png")
    plot_metric_bars_with_error(
        extra_df,
        save_path=cal_plot_path,
        title=f"Calibration Quality Comparison ({dataset_name.upper()} - {data_source_label})",
        metric_specs=[
            ("ECE_Mean", "ECE_Std", "ECE (Lower is Better)", "Expected Calibration Error (+/- 1 SD)"),
            ("NLL_Mean", "NLL_Std", "NLL (Lower is Better)", "Negative Log-Likelihood (+/- 1 SD)"),
        ],
    )

    stat_json_path = os.path.join(output_dir, "statistical_summary.json")
    stat_md_path = os.path.join(output_dir, "statistical_summary.md")
    write_statistical_report(
        dataset=dataset_name,
        data_source_type=data_source_label,
        target_accuracy=target_accuracy,
        seeds=seeds,
        cost_per_hour=cost_per_hour,
        seconds_per_annotation=seconds_per_annotation,
        strategy_stats=strategy_stats,
        json_path=stat_json_path,
        md_path=stat_md_path,
    )

    cfg_yaml_path = os.path.join(output_dir, "experiment_config.yaml")
    with open(cfg_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "dataset": dataset_name,
                "data_source_type": data_source_label,
                "strategies": strategies,
                "seeds": seeds,
                "max_samples": max_samples,
                "initial_size": initial_size,
                "batch_size": batch_size,
                "rounds": rounds,
                "epochs": epochs,
                "lr": lr,
                "target_accuracy": target_accuracy,
                "label_budget": effective_budget,
                "device": device,
                "cost_per_hour_usd": cost_per_hour,
                "seconds_per_annotation": seconds_per_annotation,
            },
            f,
            sort_keys=False,
        )

    print(f"\nSaved {dataset_name} benchmark artifacts to: {output_dir}")
    return {"dataset": dataset_name, "data_source_type": data_source_label, "output_dir": output_dir}


def main():
    parser = argparse.ArgumentParser(description="ActivePulse Studio Phase 5 Multi-Seed Benchmark Runner")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=ALL_STRATEGIES,
        help="List of strategies to benchmark",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="Random seeds to evaluate (default: 5 seeds 42..46)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "agnews", "both"],
        help="Dataset(s) to evaluate",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Force synthetic data for fast CPU execution",
    )
    parser.add_argument("--max-samples", type=int, default=600, help="Total sample pool size")
    parser.add_argument("--initial-size", type=int, default=40, help="Cold-start batch size")
    parser.add_argument("--batch-size", type=int, default=20, help="Per-round query batch size")
    parser.add_argument("--rounds", type=int, default=5, help="Number of active learning rounds")
    parser.add_argument("--epochs", type=int, default=6, help="Training epochs per round")
    parser.add_argument("--lr", type=float, default=0.01, help="Optimizer learning rate")
    parser.add_argument("--target-accuracy", type=float, default=0.65, help="Target accuracy for savings calculation")
    parser.add_argument("--label-budget", type=int, default=None, help="Annotation budget cap")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device (cpu or cuda)")
    parser.add_argument("--cost-per-hour", type=float, default=25.0, help="Assumed annotator hourly rate (USD)")
    parser.add_argument("--seconds-per-annotation", type=float, default=8.0, help="Assumed time per annotation (s)")
    parser.add_argument("--mlflow", action="store_true", help="Enable MLflow logging for benchmark runs")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save plots and tables")

    args = parser.parse_args()
    datasets = ["cifar10", "agnews"] if args.dataset == "both" else [args.dataset]

    for ds in datasets:
        sub_dir = os.path.join(args.output_dir, ds) if len(datasets) > 1 else args.output_dir
        run_dataset_benchmark(
            dataset_name=ds,
            strategies=args.strategies,
            seeds=args.seeds,
            use_synthetic=args.synthetic,
            max_samples=args.max_samples,
            initial_size=args.initial_size,
            batch_size=args.batch_size,
            rounds=args.rounds,
            epochs=args.epochs,
            lr=args.lr,
            target_accuracy=args.target_accuracy,
            label_budget=args.label_budget,
            device=args.device,
            output_dir=sub_dir,
            cost_per_hour=args.cost_per_hour,
            seconds_per_annotation=args.seconds_per_annotation,
            enable_mlflow=args.mlflow,
        )

    # When running both datasets, mirror cifar10 root artifacts into results/ for API & compatibility
    if len(datasets) > 1:
        primary_dir = os.path.join(args.output_dir, datasets[0])
        for fname in (
            "benchmark_savings.csv",
            "benchmark_savings.md",
            "learning_curves.png",
            "accuracy_comparison.png",
            "label_efficiency.png",
            "calibration_comparison.png",
            "statistical_summary.json",
            "statistical_summary.md",
            "experiment_config.yaml",
        ):
            src_f = os.path.join(primary_dir, fname)
            if os.path.exists(src_f):
                shutil.copy2(src_f, os.path.join(args.output_dir, fname))


if __name__ == "__main__":
    main()
