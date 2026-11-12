"""Multi-strategy, multi-seed benchmark runner for Active Learning."""

import argparse
import os
import sys
import tempfile
import json
import torch
import pandas as pd

# Add repo root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.loop.state import LoopConfig
from src.loop.orchestrator import ActiveLearningOrchestrator
from src.data.cifar10 import CIFAR10Adapter
from src.data.agnews import AGNewsAdapter
from src.data.db import DatabaseManager
from src.models.cnn import CIFARConvNet, ResNet18TransferNet
from src.models.text_net import TextMLP
from src.models.wrapper import ActiveLearningClassifier
from src.strategies.registry import get_strategy, list_registered_strategies
from src.labeling.oracle import SimulatedOracle
from src.eval.savings import compute_label_savings, plot_learning_curves


def parse_args():
    parser = argparse.ArgumentParser(description="Active Learning Multi-Seed Benchmark Runner")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["random", "least_confidence", "margin", "entropy", "hybrid", "coreset", "badge"],
        help="List of strategies to benchmark",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44], help="Random seeds to evaluate")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "agnews"])
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic data for fast CPU execution")
    parser.add_argument("--max-samples", type=int, default=800, help="Total sample pool size")
    parser.add_argument("--initial-size", type=int, default=40, help="Cold-start batch size")
    parser.add_argument("--batch-size", type=int, default=20, help="Per-round query batch size")
    parser.add_argument("--rounds", type=int, default=4, help="Number of active learning rounds")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs per round")
    parser.add_argument("--target-accuracy", type=float, default=0.20, help="Target accuracy for savings calculation")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save plots and tables")
    return parser.parse_args()


def run_benchmark():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 75)
    print("ACTIVE LEARNING BENCHMARK EXECUTION")
    print(f"Strategies: {args.strategies}")
    print(f"Seeds: {args.seeds} | Dataset: {args.dataset} | Batch Size: {args.batch_size} | Rounds: {args.rounds}")
    print("=" * 75)

    benchmark_results = {}  # strategy -> list of runs (each run is list of dicts)

    for strat_name in args.strategies:
        print(f"\n>>> Benchmarking Strategy: '{strat_name}' across {len(args.seeds)} seeds...")
        strategy_runs = []

        for seed in args.seeds:
            print(f"  -> Running seed {seed}...")
            torch.manual_seed(seed)

            # Temp DB per run
            fd, db_path = tempfile.mkstemp(suffix=f"_{strat_name}_{seed}.db")
            os.close(fd)
            db = DatabaseManager(f"sqlite:///{db_path}")

            # 1. Dataset
            if args.dataset == "cifar10":
                dataset = CIFAR10Adapter(max_samples=args.max_samples, use_synthetic=args.synthetic, seed=seed)
                module_cls = lambda: CIFARConvNet(num_classes=dataset.num_classes)
            else:
                dataset = AGNewsAdapter(max_samples=args.max_samples, use_synthetic=args.synthetic, seed=seed)
                module_cls = lambda: TextMLP(in_features=64, num_classes=dataset.num_classes)

            # 2. Model
            model = ActiveLearningClassifier(
                module=module_cls,
                max_epochs=args.epochs,
                lr=0.005,
                device="cpu",
                batch_size=32,
                verbose=0,
            )

            # 3. Strategy
            strategy = get_strategy(strat_name, seed=seed)

            # 4. Oracle
            oracle = SimulatedOracle(dataset=dataset, noise_rate=0.0, seed=seed)

            # 5. Config
            config = LoopConfig(
                dataset=args.dataset,
                strategy=strat_name,
                initial_size=args.initial_size,
                batch_size=args.batch_size,
                max_rounds=args.rounds,
                seed=seed,
                db_url=f"sqlite:///{db_path}",
            )

            # 6. Orchestrator
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

            db.close()
            try:
                os.remove(db_path)
            except OSError:
                pass

        benchmark_results[strat_name] = strategy_runs

    # 1. Compute Savings Table
    savings_df = compute_label_savings(
        benchmark_results=benchmark_results,
        target_accuracy=args.target_accuracy,
        baseline_strategy="random",
    )

    clean_table = savings_df[["Strategy", "Seeds", "ALC (Mean +/- 95% CI)", "Labels to Target", "% Label Reduction vs Random"]]
    md_table = clean_table.to_markdown(index=False)
    csv_path = os.path.join(args.output_dir, "benchmark_savings.csv")
    md_path = os.path.join(args.output_dir, "benchmark_savings.md")

    clean_table.to_csv(csv_path, index=False)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Active Learning Benchmark Results\n\n")
        f.write(md_table)
        f.write("\n")

    # 2. Plot Learning Curves
    plot_path = os.path.join(args.output_dir, "learning_curves.png")
    plot_learning_curves(
        benchmark_results=benchmark_results,
        output_path=plot_path,
        target_accuracy=args.target_accuracy,
        title=f"Active Learning Learning Curves ({args.dataset.upper()} - {len(args.seeds)} Seeds)",
    )

    print("\n" + "=" * 75)
    print("BENCHMARK COMPLETED SUCCESSFULLY!")
    print("=" * 75)
    print(md_table)
    print(f"\nArtifacts saved:")
    print(f"  - Plot:  {plot_path}")
    print(f"  - Table: {csv_path}")
    print(f"  - Report:{md_path}")


if __name__ == "__main__":
    run_benchmark()
