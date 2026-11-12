"""Benchmark evaluation: Label savings, Area Under Learning Curve (ALC), and plotting."""

from typing import Dict, List, Tuple, Optional, Any
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_alc(labeled_counts: List[int], accuracies: List[float]) -> float:
    """Compute normalized Area Under the Learning Curve (ALC) using trapezoidal integration.
    
    Returns:
        Float in [0.0, 1.0] representing normalized area.
    """
    if len(labeled_counts) < 2:
        return float(accuracies[0]) if accuracies else 0.0

    x = np.array(labeled_counts, dtype=float)
    y = np.array(accuracies, dtype=float)

    total_range = x[-1] - x[0]
    if total_range <= 0:
        return float(np.mean(y))

    # Trapezoid rule (NumPy 2.0 uses np.trapezoid or scipy.integrate.trapezoid)
    if hasattr(np, "trapezoid"):
        area = np.trapezoid(y, x)
    elif hasattr(np, "trapz"):
        area = np.trapz(y, x)
    else:
        from scipy.integrate import trapezoid
        area = trapezoid(y, x)
    return float(area / total_range)


def labels_to_target_accuracy(
    labeled_counts: List[int],
    accuracies: List[float],
    target_acc: float,
) -> Optional[int]:
    """Find minimum number of labeled samples required to reach or exceed target accuracy.
    
    Uses linear interpolation between rounds if target falls between steps.
    """
    x = np.array(labeled_counts, dtype=float)
    y = np.array(accuracies, dtype=float)

    # If maximum achieved accuracy never reaches target
    if np.max(y) < target_acc:
        return None

    # Check if first point already meets target
    if y[0] >= target_acc:
        return int(x[0])

    # Find first crossing point
    for i in range(len(y) - 1):
        if y[i] < target_acc <= y[i + 1]:
            # Linear interpolation
            slope = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
            interpolated_x = x[i] + (target_acc - y[i]) / slope
            return int(np.round(interpolated_x))

    return int(x[-1])


def compute_label_savings(
    benchmark_results: Dict[str, List[Dict[str, Any]]],
    target_accuracy: float,
    baseline_strategy: str = "random",
) -> pd.DataFrame:
    """Compute label reduction percentage and ALC for each strategy across seeds.
    
    Args:
        benchmark_results: Dict mapping strategy_name -> list of run histories (one per seed).
        target_accuracy: Target accuracy threshold (e.g. 0.85).
        baseline_strategy: Name of baseline strategy to compare against.
        
    Returns:
        pd.DataFrame containing summary statistics (mean ± 95% CI).
    """
    rows = []

    # 1. Compute baseline labels needed across seeds
    baseline_runs = benchmark_results.get(baseline_strategy, [])
    baseline_labels_list = []
    for run in baseline_runs:
        n_lbl = [r["n_labeled"] for r in run]
        accs = [r["test_acc"] for r in run]
        lbl_target = labels_to_target_accuracy(n_lbl, accs, target_accuracy)
        if lbl_target is not None:
            baseline_labels_list.append(lbl_target)

    avg_baseline_labels = np.mean(baseline_labels_list) if baseline_labels_list else None

    # 2. Compute metrics per strategy
    for strat_name, runs in benchmark_results.items():
        alc_list = []
        labels_needed_list = []
        reduction_list = []

        for run in runs:
            n_lbl = [r["n_labeled"] for r in run]
            accs = [r["test_acc"] for r in run]

            # ALC
            alc = compute_alc(n_lbl, accs)
            alc_list.append(alc)

            # Labels to target
            lbl_needed = labels_to_target_accuracy(n_lbl, accs, target_accuracy)
            if lbl_needed is not None:
                labels_needed_list.append(lbl_needed)
                if avg_baseline_labels and avg_baseline_labels > 0:
                    reduction = ((avg_baseline_labels - lbl_needed) / avg_baseline_labels) * 100.0
                    reduction_list.append(reduction)

        n_seeds = len(runs)
        mean_alc = np.mean(alc_list) if alc_list else 0.0
        ci_alc = 1.96 * (np.std(alc_list) / np.sqrt(n_seeds)) if n_seeds > 1 else 0.0

        mean_labels = np.mean(labels_needed_list) if labels_needed_list else np.nan
        ci_labels = 1.96 * (np.std(labels_needed_list) / np.sqrt(len(labels_needed_list))) if len(labels_needed_list) > 1 else 0.0

        mean_reduct = np.mean(reduction_list) if reduction_list else 0.0
        ci_reduct = 1.96 * (np.std(reduction_list) / np.sqrt(len(reduction_list))) if len(reduction_list) > 1 else 0.0

        rows.append({
            "Strategy": strat_name,
            "Seeds": n_seeds,
            "ALC (Mean +/- 95% CI)": f"{mean_alc:.4f} +/- {ci_alc:.4f}",
            "Labels to Target": f"{mean_labels:.1f} +/- {ci_labels:.1f}" if not np.isnan(mean_labels) else "Not Reached",
            "% Label Reduction vs Random": f"{mean_reduct:+.1f}% +/- {ci_reduct:.1f}%" if strat_name != baseline_strategy else "0.0% (Baseline)",
            "_raw_alc_mean": mean_alc,
            "_raw_labels_mean": mean_labels,
            "_raw_reduction_mean": mean_reduct,
        })

    df = pd.DataFrame(rows)
    return df


def plot_learning_curves(
    benchmark_results: Dict[str, List[Dict[str, Any]]],
    output_path: str = "results/learning_curves.png",
    target_accuracy: Optional[float] = None,
    title: str = "Active Learning Benchmark: Learning Curves",
) -> None:
    """Plot mean learning curves with 95% confidence intervals per strategy.
    
    Saves high-res publication chart to output_path.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # Style palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(benchmark_results)))

    for idx, (strat_name, runs) in enumerate(benchmark_results.items()):
        color = colors[idx]
        # Align all runs by round_num
        max_len = max(len(r) for r in runs)
        labeled_matrix = []
        acc_matrix = []

        for r in runs:
            n_lbl = [step["n_labeled"] for step in r]
            accs = [step["test_acc"] for step in r]
            labeled_matrix.append(n_lbl)
            acc_matrix.append(accs)

        x_vals = labeled_matrix[0]
        acc_arr = np.array(acc_matrix)  # (n_seeds, n_rounds)

        mean_acc = np.mean(acc_arr, axis=0)
        std_acc = np.std(acc_arr, axis=0)
        n_seeds = len(runs)
        ci_95 = 1.96 * (std_acc / np.sqrt(n_seeds)) if n_seeds > 1 else std_acc

        ax.plot(x_vals, mean_acc, label=strat_name, color=color, linewidth=2.2, marker="o", markersize=4)
        ax.fill_between(x_vals, mean_acc - ci_95, mean_acc + ci_95, color=color, alpha=0.15)

    if target_accuracy is not None:
        ax.axhline(
            y=target_accuracy,
            color="red",
            linestyle="--",
            alpha=0.7,
            label=f"Target Acc ({target_accuracy:.0%})",
        )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Number of Labeled Samples", fontsize=12, labelpad=8)
    ax.set_ylabel("Test Accuracy", fontsize=12, labelpad=8)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", frameon=True, fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
