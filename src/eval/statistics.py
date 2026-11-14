"""Statistical analysis, unequal-length curve alignment, cost assumptions, and Phase 5 plots."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def align_learning_curves(
    labeled_series: Sequence[Sequence[int]],
    metric_series: Sequence[Sequence[float]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align potentially unequal-length learning curves onto a common sorted label grid.

    Uses piecewise linear interpolation within each run's observed `[min_x, max_x]` range
    and NaN outside observed bounds so `nanmean`/`nanstd` work cleanly without shape errors.
    """
    valid_pairs = [
        (np.asarray(xs, dtype=float), np.asarray(ys, dtype=float))
        for xs, ys in zip(labeled_series, metric_series)
        if len(xs) > 0 and len(xs) == len(ys)
    ]
    if not valid_pairs:
        return np.array([]), np.array([]), np.array([])

    grid = np.unique(np.concatenate([xs for xs, _ in valid_pairs]))
    aligned = np.full((len(valid_pairs), len(grid)), np.nan, dtype=float)

    for row_idx, (xs, ys) in enumerate(valid_pairs):
        order = np.argsort(xs)
        xs_s = xs[order]
        ys_s = ys[order]
        mask = (grid >= xs_s[0]) & (grid <= xs_s[-1])
        if np.any(mask):
            aligned[row_idx, mask] = np.interp(grid[mask], xs_s, ys_s)

    mean_vals = np.nanmean(aligned, axis=0)
    std_vals = np.nanstd(aligned, axis=0)
    return grid, mean_vals, std_vals


def paired_strategy_comparison(
    baseline_values: Sequence[float],
    candidate_values: Sequence[float],
    confidence: float = 0.95,
) -> Dict[str, Any]:
    """Compute paired difference summary, 95% CI, paired t-test, and Wilcoxon signed-rank test."""
    base = np.asarray(baseline_values, dtype=float)
    cand = np.asarray(candidate_values, dtype=float)
    if base.shape != cand.shape:
        raise ValueError("Paired baseline and candidate arrays must have matching lengths.")

    valid = np.isfinite(base) & np.isfinite(cand)
    base = base[valid]
    cand = cand[valid]
    n = int(len(base))
    if n == 0:
        return {
            "n_paired_seeds": 0,
            "mean_diff": None,
            "std_diff": None,
            "ci95_low": None,
            "ci95_high": None,
            "paired_t_stat": None,
            "paired_t_pvalue": None,
            "wilcoxon_stat": None,
            "wilcoxon_pvalue": None,
            "note": "No finite paired observations available.",
        }

    diffs = cand - base
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1)) if n > 1 else 0.0

    ci_low: Optional[float] = mean_diff
    ci_high: Optional[float] = mean_diff
    t_stat: Optional[float] = None
    t_pval: Optional[float] = None
    w_stat: Optional[float] = None
    w_pval: Optional[float] = None

    if n >= 2 and std_diff > 1e-12:
        se = std_diff / np.sqrt(n)
        t_crit = float(stats.t.ppf((1.0 + confidence) / 2.0, df=n - 1))
        ci_low = round(mean_diff - t_crit * se, 6)
        ci_high = round(mean_diff + t_crit * se, 6)
        res_t = stats.ttest_rel(cand, base)
        t_stat = round(float(res_t.statistic), 6) if np.isfinite(res_t.statistic) else None
        t_pval = round(float(res_t.pvalue), 6) if np.isfinite(res_t.pvalue) else None
    elif n >= 2:
        ci_low = round(mean_diff, 6)
        ci_high = round(mean_diff, 6)

    non_zero = diffs[np.abs(diffs) > 1e-12]
    if len(non_zero) >= 1 and n >= 2:
        try:
            res_w = stats.wilcoxon(cand, base, zero_method="wilcox")
            w_stat = round(float(res_w.statistic), 6) if np.isfinite(res_w.statistic) else None
            w_pval = round(float(res_w.pvalue), 6) if np.isfinite(res_w.pvalue) else None
        except Exception:
            w_stat, w_pval = None, None

    caveat = (
        f"Paired comparison across n={n} seeds. "
        + (
            "With <=5 seeds, non-parametric Wilcoxon power is limited (min two-sided p=0.0625); "
            "interpret p-values alongside mean +/- std and 95% CI."
            if n <= 5
            else "Interpret alongside effect size and confidence interval."
        )
    )

    return {
        "n_paired_seeds": n,
        "mean_diff": round(mean_diff, 6),
        "std_diff": round(std_diff, 6),
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "paired_t_stat": t_stat,
        "paired_t_pvalue": t_pval,
        "wilcoxon_stat": w_stat,
        "wilcoxon_pvalue": w_pval,
        "note": caveat,
    }


def compute_configurable_cost_savings(
    strategy_labels_to_target: Optional[float],
    random_labels_to_target: Optional[float],
    cost_per_hour: float = 25.0,
    seconds_per_annotation: float = 8.0,
) -> Dict[str, Any]:
    """Compute label reduction and estimated cost savings strictly when target is reached.

    If either Random or the Strategy did not reach target accuracy, returns "Not Reached"
    fields rather than fabricating savings estimates.
    """
    if strategy_labels_to_target is None or random_labels_to_target is None or random_labels_to_target <= 0:
        return {
            "target_reached_both": False,
            "label_reduction_pct": None,
            "labels_saved": None,
            "estimated_hours_saved": None,
            "estimated_dollars_saved": None,
            "status_text": "Not Reached",
        }

    saved = float(random_labels_to_target) - float(strategy_labels_to_target)
    reduction_pct = (saved / float(random_labels_to_target)) * 100.0
    hours_saved = (saved * float(seconds_per_annotation)) / 3600.0
    dollars_saved = hours_saved * float(cost_per_hour)
    return {
        "target_reached_both": True,
        "label_reduction_pct": round(reduction_pct, 2),
        "labels_saved": round(saved, 2),
        "estimated_hours_saved": round(hours_saved, 4),
        "estimated_dollars_saved": round(dollars_saved, 2),
        "status_text": f"{reduction_pct:.1f}%",
    }


def plot_metric_bars_with_error(
    summary_df: pd.DataFrame,
    save_path: str,
    title: str,
    metric_specs: Sequence[Tuple[str, str, str, str]],
) -> None:
    """Plot grouped or multi-panel bar charts with mean +/- std error bars across strategies."""
    if summary_df.empty:
        return
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    strategies = summary_df["Strategy"].tolist()
    x = np.arange(len(strategies))

    n_panels = len(metric_specs)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4.8), squeeze=False)
    palette = ["#4568D9", "#159C91", "#718BE0", "#6366F1"]

    for idx, (mean_col, std_col, ylabel, panel_title) in enumerate(metric_specs):
        ax = axes[0, idx]
        means = pd.to_numeric(summary_df[mean_col], errors="coerce").fillna(0.0).to_numpy()
        stds = (
            pd.to_numeric(summary_df[std_col], errors="coerce").fillna(0.0).to_numpy()
            if std_col in summary_df.columns
            else np.zeros_like(means)
        )
        bars = ax.bar(
            x,
            means,
            yerr=stds,
            capsize=4,
            color=palette[idx % len(palette)],
            alpha=0.88,
            edgecolor="#1E293B",
            linewidth=0.7,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(strategies, rotation=28, ha="right", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(panel_title, fontsize=11, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")

        for bar, val in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_label_efficiency_chart(
    summary_df: pd.DataFrame,
    target_accuracy: float,
    save_path: str,
    title: str,
) -> None:
    """Plot Labels-to-Target (or ALC when target not reached) per strategy."""
    if summary_df.empty:
        return
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    strategies = summary_df["Strategy"].tolist()
    raw_ltt = summary_df["Labels_to_Target_Mean"] if "Labels_to_Target_Mean" in summary_df.columns else []
    reached_mask = [pd.notna(v) for v in raw_ltt]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    x = np.arange(len(strategies))

    # Left panel: Area Under Learning Curve (always defined)
    alc_means = pd.to_numeric(summary_df["ALC_Mean"], errors="coerce").fillna(0.0).to_numpy()
    alc_stds = pd.to_numeric(summary_df["ALC_Std"], errors="coerce").fillna(0.0).to_numpy()
    ax1.bar(x, alc_means, yerr=alc_stds, capsize=4, color="#4568D9", alpha=0.88)
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategies, rotation=28, ha="right", fontsize=9)
    ax1.set_ylabel("Area Under Learning Curve (ALC)")
    ax1.set_title("Area Under Learning Curve (Higher is Better)")
    ax1.grid(True, axis="y", alpha=0.3, linestyle="--")

    # Right panel: Labels to Target Accuracy (annotating Not Reached cleanly)
    ltt_vals = [float(v) if pd.notna(v) else 0.0 for v in raw_ltt] if len(raw_ltt) else [0.0] * len(strategies)
    colors = ["#159C91" if ok else "#CBD5E1" for ok in reached_mask]
    bars = ax2.bar(x, ltt_vals, color=colors, edgecolor="#334155", linewidth=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(strategies, rotation=28, ha="right", fontsize=9)
    ax2.set_ylabel(f"Mean Labels to {target_accuracy * 100:.1f}% Acc")
    ax2.set_title("Labels to Target Accuracy (Lower is Better)")
    ax2.grid(True, axis="y", alpha=0.3, linestyle="--")

    for bar, ok, val in zip(bars, reached_mask, ltt_vals):
        label = f"{val:.0f}" if ok else "Not Reached"
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            max(val, 0.0),
            label,
            ha="center",
            va="bottom",
            fontsize=8,
            color="#0F172A" if ok else "#64748B",
        )

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
    save_path: str,
    title: str,
) -> None:
    """Plot and save a normalized + count confusion matrix heatmap PNG."""
    from sklearn.metrics import confusion_matrix

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, np.where(row_sums == 0, 1, row_sums), dtype=float)

    fig, ax = plt.subplots(figsize=(max(5.5, len(class_names) * 0.72), max(4.8, len(class_names) * 0.62)))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(labels)
    ax.set_yticks(labels)
    ax.set_xticklabels(class_names, rotation=35, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    ax.set_title(title, fontsize=11, fontweight="bold")

    thresh = 0.55
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm_norm[i, j] > thresh else "#0F172A"
            ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]*100:.0f}%)", ha="center", va="center", color=color, fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_statistical_report(
    dataset: str,
    data_source_type: str,
    target_accuracy: float,
    seeds: Sequence[int],
    cost_per_hour: float,
    seconds_per_annotation: float,
    strategy_stats: List[Dict[str, Any]],
    json_path: str,
    md_path: str,
) -> None:
    """Persist structured JSON and human-readable Markdown statistical reports."""
    os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
    payload = {
        "dataset": dataset,
        "data_source_type": data_source_type,
        "target_accuracy": target_accuracy,
        "seeds": list(seeds),
        "n_seeds": len(seeds),
        "cost_assumptions": {
            "cost_per_hour_usd": cost_per_hour,
            "seconds_per_annotation": seconds_per_annotation,
        },
        "statistical_power_caveat": (
            f"All paired t-tests and Wilcoxon signed-rank tests are computed across n={len(seeds)} "
            "paired random seeds per strategy. With 5 seeds, minimum achievable two-sided Wilcoxon p-value "
            "is 0.0625; p-values should be interpreted as exploratory directional evidence alongside mean +/- std and 95% CI."
        ),
        "strategies": strategy_stats,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    lines = [
        f"# ActivePulse Studio Statistical Evaluation Report ({dataset.upper()})",
        "",
        f"- **Dataset:** `{dataset}` ({data_source_type})",
        f"- **Seeds:** `{list(seeds)}` (`n = {len(seeds)}` paired runs per strategy)",
        f"- **Target Accuracy:** `{target_accuracy * 100:.1f}%`",
        f"- **Cost Model Assumptions:** `${cost_per_hour:.2f}/hr`, `{seconds_per_annotation:.1f} sec/annotation`",
        "",
        "> **Statistical Power Note:** " + payload["statistical_power_caveat"],
        "",
        "## Strategy Summary & Paired Comparison vs Random",
        "",
        "| Strategy | Final Acc (Mean +/- Std) | Macro-F1 (Mean +/- Std) | ECE | NLL | ALC (Mean +/- Std) | Labels to Target | Savings vs Random | Paired Acc Diff (95% CI) | p-value (t / Wilcoxon) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in strategy_stats:
        p_t = row.get("paired_vs_random_acc", {}).get("paired_t_pvalue")
        p_w = row.get("paired_vs_random_acc", {}).get("wilcoxon_pvalue")
        ci_l = row.get("paired_vs_random_acc", {}).get("ci95_low")
        ci_h = row.get("paired_vs_random_acc", {}).get("ci95_high")
        diff_m = row.get("paired_vs_random_acc", {}).get("mean_diff")

        if diff_m is not None and ci_l is not None and ci_h is not None:
            ci_str = f"{diff_m:+.4f} [{ci_l:+.4f}, {ci_h:+.4f}]"
        else:
            ci_str = "Baseline"
        pval_str = f"t={p_t if p_t is not None else 'N/A'} / w={p_w if p_w is not None else 'N/A'}"
        ltt_str = f"{row['labels_to_target_mean']:.1f}" if row.get("labels_to_target_mean") is not None else "Not Reached"
        sav_str = row.get("cost_savings", {}).get("status_text", "Not Reached")
        lines.append(
            f"| {row['strategy']} | "
            f"{row['final_accuracy_mean']:.4f} +/- {row['final_accuracy_std']:.4f} | "
            f"{row['final_f1_mean']:.4f} +/- {row['final_f1_std']:.4f} | "
            f"{row['final_ece_mean']:.4f} | "
            f"{row['final_nll_mean']:.4f} | "
            f"{row['alc_mean']:.4f} +/- {row['alc_std']:.4f} | "
            f"{ltt_str} | {sav_str} | {ci_str} | {pval_str} |"
        )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

