"""Phase 5 tests: statistical analysis, unequal-length alignment, cost assumptions, and plots."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

from src.eval.savings import compute_label_savings, plot_learning_curves
from src.eval.statistics import (
    align_learning_curves,
    compute_configurable_cost_savings,
    paired_strategy_comparison,
    plot_confusion_matrix,
    plot_label_efficiency_chart,
    plot_metric_bars_with_error,
    write_statistical_report,
)


def test_unequal_length_curve_alignment_and_plot(tmp_path):
    # One seed stops early (length 3), second runs length 5
    results = {
        "random": [
            [{"n_labeled": 20, "test_acc": 0.40}, {"n_labeled": 40, "test_acc": 0.50}, {"n_labeled": 60, "test_acc": 0.55}],
            [
                {"n_labeled": 20, "test_acc": 0.42},
                {"n_labeled": 40, "test_acc": 0.52},
                {"n_labeled": 60, "test_acc": 0.58},
                {"n_labeled": 80, "test_acc": 0.62},
                {"n_labeled": 100, "test_acc": 0.65},
            ],
        ],
        "entropy": [
            [{"n_labeled": 20, "test_acc": 0.41}, {"n_labeled": 35, "test_acc": 0.56}, {"n_labeled": 50, "test_acc": 0.68}],
            [{"n_labeled": 20, "test_acc": 0.43}, {"n_labeled": 45, "test_acc": 0.63}],
        ],
    }
    grid, mean_acc, std_acc = align_learning_curves(
        [[r["n_labeled"] for r in run] for run in results["random"]],
        [[r["test_acc"] for r in run] for run in results["random"]],
    )
    assert list(grid) == [20.0, 40.0, 60.0, 80.0, 100.0]
    assert np.all(np.isfinite(mean_acc))
    assert np.all(np.isfinite(std_acc))

    out_png = tmp_path / "curves.png"
    plot_learning_curves(results, output_path=str(out_png), target_accuracy=0.65)
    assert out_png.exists() and out_png.stat().st_size > 1000


def test_truthful_savings_when_target_not_reached():
    results = {
        "random": [[{"n_labeled": 20, "test_acc": 0.50}, {"n_labeled": 40, "test_acc": 0.58}]],
        "badge": [[{"n_labeled": 20, "test_acc": 0.50}, {"n_labeled": 40, "test_acc": 0.62}]],
    }
    df = compute_label_savings(results, target_accuracy=0.90)
    assert df.loc[df["Strategy"] == "badge", "Labels to Target"].iloc[0] == "Not Reached"
    assert df.loc[df["Strategy"] == "badge", "% Label Reduction vs Random"].iloc[0] == "Not Reached"

    cs_unreached = compute_configurable_cost_savings(None, 100.0)
    assert cs_unreached["target_reached_both"] is False
    assert cs_unreached["status_text"] == "Not Reached"

    cs_reached = compute_configurable_cost_savings(60.0, 100.0, cost_per_hour=30.0, seconds_per_annotation=12.0)
    assert cs_reached["target_reached_both"] is True
    assert cs_reached["label_reduction_pct"] == pytest.approx(40.0)
    assert cs_reached["labels_saved"] == pytest.approx(40.0)
    assert cs_reached["estimated_dollars_saved"] > 0.0


def test_paired_tests_and_artifact_generation(tmp_path):
    base = [0.60, 0.61, 0.59, 0.62, 0.60]
    cand = [0.68, 0.70, 0.66, 0.69, 0.67]
    comp = paired_strategy_comparison(base, cand)
    assert comp["n_paired_seeds"] == 5
    assert comp["mean_diff"] > 0.07
    assert comp["paired_t_pvalue"] is not None
    assert comp["paired_t_pvalue"] < 0.05
    assert comp["ci95_low"] > 0.0
    assert "Wilcoxon" in comp["note"]

    summary_df = pd.DataFrame(
        [
            {
                "Strategy": "random",
                "Acc_Mean": 0.604,
                "Acc_Std": 0.011,
                "F1_Mean": 0.59,
                "F1_Std": 0.012,
                "ECE_Mean": 0.05,
                "ECE_Std": 0.01,
                "NLL_Mean": 0.92,
                "NLL_Std": 0.04,
                "ALC_Mean": 0.56,
                "ALC_Std": 0.02,
                "Labels_to_Target_Mean": np.nan,
            },
            {
                "Strategy": "entropy",
                "Acc_Mean": 0.680,
                "Acc_Std": 0.015,
                "F1_Mean": 0.67,
                "F1_Std": 0.014,
                "ECE_Mean": 0.03,
                "ECE_Std": 0.008,
                "NLL_Mean": 0.78,
                "NLL_Std": 0.03,
                "ALC_Mean": 0.63,
                "ALC_Std": 0.018,
                "Labels_to_Target_Mean": 95.0,
            },
        ]
    )

    acc_png = tmp_path / "accuracy_comparison.png"
    eff_png = tmp_path / "label_efficiency.png"
    cal_png = tmp_path / "calibration_comparison.png"
    cm_png = tmp_path / "cm.png"
    rep_json = tmp_path / "statistical_summary.json"
    rep_md = tmp_path / "statistical_summary.md"

    plot_metric_bars_with_error(
        summary_df,
        str(acc_png),
        "Accuracy & F1",
        [("Acc_Mean", "Acc_Std", "Accuracy", "Final Test Accuracy"), ("F1_Mean", "F1_Std", "F1", "Final Macro-F1")],
    )
    plot_label_efficiency_chart(summary_df, 0.70, str(eff_png), "Efficiency")
    plot_metric_bars_with_error(
        summary_df,
        str(cal_png),
        "Calibration",
        [("ECE_Mean", "ECE_Std", "ECE", "ECE (Lower Better)"), ("NLL_Mean", "NLL_Std", "NLL", "NLL (Lower Better)")],
    )
    plot_confusion_matrix([0, 1, 0, 1], [0, 1, 1, 1], ["World", "Sports"], str(cm_png), "CM")
    write_statistical_report(
        dataset="agnews",
        data_source_type="synthetic",
        target_accuracy=0.70,
        seeds=[42, 43, 44, 45, 46],
        cost_per_hour=25.0,
        seconds_per_annotation=8.0,
        strategy_stats=[
            {
                "strategy": "random",
                "final_accuracy_mean": 0.604,
                "final_accuracy_std": 0.011,
                "final_f1_mean": 0.59,
                "final_f1_std": 0.012,
                "final_ece_mean": 0.05,
                "final_nll_mean": 0.92,
                "alc_mean": 0.56,
                "alc_std": 0.02,
                "labels_to_target_mean": None,
                "cost_savings": {"status_text": "Not Reached"},
                "paired_vs_random_acc": {},
            },
            {
                "strategy": "entropy",
                "final_accuracy_mean": 0.680,
                "final_accuracy_std": 0.015,
                "final_f1_mean": 0.67,
                "final_f1_std": 0.014,
                "final_ece_mean": 0.03,
                "final_nll_mean": 0.78,
                "alc_mean": 0.63,
                "alc_std": 0.018,
                "labels_to_target_mean": 95.0,
                "cost_savings": {"status_text": "Not Reached"},
                "paired_vs_random_acc": comp,
            },
        ],
        json_path=str(rep_json),
        md_path=str(rep_md),
    )

    for p in (acc_png, eff_png, cal_png, cm_png, rep_json, rep_md):
        assert p.exists() and p.stat().st_size > 100
    parsed = json.loads(rep_json.read_text(encoding="utf-8"))
    assert parsed["data_source_type"] == "synthetic"
    assert len(parsed["strategies"]) == 2

