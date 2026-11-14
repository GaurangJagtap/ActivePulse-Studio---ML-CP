# ActivePulse Studio Statistical Evaluation Report (CIFAR10)

- **Dataset:** `cifar10` (synthetic)
- **Seeds:** `[42, 43, 44, 45, 46]` (`n = 5` paired runs per strategy)
- **Target Accuracy:** `60.0%`
- **Cost Model Assumptions:** `$25.00/hr`, `8.0 sec/annotation`

> **Statistical Power Note:** All paired t-tests and Wilcoxon signed-rank tests are computed across n=5 paired random seeds per strategy. With 5 seeds, minimum achievable two-sided Wilcoxon p-value is 0.0625; p-values should be interpreted as exploratory directional evidence alongside mean +/- std and 95% CI.

## Strategy Summary & Paired Comparison vs Random

| Strategy | Final Acc (Mean +/- Std) | Macro-F1 (Mean +/- Std) | ECE | NLL | ALC (Mean +/- Std) | Labels to Target | Savings vs Random | Paired Acc Diff (95% CI) | p-value (t / Wilcoxon) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 0.8356 +/- 0.0396 | 0.8042 +/- 0.0574 | 0.1474 | 1.2415 | 0.5120 +/- 0.0532 | 67.4 | 0.0% | Baseline | t=N/A / w=N/A |
| least_confidence | 0.7956 +/- 0.0688 | 0.7504 +/- 0.0687 | 0.1723 | 1.7481 | 0.4738 +/- 0.0560 | 74.2 | -10.1% | -0.0400 [-0.1682, +0.0882] | t=0.435086 / w=0.9375 |
| margin | 0.8689 +/- 0.0265 | 0.8322 +/- 0.0136 | 0.1419 | 0.7775 | 0.5064 +/- 0.1150 | 71.0 | -5.3% | +0.0333 [+0.0164, +0.0502] | t=0.005408 / w=0.0625 |
| entropy | 0.8356 +/- 0.0396 | 0.7945 +/- 0.0516 | 0.1259 | 1.0602 | 0.4764 +/- 0.0564 | 74.8 | -11.0% | +0.0000 [-0.0552, +0.0552] | t=1.0 / w=1.0 |
| bald | 0.7889 +/- 0.0820 | 0.7455 +/- 0.1131 | 0.1586 | 1.2808 | 0.5904 +/- 0.1034 | 57.4 | 14.8% | -0.0467 [-0.1744, +0.0811] | t=0.367729 / w=0.375 |
| coreset | 0.8378 +/- 0.0449 | 0.7903 +/- 0.0692 | 0.1410 | 1.3380 | 0.5207 +/- 0.0581 | 64.4 | 4.5% | +0.0022 [-0.0335, +0.0379] | t=0.871219 / w=0.75 |
| badge | 0.8867 +/- 0.0346 | 0.8645 +/- 0.0493 | 0.0776 | 0.2980 | 0.5367 +/- 0.0735 | 66.4 | 1.5% | +0.0511 [+0.0090, +0.0932] | t=0.027969 / w=0.0625 |
| hybrid | 0.8578 +/- 0.0709 | 0.8144 +/- 0.1103 | 0.1054 | 1.0596 | 0.5182 +/- 0.0766 | 69.8 | -3.6% | +0.0222 [-0.0246, +0.0690] | t=0.257684 / w=0.3125 |
