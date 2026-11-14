# ActivePulse Studio Statistical Evaluation Report (AGNEWS)

- **Dataset:** `agnews` (synthetic)
- **Seeds:** `[42, 43, 44, 45, 46]` (`n = 5` paired runs per strategy)
- **Target Accuracy:** `60.0%`
- **Cost Model Assumptions:** `$25.00/hr`, `8.0 sec/annotation`

> **Statistical Power Note:** All paired t-tests and Wilcoxon signed-rank tests are computed across n=5 paired random seeds per strategy. With 5 seeds, minimum achievable two-sided Wilcoxon p-value is 0.0625; p-values should be interpreted as exploratory directional evidence alongside mean +/- std and 95% CI.

## Strategy Summary & Paired Comparison vs Random

| Strategy | Final Acc (Mean +/- Std) | Macro-F1 (Mean +/- Std) | ECE | NLL | ALC (Mean +/- Std) | Labels to Target | Savings vs Random | Paired Acc Diff (95% CI) | p-value (t / Wilcoxon) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 0.9178 +/- 0.0279 | 0.9131 +/- 0.0300 | 0.4132 | 0.7319 | 0.6889 +/- 0.0440 | 54.4 | 0.0% | Baseline | t=N/A / w=N/A |
| least_confidence | 0.9511 +/- 0.0149 | 0.9476 +/- 0.0160 | 0.4135 | 0.6532 | 0.7113 +/- 0.0410 | 53.8 | 1.1% | +0.0333 [-0.0135, +0.0801] | t=0.119055 / w=0.1875 |
| margin | 0.9511 +/- 0.0169 | 0.9478 +/- 0.0161 | 0.4232 | 0.6713 | 0.6976 +/- 0.0291 | 55.4 | -1.8% | +0.0333 [-0.0210, +0.0876] | t=0.163608 / w=0.25 |
| entropy | 0.9556 +/- 0.0136 | 0.9520 +/- 0.0126 | 0.4193 | 0.6630 | 0.7104 +/- 0.0262 | 53.6 | 1.5% | +0.0378 [-0.0096, +0.0852] | t=0.091299 / w=0.125 |
| bald | 0.8244 +/- 0.0894 | 0.8106 +/- 0.1028 | 0.3640 | 0.8442 | 0.6271 +/- 0.0565 | 60.8 | -11.8% | -0.0933 [-0.1735, -0.0131] | t=0.031952 / w=0.0625 |
| coreset | 0.8600 +/- 0.0202 | 0.8555 +/- 0.0181 | 0.4024 | 0.8452 | 0.6329 +/- 0.0251 | 60.8 | -11.8% | -0.0578 [-0.0825, -0.0331] | t=0.00289 / w=0.0625 |
| badge | 0.9222 +/- 0.0208 | 0.9187 +/- 0.0197 | 0.4148 | 0.7281 | 0.6720 +/- 0.0321 | 56.6 | -4.0% | +0.0044 [-0.0513, +0.0601] | t=0.835523 / w=1.0 |
| hybrid | 0.9489 +/- 0.0169 | 0.9450 +/- 0.0192 | 0.4163 | 0.6674 | 0.7071 +/- 0.0364 | 52.8 | 2.9% | +0.0311 [-0.0190, +0.0812] | t=0.159935 / w=0.125 |
