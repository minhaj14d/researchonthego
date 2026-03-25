"""
Day 3: Statistical Significance Testing.

Tests whether the observed stability differences are statistically significant:
1. Wilcoxon signed-rank: paired test across seeds (XGBoost vs LogReg on same seed/dataset)
2. Mann-Whitney U: unpaired comparison across perturbation types
3. Effect sizes: Cohen's d for practical significance
4. Confidence intervals: Bootstrap 95% CI for all key metrics
"""
import time
from datetime import datetime
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, mannwhitneyu, friedmanchisquare

from config import RESULTS_DIR


def cohens_d(group1, group2):
    """Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std < 1e-10:
        return 0.0
    return float((np.mean(group1) - np.mean(group2)) / pooled_std)


def bootstrap_ci(data, n_bootstrap=1000, ci=0.95, seed=42):
    """Bootstrap confidence interval for the mean."""
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=len(data), replace=True)
        means.append(np.mean(sample))
    lower = np.percentile(means, (1 - ci) / 2 * 100)
    upper = np.percentile(means, (1 + ci) / 2 * 100)
    return float(lower), float(upper), float(np.mean(means))


def run_statistical_tests():
    print("=" * 70)
    print("DAY 3: Statistical Significance Tests")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    df = pd.read_csv(RESULTS_DIR / "stability_results.csv")
    df_faith = pd.read_csv(RESULTS_DIR / "faithfulness_results.csv")

    results = []

    # ============================================================
    # TEST 1: XGBoost vs LogReg Spearman ρ (paired across seeds)
    # ============================================================
    print("\n--- Test 1: XGBoost vs LogReg (Wilcoxon signed-rank on Spearman ρ) ---")
    for dataset in df["dataset"].unique():
        for pert in df["perturbation"].unique():
            if pert == "none":
                continue
            for level in df["level"].unique():
                if level == 0.0:
                    continue
                xgb = df[
                    (df["dataset"] == dataset) & (df["model"] == "xgboost") &
                    (df["perturbation"] == pert) & (df["level"] == level)
                ]["spearman_rho"].values
                lr = df[
                    (df["dataset"] == dataset) & (df["model"] == "logistic_regression") &
                    (df["perturbation"] == pert) & (df["level"] == level)
                ]["spearman_rho"].values

                if len(xgb) < 3 or len(lr) < 3:
                    continue

                diff = xgb - lr
                if np.all(diff == 0):
                    stat, p = 0.0, 1.0
                else:
                    try:
                        stat, p = wilcoxon(xgb, lr, alternative="two-sided")
                    except ValueError:
                        stat, p = 0.0, 1.0

                d = cohens_d(xgb, lr)
                results.append({
                    "test": "xgb_vs_lr_spearman",
                    "dataset": dataset, "perturbation": pert, "level": level,
                    "metric": "spearman_rho",
                    "group1_mean": float(np.mean(xgb)),
                    "group2_mean": float(np.mean(lr)),
                    "statistic": float(stat), "p_value": float(p),
                    "cohens_d": d,
                    "significant_005": p < 0.05,
                    "significant_001": p < 0.01,
                })
                sig = "***" if p < 0.01 else ("**" if p < 0.05 else "ns")
                print(f"  {dataset}/{pert}/{level}: XGB={np.mean(xgb):.3f} vs LR={np.mean(lr):.3f} "
                      f"| p={p:.4f} {sig} | d={d:.2f}")

    # ============================================================
    # TEST 2: Label Flip vs Gaussian/Dropout (Mann-Whitney U)
    # ============================================================
    print("\n--- Test 2: Label Flip vs Other Perturbations (Mann-Whitney U) ---")
    for dataset in df["dataset"].unique():
        for model in df["model"].unique():
            lf = df[
                (df["dataset"] == dataset) & (df["model"] == model) &
                (df["perturbation"] == "label_flip")
            ]["spearman_rho"].values
            others = df[
                (df["dataset"] == dataset) & (df["model"] == model) &
                (df["perturbation"].isin(["gaussian", "feature_dropout"]))
            ]["spearman_rho"].values

            if len(lf) < 3 or len(others) < 3:
                continue

            stat, p = mannwhitneyu(lf, others, alternative="two-sided")
            d = cohens_d(others, lf)

            results.append({
                "test": "label_flip_vs_others",
                "dataset": dataset, "perturbation": "label_flip_vs_gaussian+dropout",
                "level": "all", "metric": "spearman_rho",
                "group1_mean": float(np.mean(lf)),
                "group2_mean": float(np.mean(others)),
                "statistic": float(stat), "p_value": float(p),
                "cohens_d": d,
                "significant_005": p < 0.05,
                "significant_001": p < 0.01,
            })
            m_label = "XGB" if model == "xgboost" else "LR"
            sig = "***" if p < 0.01 else ("**" if p < 0.05 else "ns")
            print(f"  {dataset}/{m_label}: LabelFlip={np.mean(lf):.3f} vs Others={np.mean(others):.3f} "
                  f"| p={p:.6f} {sig} | d={d:.2f}")

    # ============================================================
    # TEST 3: Bootstrap CIs for all key metrics
    # ============================================================
    print("\n--- Test 3: Bootstrap 95% Confidence Intervals ---")
    ci_records = []
    for dataset in df["dataset"].unique():
        for model in df["model"].unique():
            for pert in df["perturbation"].unique():
                if pert == "none":
                    continue
                for level in df["level"].unique():
                    if level == 0.0:
                        continue
                    mask = (
                        (df["dataset"] == dataset) & (df["model"] == model) &
                        (df["perturbation"] == pert) & (df["level"] == level)
                    )
                    sub = df[mask]
                    if len(sub) < 3:
                        continue

                    for metric in ["spearman_rho", "lipschitz_mean", "jaccard_top3", "jaccard_top5"]:
                        vals = sub[metric].values
                        lower, upper, mean = bootstrap_ci(vals)
                        ci_records.append({
                            "dataset": dataset, "model": model,
                            "perturbation": pert, "level": level,
                            "metric": metric,
                            "mean": mean, "ci_lower": lower, "ci_upper": upper,
                            "ci_width": upper - lower,
                        })

    df_ci = pd.DataFrame(ci_records)
    ci_path = RESULTS_DIR / "confidence_intervals.csv"
    df_ci.to_csv(ci_path, index=False)
    print(f"  CIs saved to {ci_path} ({len(df_ci)} rows)")

    # ============================================================
    # TEST 4: Faithfulness significance — is fidelity different under perturbation?
    # ============================================================
    print("\n--- Test 4: Faithfulness Degradation Under Perturbation ---")
    for dataset in df_faith["dataset"].unique():
        for model in df_faith["model"].unique():
            baseline = df_faith[
                (df_faith["dataset"] == dataset) & (df_faith["model"] == model) &
                (df_faith["perturbation"] == "none")
            ]["surrogate_fidelity"].values

            for pert in ["gaussian", "label_flip", "feature_dropout"]:
                perturbed = df_faith[
                    (df_faith["dataset"] == dataset) & (df_faith["model"] == model) &
                    (df_faith["perturbation"] == pert)
                ]["surrogate_fidelity"].values

                if len(baseline) < 3 or len(perturbed) < 3:
                    continue

                stat, p = mannwhitneyu(baseline, perturbed, alternative="two-sided")
                d = cohens_d(baseline, perturbed)
                m_label = "XGB" if model == "xgboost" else "LR"
                sig = "***" if p < 0.01 else ("**" if p < 0.05 else "ns")
                print(f"  {dataset}/{m_label}/{pert}: baseline={np.mean(baseline):.3f} "
                      f"vs perturbed={np.mean(perturbed):.3f} | p={p:.4f} {sig} | d={d:.2f}")

                results.append({
                    "test": "faithfulness_degradation",
                    "dataset": dataset, "perturbation": pert,
                    "level": "all", "metric": "surrogate_fidelity",
                    "group1_mean": float(np.mean(baseline)),
                    "group2_mean": float(np.mean(perturbed)),
                    "statistic": float(stat), "p_value": float(p),
                    "cohens_d": d,
                    "significant_005": p < 0.05,
                    "significant_001": p < 0.01,
                })

    df_tests = pd.DataFrame(results)
    tests_path = RESULTS_DIR / "statistical_tests.csv"
    df_tests.to_csv(tests_path, index=False)
    print(f"\nAll test results saved to {tests_path} ({len(df_tests)} rows)")

    n_sig = df_tests["significant_005"].sum()
    n_total = len(df_tests)
    print(f"\nSignificant at α=0.05: {n_sig}/{n_total} ({100*n_sig/n_total:.0f}%)")
    n_sig_01 = df_tests["significant_001"].sum()
    print(f"Significant at α=0.01: {n_sig_01}/{n_total} ({100*n_sig_01/n_total:.0f}%)")

    return df_tests, df_ci


if __name__ == "__main__":
    run_statistical_tests()
