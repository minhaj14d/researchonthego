"""
Day 5: Final Analysis — Pareto frontiers, consolidated tables, summary statistics.

Produces all the definitive numbers for the paper.
"""
import time
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import ConvexHull

from config import RESULTS_DIR, PLOTS_DIR

matplotlib.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "figure.facecolor": "white",
})

DATASET_LABELS = {"heart": "Heart Disease", "diabetes": "Pima Diabetes", "breast_cancer": "Breast Cancer"}
MODEL_LABELS = {
    "xgboost": "XGBoost", "logistic_regression": "Logistic Reg.",
    "decision_tree": "Decision Tree", "ebm": "EBM",
}
MODEL_COLORS = {
    "xgboost": "#1565C0", "logistic_regression": "#E65100",
    "decision_tree": "#2E7D32", "ebm": "#6A1B9A",
}


def is_pareto_optimal(costs):
    """Find Pareto-optimal points (maximize both objectives)."""
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] >= c, axis=1)
            is_efficient[i] = True
    return is_efficient


def generate_pareto_plot(df):
    """Formal Pareto frontier: accuracy vs stability for each dataset."""
    lf = df[df["perturbation"] == "label_flip"].copy()
    agg = lf.groupby(["dataset", "model", "model_type"]).agg(
        accuracy=("accuracy", "mean"),
        spearman_rho=("spearman_rho", "mean"),
        fidelity=("surrogate_fidelity", "mean"),
    ).reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    for idx, dataset in enumerate(["heart", "diabetes", "breast_cancer"]):
        ax = axes[idx]
        sub = agg[agg["dataset"] == dataset]
        points = sub[["accuracy", "spearman_rho"]].values
        pareto_mask = is_pareto_optimal(points)

        for _, row in sub.iterrows():
            ax.scatter(
                row["accuracy"], row["spearman_rho"],
                c=MODEL_COLORS[row["model"]], s=180, alpha=0.9,
                edgecolors="black", linewidths=0.8, zorder=5,
            )
            offset = (8, 6)
            ax.annotate(
                MODEL_LABELS[row["model"]],
                (row["accuracy"], row["spearman_rho"]),
                textcoords="offset points", xytext=offset, fontsize=9,
                fontweight="bold",
            )

        pareto_pts = points[pareto_mask]
        if len(pareto_pts) > 1:
            sorted_idx = np.argsort(pareto_pts[:, 0])
            pareto_sorted = pareto_pts[sorted_idx]
            ax.plot(pareto_sorted[:, 0], pareto_sorted[:, 1],
                    "k--", alpha=0.5, linewidth=1.5, label="Pareto frontier")

        ax.fill_between(
            [sub["accuracy"].min() - 0.05, sub["accuracy"].max() + 0.05],
            0.8, 1.05, alpha=0.06, color="green", label="ρ ≥ 0.8 (stable)"
        )

        ax.set_title(DATASET_LABELS[dataset], fontweight="bold", fontsize=14)
        ax.set_xlabel("Accuracy →")
        ax.set_ylim(0.1, 1.05)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.set_ylabel("Spearman ρ (Explanation Stability) →")
        if idx == 2:
            ax.legend(fontsize=9, loc="lower left")

    fig.suptitle(
        "Accuracy–Stability Pareto Frontier Under Label Corruption",
        fontweight="bold", fontsize=15, y=1.03
    )
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "pareto_frontier.png", bbox_inches="tight")
    plt.close()
    print("  Saved: pareto_frontier.png")


def generate_summary_table(df):
    """Publication-ready Table 1: all models × datasets × key metrics."""
    bl = df[df["perturbation"] == "none"].copy()
    lf = df[df["perturbation"] == "label_flip"].copy()

    bl_agg = bl.groupby(["dataset", "model"]).agg(
        accuracy=("accuracy", "mean"),
        f1=("f1", "mean"),
        auc_roc=("auc_roc", "mean"),
        fidelity_clean=("surrogate_fidelity", "mean"),
    ).reset_index()

    lf_agg = lf.groupby(["dataset", "model"]).agg(
        rho_lf=("spearman_rho", "mean"),
        rho_lf_std=("spearman_rho", "std"),
        j3_lf=("jaccard_top3", "mean"),
        j5_lf=("jaccard_top5", "mean"),
        fidelity_lf=("surrogate_fidelity", "mean"),
    ).reset_index()

    table = bl_agg.merge(lf_agg, on=["dataset", "model"])
    table["rho_display"] = table.apply(
        lambda r: f"{r['rho_lf']:.3f} ± {r['rho_lf_std']:.3f}", axis=1
    )
    table["dataset_label"] = table["dataset"].map(DATASET_LABELS)
    table["model_label"] = table["model"].map(MODEL_LABELS)

    cols = ["dataset_label", "model_label", "accuracy", "auc_roc",
            "fidelity_clean", "rho_display", "j3_lf", "fidelity_lf"]
    display = table[cols].copy()
    display.columns = ["Dataset", "Model", "Accuracy", "AUC-ROC",
                       "Fidelity (clean)", "ρ (label flip)", "J@3 (label flip)", "Fidelity (label flip)"]

    for c in ["Accuracy", "AUC-ROC", "Fidelity (clean)", "J@3 (label flip)", "Fidelity (label flip)"]:
        display[c] = display[c].map(lambda x: f"{x:.3f}")

    csv_path = RESULTS_DIR / "table1_summary.csv"
    display.to_csv(csv_path, index=False)
    print(f"  Table 1 saved to {csv_path}")
    print(display.to_string(index=False))
    return display


def generate_perturbation_comparison_table(df):
    """Table 2: Spearman ρ across all perturbation types for all models."""
    sub = df[df["perturbation"] != "none"].copy()
    table = sub.groupby(["dataset", "model", "perturbation"]).agg(
        rho_mean=("spearman_rho", "mean"),
        rho_std=("spearman_rho", "std"),
    ).reset_index()

    table["rho_display"] = table.apply(
        lambda r: f"{r['rho_mean']:.3f} ± {r['rho_std']:.3f}", axis=1
    )
    pivot = table.pivot_table(
        values="rho_display", index=["dataset", "model"],
        columns="perturbation", aggfunc="first"
    )
    pivot = pivot[["gaussian", "feature_dropout", "label_flip"]]
    pivot.columns = ["Gaussian Noise", "Feature Dropout", "Label Flip"]

    csv_path = RESULTS_DIR / "table2_perturbation.csv"
    pivot.to_csv(csv_path)
    print(f"  Table 2 saved to {csv_path}")
    print(pivot.to_string())
    return pivot


def generate_final_summary_plot(df):
    """The definitive single figure: 3×3 panel (datasets × metrics) with all 4 models."""
    lf = df[df["perturbation"] == "label_flip"].copy()
    metrics = ["spearman_rho", "jaccard_top3", "surrogate_fidelity"]
    metric_labels = ["Spearman ρ", "Top-3 Jaccard", "Surrogate Fidelity"]
    datasets = ["heart", "diabetes", "breast_cancer"]

    fig, axes = plt.subplots(3, 3, figsize=(16, 12), sharey="row")
    models = ["xgboost", "logistic_regression", "decision_tree", "ebm"]

    for row, (metric, mlabel) in enumerate(zip(metrics, metric_labels)):
        for col, dataset in enumerate(datasets):
            ax = axes[row, col]
            sub = lf[(lf["dataset"] == dataset)]

            for j, model in enumerate(models):
                m = sub[sub["model"] == model]
                group = m.groupby("level")[metric].agg(["mean", "std"])
                ax.errorbar(
                    group.index, group["mean"], yerr=group["std"],
                    marker="o", markersize=5, capsize=3, linewidth=2,
                    color=MODEL_COLORS[model],
                    label=MODEL_LABELS[model] if (row == 0 and col == 0) else None,
                )

            if row == 0:
                ax.set_title(DATASET_LABELS[dataset], fontweight="bold", fontsize=13)
            if col == 0:
                ax.set_ylabel(mlabel, fontweight="bold")
            if row == 2:
                ax.set_xlabel("Label Flip Rate")
            ax.grid(True, alpha=0.2)
            if metric in ("spearman_rho", "jaccard_top3"):
                ax.set_ylim(-0.1, 1.1)
                ax.axhline(y=0.8 if metric == "spearman_rho" else 0.5,
                           color="gray", linestyle=":", alpha=0.4)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02), fontsize=11)
    fig.suptitle("Explanation Stability Under Label Flip: All Models × All Metrics",
                 fontweight="bold", fontsize=15, y=1.01)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "final_summary_panel.png", bbox_inches="tight")
    plt.close()
    print("  Saved: final_summary_panel.png")


def compute_final_statistics(df):
    """Compute all the definitive numbers cited in the paper."""
    print("\n" + "=" * 70)
    print("DEFINITIVE STATISTICS FOR THE PAPER")
    print("=" * 70)

    lf = df[df["perturbation"] == "label_flip"]
    bl = df[df["perturbation"] == "none"]

    print("\n--- Finding 1: Label flip is catastrophically destructive ---")
    worst = lf.groupby(["dataset", "model"])["spearman_rho"].mean().min()
    worst_combo = lf.groupby(["dataset", "model"])["spearman_rho"].mean().idxmin()
    print(f"  Worst case: {worst_combo} → ρ = {worst:.3f}")

    gauss = df[df["perturbation"] == "gaussian"]
    best_gauss = gauss.groupby(["dataset", "model"])["spearman_rho"].mean().min()
    print(f"  Best-case Gaussian worst: ρ = {best_gauss:.3f}")
    print(f"  Gap: {best_gauss - worst:.3f} absolute")

    print("\n--- Finding 2: EBM vs LogReg on Breast Cancer ---")
    ebm_bc = lf[(lf["dataset"] == "breast_cancer") & (lf["model"] == "ebm")]["spearman_rho"].mean()
    lr_bc = lf[(lf["dataset"] == "breast_cancer") & (lf["model"] == "logistic_regression")]["spearman_rho"].mean()
    print(f"  EBM: ρ = {ebm_bc:.3f}")
    print(f"  LogReg: ρ = {lr_bc:.3f}")
    print(f"  Ratio: {ebm_bc/lr_bc:.1f}x more stable")

    print("\n--- Finding 3: Decision Tree instability ---")
    dt_all = lf[lf["model"] == "decision_tree"].groupby("dataset")["spearman_rho"].mean()
    print(f"  Heart: ρ = {dt_all.get('heart', 'N/A'):.3f}")
    print(f"  Diabetes: ρ = {dt_all.get('diabetes', 'N/A'):.3f}")
    print(f"  Breast Cancer: ρ = {dt_all.get('breast_cancer', 'N/A'):.3f}")

    print("\n--- Finding 4: Baseline performance ---")
    perf = bl.groupby(["dataset", "model"])["accuracy"].mean()
    print(perf.to_string())

    print("\n--- Finding 5: Fidelity collapse ---")
    fid = lf.groupby(["dataset", "model"])["surrogate_fidelity"].mean()
    worst_fid = fid.min()
    worst_fid_combo = fid.idxmin()
    print(f"  Worst fidelity under label flip: {worst_fid_combo} → {worst_fid:.3f}")

    print("\n--- Total experiment count ---")
    total_day1 = 300
    total_day2 = 4590 + 90
    total_day3 = 300 + 51 + 216
    total_day4 = 600
    total = total_day1 + total_day2 + total_day3 + total_day4
    print(f"  Day 1: {total_day1}")
    print(f"  Day 2: {total_day2}")
    print(f"  Day 3: {total_day3}")
    print(f"  Day 4: {total_day4}")
    print(f"  Total: {total:,}")


def main():
    print("=" * 70)
    print("DAY 5: Final Analysis & Paper Tables")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    df = pd.read_csv(RESULTS_DIR / "day4_all_models.csv")

    print("\n[1/5] Pareto frontier plot...")
    generate_pareto_plot(df)

    print("\n[2/5] Table 1: Summary results...")
    generate_summary_table(df)

    print("\n[3/5] Table 2: Perturbation comparison...")
    generate_perturbation_comparison_table(df)

    print("\n[4/5] Final summary panel (3×3)...")
    generate_final_summary_plot(df)

    print("\n[5/5] Computing definitive statistics...")
    compute_final_statistics(df)

    print(f"\n{'='*70}")
    print("DAY 5 ANALYSIS COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
