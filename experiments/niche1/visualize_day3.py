"""
Day 3: Visualization — Faithfulness plots + statistical significance forest plots.
"""
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from config import PLOTS_DIR, RESULTS_DIR

matplotlib.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
})

DATASET_LABELS = {"heart": "Heart Disease", "diabetes": "Pima Diabetes", "breast_cancer": "Breast Cancer"}
MODEL_LABELS = {"xgboost": "XGBoost", "logistic_regression": "Logistic Reg."}
PERT_LABELS = {"gaussian": "Gaussian Noise", "label_flip": "Label Flip", "feature_dropout": "Feature Dropout", "none": "No Perturbation"}
PERT_COLORS = {"gaussian": "#2196F3", "label_flip": "#F44336", "feature_dropout": "#FF9800", "none": "#4CAF50"}


def plot_faithfulness_degradation():
    """Surrogate fidelity across perturbation types and levels."""
    df = pd.read_csv(RESULTS_DIR / "faithfulness_results.csv")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for idx, dataset in enumerate(["heart", "diabetes", "breast_cancer"]):
        ax = axes[idx]
        sub = df[(df["dataset"] == dataset) & (df["perturbation"] != "none")]

        for model in ["xgboost", "logistic_regression"]:
            for pert in ["gaussian", "label_flip", "feature_dropout"]:
                mask = (sub["model"] == model) & (sub["perturbation"] == pert)
                group = sub[mask].groupby("level")["surrogate_fidelity"].agg(["mean", "std"])
                if group.empty:
                    continue
                ls = "-" if model == "xgboost" else "--"
                label = f"{MODEL_LABELS[model]} + {PERT_LABELS[pert]}"
                ax.errorbar(
                    group.index, group["mean"], yerr=group["std"],
                    label=label, color=PERT_COLORS[pert], linestyle=ls,
                    marker="o", markersize=4, capsize=3, linewidth=1.5,
                )

        # Baseline fidelity
        for model in ["xgboost", "logistic_regression"]:
            bl = df[(df["dataset"] == dataset) & (df["model"] == model) & (df["perturbation"] == "none")]
            if not bl.empty:
                ls = "-" if model == "xgboost" else "--"
                ax.axhline(
                    y=bl["surrogate_fidelity"].mean(), color="gray", linestyle=ls, alpha=0.4,
                    label=f"{MODEL_LABELS[model]} baseline" if idx == 0 else None
                )

        ax.set_title(DATASET_LABELS[dataset], fontweight="bold")
        ax.set_xlabel("Perturbation Level")
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.set_ylabel("Surrogate Fidelity")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.15), fontsize=9)
    fig.suptitle("SHAP Surrogate Fidelity Under Perturbation", fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "faithfulness_degradation.png", bbox_inches="tight")
    plt.close()
    print("  Saved: faithfulness_degradation.png")


def plot_faithfulness_ratio():
    """Faithfulness ratio (top-k mask drop / bottom-k mask drop) — measures whether SHAP correctly identifies important features."""
    df = pd.read_csv(RESULTS_DIR / "faithfulness_results.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for k_idx, k in enumerate([3, 5]):
        ax = axes[k_idx]
        col = f"faithfulness_ratio_k{k}"

        plot_data = []
        for dataset in ["heart", "diabetes", "breast_cancer"]:
            for model in ["xgboost", "logistic_regression"]:
                for pert in ["none", "gaussian", "label_flip", "feature_dropout"]:
                    sub = df[(df["dataset"] == dataset) & (df["model"] == model) & (df["perturbation"] == pert)]
                    for _, row in sub.iterrows():
                        plot_data.append({
                            "Dataset": DATASET_LABELS[dataset],
                            "Model": MODEL_LABELS[model],
                            "Perturbation": PERT_LABELS[pert],
                            "ratio": row[col],
                        })

        pdf = pd.DataFrame(plot_data)
        sns.boxplot(data=pdf, x="Perturbation", y="ratio", hue="Model", ax=ax,
                    palette={"XGBoost": "#2196F3", "Logistic Reg.": "#FF9800"})
        ax.axhline(y=1.0, color="red", linestyle=":", alpha=0.7, label="Ratio=1 (random)")
        ax.set_title(f"Faithfulness Ratio (k={k})\nTop-k Drop / Bottom-k Drop", fontweight="bold")
        ax.set_ylabel("Faithfulness Ratio" if k_idx == 0 else "")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=15)
        ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "faithfulness_ratio.png")
    plt.close()
    print("  Saved: faithfulness_ratio.png")


def plot_significance_forest():
    """Forest plot of Cohen's d effect sizes with significance markers."""
    df = pd.read_csv(RESULTS_DIR / "statistical_tests.csv")

    xgb_vs_lr = df[df["test"] == "xgb_vs_lr_spearman"].copy()
    if xgb_vs_lr.empty:
        print("  Skipped: significance_forest.png (no data)")
        return

    xgb_vs_lr["label"] = (
        xgb_vs_lr["dataset"].map(DATASET_LABELS) + "\n" +
        xgb_vs_lr["perturbation"].map(PERT_LABELS) + "\n" +
        "(" + xgb_vs_lr["level"].astype(str) + ")"
    )
    xgb_vs_lr = xgb_vs_lr.sort_values("cohens_d")

    fig, ax = plt.subplots(figsize=(10, max(6, len(xgb_vs_lr) * 0.35)))
    y_pos = range(len(xgb_vs_lr))
    colors = ["#F44336" if row["significant_001"] else "#FF9800" if row["significant_005"] else "#9E9E9E"
              for _, row in xgb_vs_lr.iterrows()]

    ax.barh(list(y_pos), xgb_vs_lr["cohens_d"].values, color=colors, height=0.7, alpha=0.8)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(xgb_vs_lr["label"].values, fontsize=8)
    ax.set_xlabel("Cohen's d (Effect Size)", fontweight="bold")
    ax.set_title("XGBoost vs Logistic Regression: Spearman ρ Differences\n(Red=p<0.01, Orange=p<0.05, Gray=n.s.)",
                 fontweight="bold")
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.axvline(x=0.8, color="gray", linestyle=":", alpha=0.5, label="|d|=0.8 (large)")
    ax.axvline(x=-0.8, color="gray", linestyle=":", alpha=0.5)
    ax.grid(True, alpha=0.2, axis="x")

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "significance_forest.png")
    plt.close()
    print("  Saved: significance_forest.png")


def plot_confidence_intervals():
    """CI width comparison — which conditions have the tightest/widest CIs."""
    df = pd.read_csv(RESULTS_DIR / "confidence_intervals.csv")
    spearman = df[df["metric"] == "spearman_rho"].copy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for idx, dataset in enumerate(["heart", "diabetes", "breast_cancer"]):
        ax = axes[idx]
        sub = spearman[spearman["dataset"] == dataset].copy()
        sub["label"] = (
            sub["model"].map(MODEL_LABELS) + "\n" +
            sub["perturbation"].map(PERT_LABELS) + "\n" +
            "(" + sub["level"].astype(str) + ")"
        )
        sub = sub.sort_values("mean")

        y = range(len(sub))
        ax.errorbar(
            sub["mean"], list(y),
            xerr=[sub["mean"] - sub["ci_lower"], sub["ci_upper"] - sub["mean"]],
            fmt="o", color="#2196F3", capsize=3, markersize=4,
        )
        ax.set_yticks(list(y))
        ax.set_yticklabels(sub["label"].values, fontsize=7)
        ax.set_xlabel("Spearman ρ")
        ax.set_title(DATASET_LABELS[dataset], fontweight="bold")
        ax.axvline(x=0.8, color="red", linestyle=":", alpha=0.5)
        ax.grid(True, alpha=0.2, axis="x")
        ax.set_xlim(-0.1, 1.1)

    fig.suptitle("95% Bootstrap Confidence Intervals for Spearman ρ", fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "confidence_intervals.png", bbox_inches="tight")
    plt.close()
    print("  Saved: confidence_intervals.png")


def main():
    print("=" * 70)
    print("DAY 3: Generating Plots")
    print("=" * 70)

    print("\n[1/4] Faithfulness degradation curves...")
    plot_faithfulness_degradation()

    print("[2/4] Faithfulness ratio boxplots...")
    plot_faithfulness_ratio()

    print("[3/4] Significance forest plot...")
    plot_significance_forest()

    print("[4/4] Confidence interval plot...")
    plot_confidence_intervals()

    print(f"\nAll Day 3 plots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
