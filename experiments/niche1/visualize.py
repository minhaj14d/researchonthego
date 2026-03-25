"""
Day 2: Publication-quality visualizations for explanation stability results.

Generates:
1. Stability degradation curves (Spearman ρ vs perturbation level)
2. Heatmap: stability across datasets × models × perturbation types
3. Per-feature SHAP shift bar charts
4. Lipschitz ratio comparison (input-space vs model-space)
5. Top-k Jaccard degradation
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

from config import PLOTS_DIR, RESULTS_DIR

matplotlib.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
})

DATASET_LABELS = {
    "heart": "Heart Disease",
    "diabetes": "Pima Diabetes",
    "breast_cancer": "Breast Cancer",
}
MODEL_LABELS = {
    "xgboost": "XGBoost",
    "logistic_regression": "Logistic Reg.",
}
PERT_LABELS = {
    "gaussian": "Gaussian Noise",
    "label_flip": "Label Flip",
    "feature_dropout": "Feature Dropout",
}
PERT_COLORS = {
    "gaussian": "#2196F3",
    "label_flip": "#F44336",
    "feature_dropout": "#FF9800",
}
MODEL_STYLES = {
    "xgboost": "-",
    "logistic_regression": "--",
}


def load_results():
    df = pd.read_csv(RESULTS_DIR / "stability_results.csv")
    return df


def load_feature_results():
    path = RESULTS_DIR / "per_feature_stability.csv"
    return pd.read_csv(path)


def load_model_lipschitz():
    path = RESULTS_DIR / "model_lipschitz.csv"
    return pd.read_csv(path)


def plot_degradation_curves(df):
    """Spearman ρ vs perturbation level, one subplot per dataset."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

    for idx, dataset in enumerate(["heart", "diabetes", "breast_cancer"]):
        ax = axes[idx]
        subset = df[(df["dataset"] == dataset) & (df["perturbation"] != "none")]

        for model_name in ["xgboost", "logistic_regression"]:
            for pert_type in ["gaussian", "label_flip", "feature_dropout"]:
                mask = (subset["model"] == model_name) & (subset["perturbation"] == pert_type)
                group = subset[mask].groupby("level")["spearman_rho"].agg(["mean", "std"])
                if group.empty:
                    continue

                label = f"{MODEL_LABELS[model_name]} + {PERT_LABELS[pert_type]}"
                ax.errorbar(
                    group.index, group["mean"], yerr=group["std"],
                    label=label,
                    color=PERT_COLORS[pert_type],
                    linestyle=MODEL_STYLES[model_name],
                    marker="o", markersize=4, capsize=3, linewidth=1.5,
                )

        ax.set_title(DATASET_LABELS[dataset], fontweight="bold")
        ax.set_xlabel("Perturbation Level")
        ax.set_ylim(0, 1.05)
        ax.axhline(y=0.8, color="gray", linestyle=":", alpha=0.5, label="ρ=0.8 threshold")
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.set_ylabel("Spearman ρ (Rank Stability)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.12))
    fig.suptitle("Explanation Stability Degradation Under Perturbation", fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "degradation_curves.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved: degradation_curves.png")


def plot_stability_heatmap(df):
    """Heatmap: mean Spearman ρ for each (dataset×model) × perturbation_level combination."""
    subset = df[df["perturbation"] != "none"].copy()
    subset["config"] = subset["dataset"].map(DATASET_LABELS) + "\n" + subset["model"].map(MODEL_LABELS)
    subset["pert_config"] = subset["perturbation"].map(PERT_LABELS) + "\n(" + subset["level"].astype(str) + ")"

    pivot = subset.pivot_table(
        values="spearman_rho", index="config", columns="pert_config", aggfunc="mean"
    )
    col_order = []
    for p in ["Gaussian Noise", "Feature Dropout", "Label Flip"]:
        for l in ["0.05", "0.1", "0.15"]:
            col_name = f"{p}\n({l})"
            if col_name in pivot.columns:
                col_order.append(col_name)
    pivot = pivot[col_order]

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(
        pivot, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1,
        linewidths=0.5, ax=ax, cbar_kws={"label": "Spearman ρ"}
    )
    ax.set_title("Explanation Stability Matrix: Spearman ρ Across All Conditions", fontweight="bold")
    ax.set_xlabel("Perturbation Type (Level)")
    ax.set_ylabel("")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "stability_heatmap.png")
    plt.close()
    print(f"  Saved: stability_heatmap.png")


def plot_lipschitz_comparison(df, df_ml):
    """Side-by-side: input-space Lipschitz (gaussian/dropout) vs model-space (label_flip)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Input-space Lipschitz
    ax = axes[0]
    input_lip = df[df["perturbation"].isin(["gaussian", "feature_dropout"])].copy()
    input_lip["config"] = input_lip["dataset"].map(DATASET_LABELS) + " | " + input_lip["model"].map(MODEL_LABELS)

    for pert in ["gaussian", "feature_dropout"]:
        sub = input_lip[input_lip["perturbation"] == pert]
        group = sub.groupby(["config", "level"])["lipschitz_mean"].mean().reset_index()
        for cfg in group["config"].unique():
            cfg_data = group[group["config"] == cfg]
            ax.plot(
                cfg_data["level"], cfg_data["lipschitz_mean"],
                marker="o", markersize=4, label=f"{cfg} ({PERT_LABELS[pert]})"
            )

    ax.set_title("Input-Space Lipschitz Ratio\n(Gaussian & Feature Dropout)", fontweight="bold")
    ax.set_xlabel("Perturbation Level")
    ax.set_ylabel("||ΔSHAP||₂ / ||ΔX||₂")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="upper right")

    # Model-space Lipschitz
    ax = axes[1]
    if not df_ml.empty:
        df_ml_plot = df_ml.copy()
        df_ml_plot["config"] = df_ml_plot["dataset"].map(DATASET_LABELS) + " | " + df_ml_plot["model"].map(MODEL_LABELS)
        for cfg in df_ml_plot["config"].unique():
            cfg_data = df_ml_plot[df_ml_plot["config"] == cfg].groupby("level")["model_lipschitz"].mean().reset_index()
            ax.plot(cfg_data["level"], cfg_data["model_lipschitz"], marker="s", markersize=5, label=cfg)
        ax.legend(fontsize=8)

    ax.set_title("Model-Space Lipschitz Ratio\n(Label Flip — ||ΔSHAP||₂ / ||Δθ||₂)", fontweight="bold")
    ax.set_xlabel("Perturbation Level (label flip rate)")
    ax.set_ylabel("||ΔSHAP||₂ / ||Δθ||₂")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "lipschitz_comparison.png")
    plt.close()
    print(f"  Saved: lipschitz_comparison.png")


def plot_per_feature_bars(df_feat):
    """Top-10 most/least stable features under label_flip@15%, per dataset."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    for col_idx, dataset in enumerate(["heart", "diabetes", "breast_cancer"]):
        mask = (
            (df_feat["dataset"] == dataset) &
            (df_feat["perturbation"] == "label_flip") &
            (df_feat["level"] == 0.15)
        )
        sub = df_feat[mask].groupby("feature").agg(
            mean_shift=("shap_shift", "mean"),
            mean_corr=("feature_correlation", "mean"),
        ).reset_index().sort_values("mean_shift", ascending=False)

        n_show = min(10, len(sub))
        top = sub.head(n_show)
        bottom = sub.tail(n_show)

        # Most unstable
        ax = axes[0, col_idx]
        colors = plt.cm.Reds(np.linspace(0.3, 0.9, n_show))
        ax.barh(range(n_show), top["mean_shift"].values, color=colors)
        ax.set_yticks(range(n_show))
        ax.set_yticklabels(top["feature"].values, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Mean |ΔSHAP|")
        ax.set_title(f"{DATASET_LABELS[dataset]}\nMost Unstable Features", fontweight="bold")
        ax.grid(True, alpha=0.2, axis="x")

        # Most stable
        ax = axes[1, col_idx]
        colors = plt.cm.Greens(np.linspace(0.3, 0.9, n_show))
        ax.barh(range(n_show), bottom["mean_shift"].values, color=colors)
        ax.set_yticks(range(n_show))
        ax.set_yticklabels(bottom["feature"].values, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Mean |ΔSHAP|")
        ax.set_title(f"{DATASET_LABELS[dataset]}\nMost Stable Features", fontweight="bold")
        ax.grid(True, alpha=0.2, axis="x")

    fig.suptitle("Per-Feature Explanation Stability Under Label Flip (15%)", fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "per_feature_stability.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved: per_feature_stability.png")


def plot_feature_correlation_heatmap(df_feat):
    """Heatmap of per-feature Pearson correlation (SHAP stability) for each dataset under label_flip."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, dataset in enumerate(["heart", "diabetes", "breast_cancer"]):
        ax = axes[idx]
        mask = (
            (df_feat["dataset"] == dataset) &
            (df_feat["perturbation"] == "label_flip") &
            (df_feat["model"] == "xgboost")
        )
        sub = df_feat[mask].pivot_table(
            values="feature_correlation", index="feature", columns="level", aggfunc="mean"
        )
        sub = sub.sort_values(0.15, ascending=False)

        n_features = len(sub)
        height_ratio = max(4, n_features * 0.25)
        sns.heatmap(
            sub, annot=n_features <= 15, fmt=".2f",
            cmap="RdYlGn", vmin=0, vmax=1, linewidths=0.3, ax=ax,
            cbar_kws={"label": "Pearson r"},
        )
        ax.set_title(f"{DATASET_LABELS[dataset]} (XGBoost)", fontweight="bold")
        ax.set_xlabel("Label Flip Rate")
        ax.set_ylabel("Feature" if idx == 0 else "")

    fig.suptitle("Per-Feature SHAP Correlation Stability Under Label Flip", fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "feature_correlation_heatmap.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved: feature_correlation_heatmap.png")


def plot_topk_jaccard(df):
    """Top-k Jaccard similarity degradation for k=3 and k=5."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for k_idx, k in enumerate([3, 5]):
        ax = axes[k_idx]
        col = f"jaccard_top{k}"

        for dataset in ["heart", "diabetes", "breast_cancer"]:
            for model in ["xgboost", "logistic_regression"]:
                mask = (
                    (df["dataset"] == dataset) &
                    (df["model"] == model) &
                    (df["perturbation"] == "label_flip")
                )
                sub = df[mask].groupby("level")[col].mean()
                label = f"{DATASET_LABELS[dataset]} | {MODEL_LABELS[model]}"
                ax.plot(sub.index, sub.values, marker="o", markersize=5, label=label)

        ax.set_title(f"Top-{k} Jaccard Similarity\n(Label Flip)", fontweight="bold")
        ax.set_xlabel("Label Flip Rate")
        ax.set_ylabel(f"Jaccard@{k}")
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "topk_jaccard.png")
    plt.close()
    print(f"  Saved: topk_jaccard.png")


def plot_rank_vs_magnitude(df):
    """Scatter: Spearman ρ (rank stability) vs Lipschitz ratio (magnitude sensitivity).
    Key visual for the 'rank ≠ magnitude' finding."""
    mask = df["perturbation"].isin(["gaussian", "feature_dropout"])
    sub = df[mask].copy()

    fig, ax = plt.subplots(figsize=(8, 6))

    for model in ["xgboost", "logistic_regression"]:
        for dataset in ["heart", "diabetes", "breast_cancer"]:
            m = (sub["model"] == model) & (sub["dataset"] == dataset)
            data = sub[m]
            marker = "o" if model == "xgboost" else "^"
            ax.scatter(
                data["spearman_rho"], data["lipschitz_mean"],
                label=f"{MODEL_LABELS[model]} | {DATASET_LABELS[dataset]}",
                marker=marker, alpha=0.7, s=50
            )

    ax.set_xlabel("Spearman ρ (Rank Stability) →", fontweight="bold")
    ax.set_ylabel("Lipschitz Ratio (Magnitude Sensitivity) →", fontweight="bold")
    ax.set_title("Rank Stability ≠ Magnitude Stability", fontweight="bold")
    ax.axvline(x=0.8, color="gray", linestyle=":", alpha=0.5)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "rank_vs_magnitude.png")
    plt.close()
    print(f"  Saved: rank_vs_magnitude.png")


def main():
    print("=" * 70)
    print("DAY 2: Generating Visualizations")
    print("=" * 70)

    df = load_results()
    df_feat = load_feature_results()
    df_ml = load_model_lipschitz()

    print("\n[1/6] Degradation curves...")
    plot_degradation_curves(df)

    print("[2/6] Stability heatmap...")
    plot_stability_heatmap(df)

    print("[3/6] Lipschitz comparison...")
    plot_lipschitz_comparison(df, df_ml)

    print("[4/6] Per-feature stability bars...")
    plot_per_feature_bars(df_feat)

    print("[5/6] Feature correlation heatmap...")
    plot_feature_correlation_heatmap(df_feat)

    print("[6/6] Top-k Jaccard + Rank vs Magnitude...")
    plot_topk_jaccard(df)
    plot_rank_vs_magnitude(df)

    print(f"\nAll plots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
