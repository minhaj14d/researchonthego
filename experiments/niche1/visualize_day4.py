"""
Day 4: Comparison visualizations — Interpretable vs Black-Box models.

Generates:
1. Grouped bar chart: stability (Spearman ρ) by model under label flip
2. Accuracy vs Stability scatter (proto-Pareto for Day 5)
3. Radar chart: multi-metric comparison per model family
4. Native vs SHAP importance stability comparison
"""
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from config import PLOTS_DIR, RESULTS_DIR

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
TYPE_COLORS = {"black_box": "#D32F2F", "interpretable": "#388E3C"}


def load():
    return pd.read_csv(RESULTS_DIR / "day4_all_models.csv")


def plot_stability_by_model(df):
    """Grouped bar: mean Spearman ρ under label flip, per model per dataset."""
    lf = df[df["perturbation"] == "label_flip"].copy()
    summary = lf.groupby(["dataset", "model"])["spearman_rho"].agg(["mean", "std"]).reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    models = ["xgboost", "logistic_regression", "decision_tree", "ebm"]
    x = np.arange(3)  # 3 perturbation levels... actually group by model
    bar_width = 0.18

    for idx, dataset in enumerate(["heart", "diabetes", "breast_cancer"]):
        ax = axes[idx]
        sub = summary[summary["dataset"] == dataset]

        for j, model in enumerate(models):
            row = sub[sub["model"] == model]
            if row.empty:
                continue
            ax.bar(
                j, row["mean"].values[0], bar_width * 3,
                yerr=row["std"].values[0], capsize=4,
                color=MODEL_COLORS[model], alpha=0.85,
                label=MODEL_LABELS[model] if idx == 0 else None,
            )

        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([MODEL_LABELS[m] for m in models], rotation=20, fontsize=9)
        ax.set_title(DATASET_LABELS[dataset], fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.axhline(y=0.8, color="gray", linestyle=":", alpha=0.5)
        ax.grid(True, alpha=0.2, axis="y")
        if idx == 0:
            ax.set_ylabel("Mean Spearman ρ (Label Flip)")

    fig.suptitle("Explanation Stability Under Label Flip: 4 Model Families", fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "model_comparison_bars.png", bbox_inches="tight")
    plt.close()
    print("  Saved: model_comparison_bars.png")


def plot_accuracy_vs_stability(df):
    """Scatter: accuracy vs Spearman ρ under label flip — proto-Pareto plot."""
    lf = df[df["perturbation"] == "label_flip"].copy()
    agg = lf.groupby(["dataset", "model", "model_type"]).agg(
        accuracy=("accuracy", "mean"),
        spearman_rho=("spearman_rho", "mean"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(9, 7))
    markers = {"heart": "o", "diabetes": "s", "breast_cancer": "D"}

    for _, row in agg.iterrows():
        ax.scatter(
            row["accuracy"], row["spearman_rho"],
            c=MODEL_COLORS[row["model"]], marker=markers[row["dataset"]],
            s=120, alpha=0.85, edgecolors="white", linewidths=0.5,
        )

    for model in MODEL_LABELS:
        ax.scatter([], [], c=MODEL_COLORS[model], s=80, label=MODEL_LABELS[model])
    for dataset in DATASET_LABELS:
        ax.scatter([], [], c="gray", marker=markers[dataset], s=80, label=DATASET_LABELS[dataset])

    ax.set_xlabel("Accuracy →", fontweight="bold")
    ax.set_ylabel("Spearman ρ (Explanation Stability) →", fontweight="bold")
    ax.set_title("Accuracy vs Explanation Stability Under Label Flip\n(Proto-Pareto: Upper-Right is Best)",
                 fontweight="bold")
    ax.axhline(y=0.8, color="gray", linestyle=":", alpha=0.4)
    ax.legend(loc="lower left", fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.4, 1.0)
    ax.set_ylim(0.0, 1.05)

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "accuracy_vs_stability.png")
    plt.close()
    print("  Saved: accuracy_vs_stability.png")


def plot_radar_chart(df):
    """Radar chart: multi-metric comparison across 4 models (averaged across datasets)."""
    bl = df[df["perturbation"] == "none"]
    lf = df[df["perturbation"] == "label_flip"]

    metrics = {}
    for model in ["xgboost", "logistic_regression", "decision_tree", "ebm"]:
        bl_m = bl[bl["model"] == model]
        lf_m = lf[lf["model"] == model]
        metrics[model] = {
            "Accuracy": bl_m["accuracy"].mean(),
            "AUC-ROC": bl_m["auc_roc"].mean(),
            "SHAP Stability\n(Label Flip ρ)": lf_m["spearman_rho"].mean(),
            "Surrogate\nFidelity": bl_m["surrogate_fidelity"].mean(),
            "Top-3\nJaccard\n(Label Flip)": lf_m["jaccard_top3"].mean(),
        }

    categories = list(next(iter(metrics.values())).keys())
    n = len(categories)
    angles = [i / float(n) * 2 * np.pi for i in range(n)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for model, vals in metrics.items():
        values = [max(0, v) for v in vals.values()]
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=MODEL_LABELS[model],
                color=MODEL_COLORS[model])
        ax.fill(angles, values, alpha=0.08, color=MODEL_COLORS[model])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title("Multi-Metric Model Comparison\n(Averaged Across All Datasets)", fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "radar_comparison.png", bbox_inches="tight")
    plt.close()
    print("  Saved: radar_comparison.png")


def plot_native_vs_shap(df):
    """Compare native feature importance stability vs SHAP stability under label flip."""
    lf = df[(df["perturbation"] == "label_flip") & df["native_spearman"].notna()].copy()

    agg = lf.groupby(["dataset", "model"]).agg(
        shap_rho=("spearman_rho", "mean"),
        native_rho=("native_spearman", "mean"),
    ).reset_index()

    SHORT_NAMES = {"heart": "Heart", "diabetes": "Diabetes", "breast_cancer": "Breast"}
    MARKER_SHAPES = {"heart": "o", "diabetes": "s", "breast_cancer": "D"}

    fig, ax = plt.subplots(figsize=(9, 7))

    for _, row in agg.iterrows():
        ax.scatter(
            row["shap_rho"], row["native_rho"],
            c=MODEL_COLORS[row["model"]],
            marker=MARKER_SHAPES[row["dataset"]],
            s=140, alpha=0.85, edgecolors="white", linewidths=0.8,
            zorder=3,
        )

    from adjustText import adjust_text
    texts = []
    xs, ys = [], []
    for _, row in agg.iterrows():
        xs.append(row["shap_rho"])
        ys.append(row["native_rho"])
        lbl = f"{MODEL_LABELS[row['model']]}\n({SHORT_NAMES[row['dataset']]})"
        t = ax.text(
            row["shap_rho"], row["native_rho"], lbl,
            fontsize=7, va="center", ha="left",
        )
        texts.append(t)
    try:
        adjust_text(
            texts, x=xs, y=ys, ax=ax,
            force_text=(1.5, 1.5), force_points=(2.0, 2.0),
            expand=(1.8, 1.8),
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.5, alpha=0.6),
        )
    except Exception:
        pass

    for model in MODEL_LABELS:
        ax.scatter([], [], c=MODEL_COLORS[model], s=80, label=MODEL_LABELS[model])
    for ds, mk in MARKER_SHAPES.items():
        ax.scatter([], [], c="gray", marker=mk, s=80, label=SHORT_NAMES[ds])

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="SHAP = Native")
    ax.set_xlabel("SHAP-Based Spearman ρ →", fontweight="bold")
    ax.set_ylabel("Native Importance Spearman ρ →", fontweight="bold")
    ax.set_title("SHAP vs Native Feature Importance Stability\n(Under Label Flip)", fontweight="bold")
    ax.legend(fontsize=9, loc="lower right", ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "native_vs_shap.png")
    plt.close()
    print("  Saved: native_vs_shap.png")


def plot_interpretable_vs_blackbox_heatmap(df):
    """Side-by-side heatmaps: black-box vs interpretable stability across all conditions."""
    sub = df[df["perturbation"] != "none"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for idx, (mtype, title) in enumerate([("black_box", "Black-Box Models"), ("interpretable", "Interpretable Models")]):
        ax = axes[idx]
        t = sub[sub["model_type"] == mtype].copy()
        t["config"] = t["dataset"].map(DATASET_LABELS) + "\n" + t["model"].map(MODEL_LABELS)

        PERT_LABELS = {"gaussian": "Gauss", "label_flip": "Label Flip", "feature_dropout": "Feat Drop"}
        t["pert_config"] = t["perturbation"].map(PERT_LABELS) + " (" + t["level"].astype(str) + ")"

        pivot = t.pivot_table(values="spearman_rho", index="config", columns="pert_config", aggfunc="mean")
        col_order = []
        for p in ["Gauss", "Feat Drop", "Label Flip"]:
            for l in ["0.05", "0.1", "0.15"]:
                cn = f"{p} ({l})"
                if cn in pivot.columns:
                    col_order.append(cn)
        pivot = pivot[[c for c in col_order if c in pivot.columns]]

        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1,
                    linewidths=0.5, ax=ax, cbar_kws={"label": "Spearman ρ"})
        ax.set_title(f"{title}\nSpearman ρ Stability", fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("" if idx == 1 else "")

    fig.suptitle("Black-Box vs Interpretable Models: Explanation Stability Matrix", fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "blackbox_vs_interpretable_heatmap.png", bbox_inches="tight")
    plt.close()
    print("  Saved: blackbox_vs_interpretable_heatmap.png")


def main():
    print("=" * 70)
    print("DAY 4: Generating Comparison Plots")
    print("=" * 70)

    df = load()

    print("\n[1/5] Stability by model (bar chart)...")
    plot_stability_by_model(df)

    print("[2/5] Accuracy vs stability (proto-Pareto)...")
    plot_accuracy_vs_stability(df)

    print("[3/5] Radar chart...")
    plot_radar_chart(df)

    print("[4/5] Native vs SHAP stability...")
    plot_native_vs_shap(df)

    print("[5/5] Black-box vs interpretable heatmaps...")
    plot_interpretable_vs_blackbox_heatmap(df)

    print(f"\nAll Day 4 plots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
