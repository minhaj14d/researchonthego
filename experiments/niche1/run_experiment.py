"""
Main experiment runner for Niche 1: Explanation Stability Under Distributional Shift.

Day 1: Trains baselines, computes SHAP, runs perturbation experiments, logs results.
"""
import json
import time
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd

from config import (
    DATASETS, PERTURBATION_TYPES, PERTURBATION_LEVELS,
    TOP_K_VALUES, RANDOM_SEEDS, RESULTS_DIR
)
from data_loader import get_dataset
from train_baselines import (
    build_model, train_and_evaluate, compute_shap_values, get_feature_importance
)
from perturbations import apply_perturbation
from stability_metrics import compute_all_stability_metrics


MODEL_NAMES = ["xgboost", "logistic_regression"]


def run_single_experiment(dataset_name, model_name, seed):
    """Train a model, compute SHAP, run all perturbations, return results."""
    print(f"  [{dataset_name}] [{model_name}] seed={seed}")

    data = get_dataset(dataset_name, seed=seed)
    model = build_model(model_name, seed=seed)
    model, perf_metrics = train_and_evaluate(
        model, data["X_train"], data["y_train"], data["X_test"], data["y_test"]
    )

    shap_orig = compute_shap_values(
        model, data["X_train"], data["X_test"], model_name
    )
    imp_orig = get_feature_importance(shap_orig)

    results = []
    base_record = {
        "dataset": dataset_name,
        "model": model_name,
        "seed": seed,
        "n_features": len(data["feature_names"]),
        **perf_metrics,
    }

    no_pert_record = {
        **base_record,
        "perturbation": "none",
        "level": 0.0,
        "spearman_rho": 1.0,
        "spearman_p_value": 0.0,
        "lipschitz_mean": 0.0,
        "lipschitz_std": 0.0,
    }
    for k in TOP_K_VALUES:
        no_pert_record[f"jaccard_top{k}"] = 1.0
    no_pert_record["feature_importance"] = imp_orig.tolist()
    results.append(no_pert_record)

    for pert_type, level in product(PERTURBATION_TYPES, PERTURBATION_LEVELS):
        X_tr_p, y_tr_p, X_te_p, needs_retrain = apply_perturbation(
            data["X_train"], data["y_train"], data["X_test"],
            pert_type, level, seed
        )

        if needs_retrain:
            model_p = build_model(model_name, seed=seed)
            model_p.fit(X_tr_p, y_tr_p)
            shap_pert = compute_shap_values(model_p, X_tr_p, X_te_p, model_name)
        else:
            shap_pert = compute_shap_values(
                model, data["X_train"], X_te_p, model_name
            )

        X_ref = data["X_test"] if pert_type != "label_flip" else data["X_test"]
        stability = compute_all_stability_metrics(
            shap_orig, shap_pert, X_ref, X_te_p, TOP_K_VALUES
        )

        imp_pert = get_feature_importance(shap_pert)

        record = {
            **base_record,
            "perturbation": pert_type,
            "level": level,
            **stability,
            "feature_importance": imp_pert.tolist(),
        }
        results.append(record)

    return results


def main():
    print("=" * 70)
    print("NICHE 1: Explanation Stability Under Distributional Shift")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    all_results = []
    t_start = time.time()

    for dataset_name in DATASETS:
        print(f"\n--- Dataset: {dataset_name} ---")
        for model_name in MODEL_NAMES:
            for seed in RANDOM_SEEDS:
                try:
                    results = run_single_experiment(dataset_name, model_name, seed)
                    all_results.extend(results)
                except Exception as e:
                    print(f"  ERROR: {dataset_name}/{model_name}/seed={seed}: {e}")

    elapsed = time.time() - t_start
    print(f"\nTotal experiments: {len(all_results)}")
    print(f"Elapsed: {elapsed:.1f}s")

    df = pd.DataFrame(all_results)

    fi_col = df.pop("feature_importance")
    csv_path = RESULTS_DIR / "stability_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

    fi_path = RESULTS_DIR / "feature_importances.json"
    fi_data = []
    for i, row in df.iterrows():
        fi_data.append({
            "dataset": row["dataset"],
            "model": row["model"],
            "seed": row["seed"],
            "perturbation": row["perturbation"],
            "level": row["level"],
            "importance": fi_col.iloc[i],
        })
    with open(fi_path, "w") as f:
        json.dump(fi_data, f, indent=2)
    print(f"Feature importances saved to {fi_path}")

    print("\n--- Summary Statistics ---")
    summary_cols = ["dataset", "model", "perturbation", "level",
                    "spearman_rho", "lipschitz_mean", "jaccard_top3", "jaccard_top5"]
    summary = df[summary_cols].groupby(
        ["dataset", "model", "perturbation", "level"]
    ).agg(["mean", "std"])
    summary_path = RESULTS_DIR / "summary_stats.csv"
    summary.to_csv(summary_path)
    print(summary.to_string())
    print(f"\nSummary saved to {summary_path}")

    return df


if __name__ == "__main__":
    main()
