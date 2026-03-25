"""
Day 2: Per-feature stability analysis + model-space Lipschitz for label-flip.

Tracks how each individual feature's SHAP attribution changes under perturbation,
rather than the aggregate metrics from Day 1.
"""
import json
import time
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config import (
    DATASETS, PERTURBATION_TYPES, PERTURBATION_LEVELS,
    RANDOM_SEEDS, RESULTS_DIR
)
from data_loader import get_dataset
from train_baselines import (
    build_model, train_and_evaluate, compute_shap_values, get_feature_importance
)
from perturbations import apply_perturbation


MODEL_NAMES = ["xgboost", "logistic_regression"]


def per_feature_shap_shift(shap_orig, shap_pert):
    """
    Per-feature: mean absolute change in SHAP value across samples.
    Returns array of shape (n_features,).
    """
    return np.mean(np.abs(shap_pert - shap_orig), axis=0)


def per_feature_rank_shift(shap_orig, shap_pert):
    """
    Per-feature: how much each feature's rank changes.
    Rank 0 = least important, rank n-1 = most important.
    """
    imp_orig = np.mean(np.abs(shap_orig), axis=0)
    imp_pert = np.mean(np.abs(shap_pert), axis=0)
    rank_orig = np.argsort(np.argsort(imp_orig))
    rank_pert = np.argsort(np.argsort(imp_pert))
    return rank_pert - rank_orig


def per_feature_correlation(shap_orig, shap_pert):
    """
    Per-feature: Pearson correlation of that feature's SHAP values across samples.
    High correlation = stable attribution pattern for that feature.
    """
    n_features = shap_orig.shape[1]
    correlations = np.zeros(n_features)
    for j in range(n_features):
        if np.std(shap_orig[:, j]) < 1e-10 or np.std(shap_pert[:, j]) < 1e-10:
            correlations[j] = 1.0 if np.allclose(shap_orig[:, j], shap_pert[:, j]) else 0.0
        else:
            correlations[j] = np.corrcoef(shap_orig[:, j], shap_pert[:, j])[0, 1]
    return correlations


def model_space_lipschitz(shap_orig, shap_pert, model_orig, model_pert, model_name):
    """
    Model-space Lipschitz: ||ΔSHAP||₂ / ||Δθ||₂
    where θ are model parameters. Applicable to label-flip where ΔX = 0.

    For LogReg: θ = coef_ vector.
    For XGBoost: θ = feature_importances_ vector (proxy for internal weights).
    """
    delta_shap = np.mean(np.abs(shap_pert - shap_orig), axis=0)
    shap_norm = np.linalg.norm(delta_shap)

    if model_name == "logistic_regression":
        theta_orig = model_orig.coef_.flatten()
        theta_pert = model_pert.coef_.flatten()
    elif model_name in ("xgboost", "decision_tree"):
        theta_orig = model_orig.feature_importances_
        theta_pert = model_pert.feature_importances_
    else:
        return 0.0, 0.0

    delta_theta = np.linalg.norm(theta_pert - theta_orig)
    if delta_theta < 1e-10:
        return 0.0, 0.0

    ratio = shap_norm / delta_theta
    return float(ratio), float(delta_theta)


def run_feature_analysis():
    print("=" * 70)
    print("DAY 2: Per-Feature Stability Analysis")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    feature_records = []
    model_lipschitz_records = []
    t_start = time.time()

    for dataset_name in DATASETS:
        print(f"\n--- Dataset: {dataset_name} ---")
        for model_name in MODEL_NAMES:
            for seed in RANDOM_SEEDS:
                print(f"  [{dataset_name}] [{model_name}] seed={seed}")
                data = get_dataset(dataset_name, seed=seed)
                fnames = data["feature_names"]

                model_orig = build_model(model_name, seed=seed)
                model_orig, _ = train_and_evaluate(
                    model_orig, data["X_train"], data["y_train"],
                    data["X_test"], data["y_test"]
                )
                shap_orig = compute_shap_values(
                    model_orig, data["X_train"], data["X_test"], model_name
                )

                for pert_type, level in product(PERTURBATION_TYPES, PERTURBATION_LEVELS):
                    X_tr_p, y_tr_p, X_te_p, needs_retrain = apply_perturbation(
                        data["X_train"], data["y_train"], data["X_test"],
                        pert_type, level, seed
                    )

                    model_pert = model_orig
                    if needs_retrain:
                        model_pert = build_model(model_name, seed=seed)
                        model_pert.fit(X_tr_p, y_tr_p)
                        shap_pert = compute_shap_values(
                            model_pert, X_tr_p, X_te_p, model_name
                        )
                    else:
                        shap_pert = compute_shap_values(
                            model_orig, data["X_train"], X_te_p, model_name
                        )

                    shap_shift = per_feature_shap_shift(shap_orig, shap_pert)
                    rank_shift = per_feature_rank_shift(shap_orig, shap_pert)
                    feat_corr = per_feature_correlation(shap_orig, shap_pert)
                    imp_orig = get_feature_importance(shap_orig)
                    imp_pert = get_feature_importance(shap_pert)

                    for j, fname in enumerate(fnames):
                        feature_records.append({
                            "dataset": dataset_name,
                            "model": model_name,
                            "seed": seed,
                            "perturbation": pert_type,
                            "level": level,
                            "feature": fname,
                            "feature_idx": j,
                            "shap_shift": float(shap_shift[j]),
                            "rank_shift": int(rank_shift[j]),
                            "feature_correlation": float(feat_corr[j]),
                            "importance_original": float(imp_orig[j]),
                            "importance_perturbed": float(imp_pert[j]),
                        })

                    if needs_retrain:
                        ml_ratio, delta_theta = model_space_lipschitz(
                            shap_orig, shap_pert, model_orig, model_pert, model_name
                        )
                        model_lipschitz_records.append({
                            "dataset": dataset_name,
                            "model": model_name,
                            "seed": seed,
                            "perturbation": pert_type,
                            "level": level,
                            "model_lipschitz": ml_ratio,
                            "delta_theta_norm": delta_theta,
                        })

    elapsed = time.time() - t_start
    print(f"\nElapsed: {elapsed:.1f}s")

    df_feat = pd.DataFrame(feature_records)
    feat_path = RESULTS_DIR / "per_feature_stability.csv"
    df_feat.to_csv(feat_path, index=False)
    print(f"Per-feature results saved to {feat_path} ({len(df_feat)} rows)")

    df_ml = pd.DataFrame(model_lipschitz_records)
    ml_path = RESULTS_DIR / "model_lipschitz.csv"
    df_ml.to_csv(ml_path, index=False)
    print(f"Model-space Lipschitz saved to {ml_path} ({len(df_ml)} rows)")

    print("\n--- Most Unstable Features (by mean SHAP shift, label_flip @ 15%) ---")
    mask = (df_feat["perturbation"] == "label_flip") & (df_feat["level"] == 0.15)
    top_unstable = (
        df_feat[mask]
        .groupby(["dataset", "model", "feature"])["shap_shift"]
        .mean()
        .reset_index()
        .sort_values("shap_shift", ascending=False)
    )
    print(top_unstable.head(20).to_string(index=False))

    print("\n--- Model-Space Lipschitz (label_flip) ---")
    if not df_ml.empty:
        ml_summary = df_ml.groupby(
            ["dataset", "model", "perturbation", "level"]
        )["model_lipschitz"].agg(["mean", "std"])
        print(ml_summary.to_string())

    return df_feat, df_ml


if __name__ == "__main__":
    run_feature_analysis()
