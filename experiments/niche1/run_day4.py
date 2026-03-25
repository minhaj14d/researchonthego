"""
Day 4: Inherently Interpretable vs Post-Hoc Explained Models.

Runs all 4 models (XGBoost, LogReg, DecisionTree, EBM) through the full
stability + faithfulness pipeline, then compares interpretable vs black-box.
"""
import json
import time
import warnings
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config import (
    DATASETS, PERTURBATION_TYPES, PERTURBATION_LEVELS,
    TOP_K_VALUES, RANDOM_SEEDS, RESULTS_DIR, ALL_MODELS
)
from data_loader import get_dataset
from train_baselines import (
    build_model, train_and_evaluate, compute_shap_values,
    get_feature_importance, compute_native_importance
)
from perturbations import apply_perturbation
from stability_metrics import compute_all_stability_metrics

warnings.filterwarnings("ignore")


def surrogate_fidelity(model, shap_values, X_test, model_name=None, feature_names=None):
    import pandas as pd
    from sklearn.metrics import mean_absolute_error
    if model_name == "ebm" and feature_names is not None:
        X_df = pd.DataFrame(X_test, columns=feature_names)
        y_prob = model.predict_proba(X_df)[:, 1]
    else:
        y_prob = model.predict_proba(X_test)[:, 1]
    base_value = np.mean(y_prob)
    shap_pred = np.clip(base_value + np.sum(shap_values, axis=1), 0, 1)
    mae_s = mean_absolute_error(y_prob, shap_pred)
    mae_b = mean_absolute_error(y_prob, np.full_like(y_prob, base_value))
    return float(1.0 - mae_s / mae_b) if mae_b > 1e-10 else 1.0


def native_importance_stability(imp_orig, imp_pert, top_k_values=(3, 5)):
    """Stability of native (non-SHAP) feature importances."""
    if imp_orig is None or imp_pert is None:
        return {}
    corr, p = spearmanr(imp_orig, imp_pert)
    results = {"native_spearman": float(corr), "native_spearman_p": float(p)}
    for k in top_k_values:
        top_o = set(np.argsort(imp_orig)[-k:])
        top_p = set(np.argsort(imp_pert)[-k:])
        inter = len(top_o & top_p)
        union = len(top_o | top_p)
        results[f"native_jaccard_top{k}"] = inter / union if union > 0 else 1.0
    return results


def run_day4():
    print("=" * 70)
    print("DAY 4: Interpretable vs Black-Box Model Comparison")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    all_records = []
    t_start = time.time()

    for dataset_name in DATASETS:
        print(f"\n{'='*50}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*50}")

        for model_name in ALL_MODELS:
            for seed in RANDOM_SEEDS:
                print(f"  [{model_name}] seed={seed}", end=" ")
                t0 = time.time()

                data = get_dataset(dataset_name, seed=seed)
                model_orig = build_model(model_name, seed=seed)
                model_orig, perf = train_and_evaluate(
                    model_orig, data["X_train"], data["y_train"],
                    data["X_test"], data["y_test"],
                    feature_names=data["feature_names"]
                )

                fnames = data["feature_names"]
                shap_orig = compute_shap_values(
                    model_orig, data["X_train"], data["X_test"], model_name,
                    feature_names=fnames
                )
                native_orig = compute_native_importance(
                    model_orig, data["X_test"], model_name
                )
                fid_orig = surrogate_fidelity(model_orig, shap_orig, data["X_test"],
                                              model_name=model_name, feature_names=fnames)

                base = {
                    "dataset": dataset_name, "model": model_name, "seed": seed,
                    "model_type": "interpretable" if model_name in ("decision_tree", "ebm") else "black_box",
                    **perf,
                }

                # Baseline record
                rec = {
                    **base, "perturbation": "none", "level": 0.0,
                    "spearman_rho": 1.0, "lipschitz_mean": 0.0, "lipschitz_std": 0.0,
                    "surrogate_fidelity": fid_orig,
                }
                for k in TOP_K_VALUES:
                    rec[f"jaccard_top{k}"] = 1.0
                rec.update(native_importance_stability(native_orig, native_orig, TOP_K_VALUES))
                all_records.append(rec)

                # Perturbations
                for pert_type, level in product(PERTURBATION_TYPES, PERTURBATION_LEVELS):
                    X_tr_p, y_tr_p, X_te_p, needs_retrain = apply_perturbation(
                        data["X_train"], data["y_train"], data["X_test"],
                        pert_type, level, seed
                    )

                    if needs_retrain:
                        model_p = build_model(model_name, seed=seed)
                        _, _ = train_and_evaluate(
                            model_p, X_tr_p, y_tr_p, X_te_p,
                            data["y_test"], feature_names=fnames
                        )
                        shap_pert = compute_shap_values(
                            model_p, X_tr_p, X_te_p, model_name,
                            feature_names=fnames
                        )
                        native_pert = compute_native_importance(model_p, X_te_p, model_name)
                        active_model = model_p
                    else:
                        shap_pert = compute_shap_values(
                            model_orig, data["X_train"], X_te_p, model_name,
                            feature_names=fnames
                        )
                        native_pert = native_orig
                        active_model = model_orig

                    X_ref = data["X_test"]
                    stability = compute_all_stability_metrics(
                        shap_orig, shap_pert, X_ref, X_te_p, TOP_K_VALUES
                    )
                    fid = surrogate_fidelity(active_model, shap_pert, X_te_p,
                                            model_name=model_name, feature_names=fnames)
                    native_stab = native_importance_stability(
                        native_orig, native_pert, TOP_K_VALUES
                    )

                    rec = {**base, "perturbation": pert_type, "level": level,
                           **stability, "surrogate_fidelity": fid, **native_stab}
                    all_records.append(rec)

                elapsed_model = time.time() - t0
                print(f"({elapsed_model:.1f}s)")

    total_elapsed = time.time() - t_start
    df = pd.DataFrame(all_records)

    out_path = RESULTS_DIR / "day4_all_models.csv"
    df.to_csv(out_path, index=False)
    print(f"\nTotal: {len(df)} records in {total_elapsed:.1f}s")
    print(f"Saved to {out_path}")

    # Summary: stability by model type
    print("\n" + "=" * 70)
    print("SUMMARY: Black-Box vs Interpretable Under Label Flip")
    print("=" * 70)
    lf = df[df["perturbation"] == "label_flip"]
    summary = lf.groupby(["dataset", "model", "model_type"]).agg(
        accuracy=("accuracy", "mean"),
        spearman_rho=("spearman_rho", "mean"),
        fidelity=("surrogate_fidelity", "mean"),
        native_spearman=("native_spearman", "mean"),
    ).round(3)
    print(summary.to_string())

    print("\n" + "=" * 70)
    print("SUMMARY: Model Performance (Baseline, No Perturbation)")
    print("=" * 70)
    bl = df[df["perturbation"] == "none"]
    perf_summary = bl.groupby(["dataset", "model", "model_type"]).agg(
        accuracy=("accuracy", "mean"),
        f1=("f1", "mean"),
        auc_roc=("auc_roc", "mean"),
        fidelity=("surrogate_fidelity", "mean"),
    ).round(3)
    print(perf_summary.to_string())

    return df


if __name__ == "__main__":
    run_day4()
