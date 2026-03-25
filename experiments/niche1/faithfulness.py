"""
Day 3: Explanation Faithfulness / Fidelity Analysis.

Measures how well SHAP explanations actually represent the model's behavior:
1. Surrogate Fidelity: Build a linear model from SHAP values, measure prediction agreement.
2. Perturbation Faithfulness: Mask top-k SHAP features, measure prediction drop (should drop if SHAP is faithful).
3. Compare faithfulness before vs. after distributional shift.
"""
import time
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, accuracy_score

from config import (
    DATASETS, PERTURBATION_TYPES, PERTURBATION_LEVELS,
    RANDOM_SEEDS, RESULTS_DIR, TOP_K_VALUES
)
from data_loader import get_dataset
from train_baselines import build_model, train_and_evaluate, compute_shap_values


MODEL_NAMES = ["xgboost", "logistic_regression"]


def surrogate_fidelity(model, shap_values, X_test, model_name):
    """
    Build a linear surrogate from SHAP values and measure how well it
    reproduces the original model's probability predictions.

    g_SHAP(x) = E[f(x)] + Σ SHAP_j(x)   (SHAP's additive property)

    Fidelity = 1 - MAE(f(x), g_SHAP(x)) / MAE(f(x), mean(f(x)))
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    base_value = np.mean(y_prob)

    shap_pred = base_value + np.sum(shap_values, axis=1)
    shap_pred = np.clip(shap_pred, 0, 1)

    mae_surrogate = mean_absolute_error(y_prob, shap_pred)
    mae_baseline = mean_absolute_error(y_prob, np.full_like(y_prob, base_value))

    if mae_baseline < 1e-10:
        fidelity = 1.0
    else:
        fidelity = 1.0 - mae_surrogate / mae_baseline

    return {
        "surrogate_fidelity": float(fidelity),
        "surrogate_mae": float(mae_surrogate),
        "baseline_mae": float(mae_baseline),
    }


def perturbation_faithfulness(model, shap_values, X_test, k_values=(3, 5)):
    """
    Mask top-k most important features (per sample) and measure prediction drop.

    If SHAP is faithful, removing the most important features should cause
    the largest prediction change (Monotonicity test).

    Also masks bottom-k features as control — prediction should change less.
    """
    y_prob_orig = model.predict_proba(X_test)[:, 1]
    results = {}

    for k in k_values:
        abs_shap = np.abs(shap_values)

        top_k_drop = []
        bottom_k_drop = []
        for i in range(len(X_test)):
            sorted_idx = np.argsort(abs_shap[i])

            # Mask top-k (most important)
            X_masked_top = X_test[i].copy()
            top_idx = sorted_idx[-k:]
            X_masked_top[top_idx] = 0.0
            prob_top = model.predict_proba(X_masked_top.reshape(1, -1))[0, 1]
            top_k_drop.append(abs(y_prob_orig[i] - prob_top))

            # Mask bottom-k (least important)
            X_masked_bot = X_test[i].copy()
            bot_idx = sorted_idx[:k]
            X_masked_bot[bot_idx] = 0.0
            prob_bot = model.predict_proba(X_masked_bot.reshape(1, -1))[0, 1]
            bottom_k_drop.append(abs(y_prob_orig[i] - prob_bot))

        mean_top = float(np.mean(top_k_drop))
        mean_bot = float(np.mean(bottom_k_drop))

        results[f"top{k}_mask_drop"] = mean_top
        results[f"bot{k}_mask_drop"] = mean_bot
        # Faithfulness ratio: top-k masking should cause MORE drop than bottom-k
        if mean_bot < 1e-10:
            results[f"faithfulness_ratio_k{k}"] = float(mean_top / 1e-10)
        else:
            results[f"faithfulness_ratio_k{k}"] = float(mean_top / mean_bot)

    return results


def run_faithfulness_analysis():
    print("=" * 70)
    print("DAY 3: Explanation Faithfulness Analysis")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    records = []
    t_start = time.time()

    for dataset_name in DATASETS:
        print(f"\n--- Dataset: {dataset_name} ---")
        for model_name in MODEL_NAMES:
            for seed in RANDOM_SEEDS:
                print(f"  [{dataset_name}] [{model_name}] seed={seed}")
                data = get_dataset(dataset_name, seed=seed)

                model_orig = build_model(model_name, seed=seed)
                model_orig, perf = train_and_evaluate(
                    model_orig, data["X_train"], data["y_train"],
                    data["X_test"], data["y_test"]
                )
                shap_orig = compute_shap_values(
                    model_orig, data["X_train"], data["X_test"], model_name
                )

                # Baseline (no perturbation)
                sf = surrogate_fidelity(model_orig, shap_orig, data["X_test"], model_name)
                pf = perturbation_faithfulness(model_orig, shap_orig, data["X_test"], TOP_K_VALUES)
                records.append({
                    "dataset": dataset_name, "model": model_name, "seed": seed,
                    "perturbation": "none", "level": 0.0,
                    **perf, **sf, **pf,
                })

                # Under each perturbation
                for pert_type, level in product(PERTURBATION_TYPES, PERTURBATION_LEVELS):
                    from perturbations import apply_perturbation
                    X_tr_p, y_tr_p, X_te_p, needs_retrain = apply_perturbation(
                        data["X_train"], data["y_train"], data["X_test"],
                        pert_type, level, seed
                    )

                    if needs_retrain:
                        model_p = build_model(model_name, seed=seed)
                        model_p.fit(X_tr_p, y_tr_p)
                        shap_pert = compute_shap_values(model_p, X_tr_p, X_te_p, model_name)
                        active_model = model_p
                    else:
                        shap_pert = compute_shap_values(
                            model_orig, data["X_train"], X_te_p, model_name
                        )
                        active_model = model_orig

                    sf = surrogate_fidelity(active_model, shap_pert, X_te_p, model_name)
                    pf = perturbation_faithfulness(active_model, shap_pert, X_te_p, TOP_K_VALUES)

                    records.append({
                        "dataset": dataset_name, "model": model_name, "seed": seed,
                        "perturbation": pert_type, "level": level,
                        **perf, **sf, **pf,
                    })

    elapsed = time.time() - t_start
    df = pd.DataFrame(records)

    out_path = RESULTS_DIR / "faithfulness_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nElapsed: {elapsed:.1f}s")
    print(f"Faithfulness results saved to {out_path} ({len(df)} rows)")

    print("\n--- Surrogate Fidelity Summary ---")
    summary = df.groupby(
        ["dataset", "model", "perturbation", "level"]
    )[["surrogate_fidelity", "faithfulness_ratio_k3", "faithfulness_ratio_k5"]].agg(["mean", "std"])
    print(summary.to_string())

    return df


if __name__ == "__main__":
    run_faithfulness_analysis()
