"""
Stability metrics comparing original vs. perturbed SHAP explanations.
"""
import numpy as np
from scipy.stats import spearmanr


def spearman_rank_stability(importance_original, importance_perturbed):
    """
    Spearman rank correlation between feature importance vectors.
    Returns correlation coefficient (higher = more stable).
    """
    corr, p_value = spearmanr(importance_original, importance_perturbed)
    return corr, p_value


def lipschitz_ratio(shap_original, shap_perturbed, X_original, X_perturbed):
    """
    Per-sample Lipschitz ratio: ||ΔSHAP||₂ / ||ΔX||₂.
    Returns mean and std across samples.
    """
    delta_shap = np.linalg.norm(shap_perturbed - shap_original, axis=1)
    delta_x = np.linalg.norm(X_perturbed - X_original, axis=1)

    valid = delta_x > 1e-10
    if valid.sum() == 0:
        return 0.0, 0.0

    ratios = delta_shap[valid] / delta_x[valid]
    return float(np.mean(ratios)), float(np.std(ratios))


def topk_jaccard(importance_original, importance_perturbed, k):
    """
    Jaccard similarity of top-k features by absolute importance.
    Returns J ∈ [0, 1] (higher = more stable).
    """
    top_orig = set(np.argsort(np.abs(importance_original))[-k:])
    top_pert = set(np.argsort(np.abs(importance_perturbed))[-k:])
    intersection = len(top_orig & top_pert)
    union = len(top_orig | top_pert)
    return intersection / union if union > 0 else 1.0


def compute_all_stability_metrics(
    shap_orig, shap_pert, X_orig, X_pert, top_k_values=(3, 5)
):
    """Compute all stability metrics and return as a dict."""
    imp_orig = np.mean(np.abs(shap_orig), axis=0)
    imp_pert = np.mean(np.abs(shap_pert), axis=0)

    spearman_corr, spearman_p = spearman_rank_stability(imp_orig, imp_pert)

    lip_mean, lip_std = lipschitz_ratio(shap_orig, shap_pert, X_orig, X_pert)

    results = {
        "spearman_rho": spearman_corr,
        "spearman_p_value": spearman_p,
        "lipschitz_mean": lip_mean,
        "lipschitz_std": lip_std,
    }

    for k in top_k_values:
        results[f"jaccard_top{k}"] = topk_jaccard(imp_orig, imp_pert, k)

    return results
