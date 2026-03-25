# Metrics
> Last updated: 2026-03-25
> Status: Active

## Overview

Metrics are divided into two categories: **predictive performance** metrics and **explanation stability** metrics.

## Predictive Performance

| Metric | Formula | Purpose |
|--------|---------|---------|
| Accuracy | (TP+TN)/(TP+TN+FP+FN) | Overall correctness |
| F1 Score | 2·(P·R)/(P+R) | Balance precision and recall |
| AUC-ROC | Area under ROC curve | Threshold-independent discrimination |

## Explanation Stability Metrics

### 1. Spearman Rank Correlation (ρ_s)

Measures whether the **ranking** of feature importances is preserved after perturbation.

```
ρ_s = 1 - (6 Σ d_i²) / (n(n²-1))
```

Where `d_i` is the rank difference for feature `i`, `n` = number of features.

- Range: [-1, 1]. Higher = more stable.
- Interpretation: ρ_s > 0.8 = strong stability, ρ_s < 0.5 = unstable.

### 2. Lipschitz Stability Ratio (L)

Measures the **magnitude** of explanation change relative to input change.

```
L = ||SHAP(x') - SHAP(x)||₂ / ||x' - x||₂
```

- Lower = more stable.
- Computed per-sample, then aggregated (mean ± std across test set).

### 3. Top-k Jaccard Similarity (J_k)

Measures overlap of the **top-k most important features** before/after perturbation.

```
J_k = |S_k ∩ S'_k| / |S_k ∪ S'_k|
```

Where `S_k` = set of top-k features by |SHAP value| in original, `S'_k` = after perturbation.

- k ∈ {3, 5}
- Range: [0, 1]. Higher = more stable.

### 4. Explanation Faithfulness (Fidelity)

How well a linear surrogate built from SHAP values reproduces the original model's predictions.

```
Fidelity = 1 - MAE(f(x), g_SHAP(x)) / MAE(f(x), ȳ)
```

Where `g_SHAP` is a linear model using SHAP values as coefficients.

## Notes

- All stability metrics are computed per perturbation type × perturbation level × model × dataset.
- Results aggregated over 5 random seeds for statistical robustness.
