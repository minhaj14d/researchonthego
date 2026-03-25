# Niche 1 — Stability of Post-Hoc Explanations Under Distributional Shift
> Last updated: 2026-03-25
> Status: Active

## Overview

Most XAI research evaluates SHAP/LIME on static, in-distribution data. This niche investigates how feature attributions drift when the input distribution shifts — a critical concern for real-world deployment where data quality is variable.

## Research Gap

- Jan 2026 comparative analysis (arXiv:2601.00428) benchmarked 16 interpretable methods on accuracy but **did not measure explanation stability** under perturbation.
- Mar 2026 ViT sparsity paper (arXiv:2603.15919) showed structural sparsity "redistributes rather than isolates computation" — explanations can be fragile even in sparse models.
- Schmidt Sciences 2026 Interpretability RFP calls for explanations that "generalize beyond academic benchmarks" — stability under shift is exactly this.
- **Fewer than 10 papers** formally quantify explanation stability under distributional shift.

## Mathematical Foundation

### Lipschitz Stability Metric

For a model `f` and explanation function `g` (e.g., SHAP), define:

```
L(g, f) = sup_{x, x'} ||g(f, x) - g(f, x')|| / ||x - x'||
```

Where `||·||` is the L2 norm. A lower Lipschitz ratio = more stable explanations.

### Perturbation Types

1. **Gaussian noise**: x' = x + ε, ε ~ N(0, σ²I), σ ∈ {0.05, 0.10, 0.15}
2. **Label flip**: Randomly corrupt y labels at rates {5%, 10%, 15%} and retrain
3. **Feature dropout**: Zero out random features at rates {5%, 10%, 15%}

### Stability Metrics

1. **Spearman rank correlation** of feature importance vectors (before vs. after perturbation)
2. **Lipschitz ratio** (L2 norm of SHAP change / L2 norm of input change)
3. **Top-k Jaccard similarity** (overlap of top-k important features, k=3,5)

## Tasks

- [x] Define research gap and mathematical framework
- [x] Load and preprocess 3 UCI datasets
- [x] Train baseline models (XGBoost, Logistic Regression)
- [x] Compute baseline SHAP values
- [x] Run perturbation experiments (3 types × 3 levels × 5 seeds = 300 experiments)
- [x] Compute stability metrics (Spearman ρ, Lipschitz ratio, Top-k Jaccard)
- [x] Per-feature stability analysis (Day 2) — most unstable features = most important
- [x] Model-space Lipschitz for label-flip (Day 2) — XGBoost 10–20x more sensitive than LogReg
- [x] Surrogate fidelity + perturbation faithfulness analysis (Day 3)
- [x] Statistical significance tests: Wilcoxon, Mann-Whitney, Cohen's d, Bootstrap CIs (Day 3)
- [x] 11 publication plots generated (7 Day 2 + 4 Day 3)
- [x] Compare inherently interpretable vs. post-hoc explained models (Day 4) — EBM wins on stability, Decision Tree surprisingly unstable
- [ ] Final Pareto analysis, consolidated results table, and paper draft (Day 5)

## References

- arXiv:2601.00428 — Comparative Analysis of Interpretable ML Methods (Jan 2026)
- arXiv:2603.15919 — Sparse but not Simpler: ViT Interpretability (Mar 2026)
- arXiv:2408.06183 — Shapley-value Interpretability for Heart Disease (2024)
- arXiv:2603.13806 — Interpretable Sparse PCA Framework (Mar 2026)
- Schmidt Sciences 2026 Interpretability RFP
