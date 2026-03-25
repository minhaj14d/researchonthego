# Decisions Log
> Last updated: 2026-03-25
> Status: Active

## Overview

Traceable record of research decisions — what was chosen, what was rejected, and why.

## Decisions

### DEC-001: Chose Niche 1 (Explanation Stability) Over Niches 2 and 3
- **Date**: 2026-03-25
- **Chose**: Stability of post-hoc explanations under distributional shift
- **Over**: Symbolic regression for engineering data (Niche 2), Cost-aware interpretability trade-off (Niche 3)
- **Rationale**: Highest novelty-to-effort ratio. Clear gap (<10 papers). Lipschitz stability metric is a publishable mathematical contribution on its own. Straightforward experimental design fits 5-day window.

### DEC-002: Selected 3 UCI Datasets for Diversity
- **Date**: 2026-03-25
- **Chose**: Heart Disease (303×13), Pima Diabetes (768×8), Breast Cancer (569×30)
- **Over**: Adult Income (too large for quick SHAP), MNIST (not tabular)
- **Rationale**: Cover diverse sample sizes (303–768), feature counts (8–30), and domains (cardiology, endocrinology, oncology). All fit in memory. SHAP computation under 1 min each.

### DEC-003: XGBoost + Logistic Regression as Baseline Pair
- **Date**: 2026-03-25
- **Chose**: XGBoost (nonlinear, tree-based) + Logistic Regression (linear)
- **Over**: Random Forest (redundant with XGBoost), SVM (SHAP is slower), Neural Net (overkill for these dataset sizes)
- **Rationale**: Maximizes contrast between model families. Tree SHAP is exact and fast. Linear SHAP is exact and trivial. Clean comparison.
