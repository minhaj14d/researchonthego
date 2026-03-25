# Research Notes — Explanation Stability Under Distributional Shift
> Last updated: 2026-03-25
> Status: Active

## Overview

This project investigates the **stability of post-hoc explanations (SHAP, LIME) under distributional shift** — an underserved niche at the intersection of AI interpretability and applied mathematics.

**Core Research Question:** Do feature attributions from post-hoc explanation methods remain stable when input data undergoes realistic perturbations (noise injection, label corruption, feature dropout)?

## Navigation

| Folder | Contents |
|--------|----------|
| `niches/` | Detailed niche analysis and scoping |
| `daily-log/` | Day-by-day experiment logs |
| `methods/` | Datasets, metrics, and model configurations |
| `literature/` | Paper summaries and references |
| `decisions/` | Decision log with rationale |
| `drafts/` | Working paper outline |

## 5-Day Plan

- **Day 1** (2026-03-25): Dataset loading, EDA, baseline model training, initial SHAP computation
- **Day 2** (2026-03-26): Perturbation injection (Gaussian noise, label flip, feature dropout at 5/10/15%)
- **Day 3** (2026-03-27): Stability metrics — Spearman rank correlation, Lipschitz ratio, top-k Jaccard
- **Day 4** (2026-03-28): Compare inherently interpretable models (EBM, Decision Tree) vs. post-hoc explained black-box
- **Day 5** (2026-03-29): Statistical analysis, Pareto plots, draft findings

## Datasets

- UCI Heart Disease (303 × 13)
- Kaggle Pima Indians Diabetes (768 × 8)
- UCI Breast Cancer Wisconsin (569 × 30)

## Target Venues

NeurIPS XAI Workshop, AAAI, IJCAI
