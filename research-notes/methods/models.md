# Models
> Last updated: 2026-03-25
> Status: Active

## Overview

Two categories of models: **black-box models** (explained post-hoc with SHAP) and **inherently interpretable models** (explanations are the model itself).

## Black-Box Models (Post-Hoc Explained)

### XGBoost Classifier

```python
{
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "eval_metric": "logloss"
}
```

SHAP method: `shap.TreeExplainer` (exact, polynomial time for trees).

### Logistic Regression

```python
{
    "penalty": "l2",
    "C": 1.0,
    "solver": "lbfgs",
    "max_iter": 1000,
    "random_state": 42
}
```

SHAP method: `shap.LinearExplainer` (exact for linear models).

## Inherently Interpretable Models (Day 4)

### Decision Tree

```python
{
    "max_depth": 5,
    "min_samples_leaf": 10,
    "random_state": 42
}
```

Explanation: Feature importance from tree structure (Gini importance).

### Explainable Boosting Machine (EBM)

```python
{
    "max_bins": 256,
    "interactions": 10,
    "outer_bags": 8,
    "inner_bags": 0,
    "random_state": 42
}
```

Explanation: Built-in shape functions per feature (inherently additive).

## Notes

- All models use 80/20 stratified train/test split.
- 5-fold cross-validation for hyperparameter confirmation.
- Random seed = 42 as base, then {0, 1, 2, 3, 4} for multi-seed experiments.
