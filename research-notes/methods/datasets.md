# Datasets
> Last updated: 2026-03-25
> Status: Active

## Overview

Three UCI/Kaggle classification datasets selected for diversity in sample size, feature count, and domain.

## Dataset Details

### 1. UCI Heart Disease (Cleveland)

| Property | Value |
|----------|-------|
| Source | UCI ML Repository |
| Samples | 303 |
| Features | 13 (mixed numeric/categorical) |
| Target | Binary (presence of heart disease) |
| Class Balance | ~54% negative, ~46% positive |
| Key Features | age, sex, cp (chest pain type), trestbps, chol, thalach, oldpeak |
| Preprocessing | Drop rows with missing values (~6 rows). Encode categoricals. StandardScaler on numerics. |

### 2. Pima Indians Diabetes

| Property | Value |
|----------|-------|
| Source | Kaggle (originally NIDDK) |
| Samples | 768 |
| Features | 8 (all numeric) |
| Target | Binary (diabetes onset within 5 years) |
| Class Balance | ~65% negative, ~35% positive |
| Key Features | Glucose, BMI, Age, Insulin, BloodPressure |
| Preprocessing | Replace 0-valued columns (Glucose, BP, SkinThickness, Insulin, BMI) with NaN → median impute. StandardScaler. |

### 3. UCI Breast Cancer Wisconsin (Diagnostic)

| Property | Value |
|----------|-------|
| Source | UCI ML Repository / sklearn |
| Samples | 569 |
| Features | 30 (all numeric, computed from cell nuclei images) |
| Target | Binary (malignant vs. benign) |
| Class Balance | ~63% benign, ~37% malignant |
| Key Features | radius_mean, texture_mean, perimeter_mean, area_mean, concavity_mean |
| Preprocessing | StandardScaler. No missing values. |

## Notes

- All datasets are small enough for CPU-only SHAP computation (no GPU needed).
- Feature standardization is critical for fair Lipschitz ratio computation across datasets.
- Label encoding: positive class = 1 in all datasets for consistency.
