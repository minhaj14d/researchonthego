# Paper Outline — Stability of Post-Hoc Explanations Under Distributional Shift
> Last updated: 2026-03-25
> Status: Draft

## Working Title

**"How Fragile Are Your Explanations? Measuring the Stability of SHAP Attributions Under Distributional Shift"**

## Abstract (Draft)

Post-hoc explanation methods like SHAP are widely used to interpret black-box models, yet their behavior under realistic data perturbations remains poorly understood. We introduce a formal Lipschitz-style stability metric and systematically evaluate SHAP explanation stability across three perturbation types (Gaussian noise, label corruption, feature dropout), three UCI classification datasets, and four model families. Our results reveal [TBD: key finding] and provide actionable guidelines for practitioners deploying interpretable ML in noisy, real-world environments.

## Structure

### 1. Introduction
- XAI is standard practice, but explanations are evaluated on static data
- Real-world data drifts — do explanations drift too?
- Contribution: formal stability metric + systematic empirical study

### 2. Related Work
- SHAP/LIME foundations (Lundberg & Lee 2017, Ribeiro et al. 2016)
- Explanation robustness: Alvarez-Melis & Jaakkola (2018) — local Lipschitz
- Gap: no systematic study across perturbation types, datasets, model families

### 3. Methodology
- 3.1 Stability metrics: Spearman ρ, Lipschitz ratio, Top-k Jaccard
- 3.2 Perturbation protocol: Gaussian noise, label flip, feature dropout
- 3.3 Experimental setup: datasets, models, SHAP computation

### 4. Experiments & Results
- 4.1 Baseline model performance
- 4.2 Explanation stability under each perturbation type
- 4.3 Model family comparison (black-box vs. interpretable)
- 4.4 Scaling behavior: stability vs. perturbation intensity

### 5. Analysis & Discussion
- Which perturbation type causes the most instability?
- Are inherently interpretable models more stable?
- Pareto frontier: accuracy vs. explanation stability

### 6. Conclusion
- Summary of findings
- Guidelines for practitioners
- Future work: temporal drift, adversarial perturbations

## References

[To be populated from literature/papers.md]
