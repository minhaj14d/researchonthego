# Literature
> Last updated: 2026-03-25
> Status: Active

## Overview

Key papers informing the research on explanation stability under distributional shift.

## Paper Summaries

### 1. A Comparative Analysis of Interpretable Machine Learning Methods
- **arXiv**: 2601.00428 (Jan 2026)
- **Key Finding**: Evaluated 16 inherently interpretable methods on 216 tabular datasets. Performance differences across models are marginal, but interpretability advantages of white-box approaches remain crucial.
- **Gap Identified**: Did not measure explanation stability under data perturbation.
- **Relevance**: Direct motivation — extends this work by adding a stability dimension.

### 2. Sparse but not Simpler: Multi-Level Interpretability Analysis of Vision Transformers
- **arXiv**: 2603.15919 (Mar 2026)
- **Key Finding**: Structural sparsity produces circuits with 2.5× fewer edges but shows limited improvement in neuron selectivity or attribution faithfulness. Sparsity "redistributes rather than isolates computation."
- **Gap Identified**: Sparsity ≠ interpretability. Need to measure explanation faithfulness directly.
- **Relevance**: Supports our hypothesis that explanations need stability testing beyond static evaluation.

### 3. Centralized and Federated Heart Disease Classification with Shapley-value Interpretability
- **arXiv**: 2408.06183 (Aug 2024)
- **Key Finding**: Applied SHAP to UCI heart disease data. SVM achieved 83.3% accuracy. Feature importances aligned with clinical knowledge.
- **Gap Identified**: Only evaluated on clean, static data. No perturbation or stability analysis.
- **Relevance**: We extend this work by stress-testing the same SHAP explanations under noise.

### 4. An Interpretable and Stable Framework for Sparse Principal Component Analysis
- **arXiv**: 2603.13806 (Mar 2026)
- **Key Finding**: Adaptive variable penalties improve stability in high-dimensional sparse PCA settings.
- **Gap Identified**: Stability of PCA components ≠ stability of prediction explanations.
- **Relevance**: Mathematical framework for stability (Weyl's inequality) applicable to our Lipschitz metric.

### 5. Schmidt Sciences 2026 Interpretability RFP
- **Source**: schmidtsciences.smapply.io
- **Key Finding**: $300k–$1M funding for methods that "generalize beyond academic benchmarks to realistic use cases" and detect deceptive AI behaviors.
- **Relevance**: Validates that explanation robustness in realistic conditions is a funded priority.
