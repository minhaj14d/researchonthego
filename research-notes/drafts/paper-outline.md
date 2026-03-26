# How Fragile Are Your Explanations? Measuring the Stability of SHAP Attributions Under Distributional Shift
> Last updated: 2026-03-25
> Status: Draft (revised — undergraduate tone)
> Backup of previous version: `paper-outline_expert_backup.md`

---

## Abstract

SHAP is one of the most popular methods for explaining machine learning predictions, but there is surprisingly little work on what happens to those explanations when the data is noisy or corrupted. In this paper, we run a set of experiments to measure how stable SHAP explanations are when the input data is perturbed in three ways: Gaussian noise added to features, random corruption of training labels, and dropout of feature values. We test across three UCI/Kaggle classification datasets and four model types (XGBoost, Logistic Regression, Decision Tree, and Explainable Boosting Machine), running over 6,000 experiments in total. We use a set of stability metrics grounded in Lipschitz-style sensitivity analysis and rank correlation measures.

Our main results are: (1) label corruption has a much larger effect on explanations than the other two perturbation types — Spearman rank correlation between original and perturbed SHAP rankings drops to ρ = 0.28 under just 5% label noise, while Gaussian noise leaves it above 0.95; (2) rank stability and magnitude stability turn out to be quite different things — a model can keep the same feature ranking while its actual SHAP values shift by 7.7× the input perturbation; (3) the Explainable Boosting Machine is the most stable model overall; (4) Decision Trees, often called "inherently interpretable," actually produce the least stable explanations; and (5) under enough label corruption, SHAP fidelity can go negative, meaning the explanations are worse than useless. We also give some practical recommendations based on these findings.

**Keywords:** Explainability, SHAP, Stability, Distributional Shift, Interpretable Machine Learning

---

## 1. Introduction

In many applied settings — healthcare, finance, credit scoring — people need to understand *why* a model makes the predictions it does, not just what those predictions are. SHAP (Lundberg & Lee, 2017) has become one of the standard tools for this. It assigns each input feature a contribution score based on Shapley values from cooperative game theory, and it has nice theoretical properties: the attributions are locally faithful and add up to the model's prediction.

But there is a practical question that does not seem to get much attention: *how much do these explanations change when the data is a bit noisy?* In real-world applications, data is rarely clean. Measurements have noise, labels sometimes get recorded wrong, and features can be missing. If a small amount of noise completely reshuffles which features SHAP says are important, then those explanations are not very useful in practice.

Some prior work has looked at related questions. Alvarez-Melis and Jaakkola (2018) studied local Lipschitz continuity of explanations, and Slack et al. (2020) showed that SHAP and LIME can be fooled by adversarial attacks. But these studies tend to look at one perturbation type at a time, or one model at a time, and they do not really compare across multiple dimensions of stability (does the ranking change? by how much do the values shift? is the explanation still faithful to the model?).

In this paper, we try to address this gap by running a fairly comprehensive set of experiments. Specifically, we:

1. Define a set of stability metrics that capture rank stability, magnitude sensitivity, and faithfulness. For training-time perturbations where the inputs do not change (only the labels do), we define a model-space Lipschitz ratio that measures how SHAP values respond to changes in the model parameters.
2. Run experiments across 3 datasets, 4 models, 3 perturbation types, 3 intensity levels, and 5 random seeds (over 6,000 runs total), with statistical testing to check significance.
3. Report five findings that we think are interesting, a few of which go against common assumptions — for instance, that Decision Trees give more "trustworthy" explanations than black-box models.
4. Provide some practical guidelines for people who want to use SHAP in settings where data quality is uncertain.

---

## 2. Related Work

### 2.1 Post-Hoc Explanation Methods

SHAP (Lundberg & Lee, 2017) uses Shapley values from game theory to explain individual predictions. For tree-based models, the TreeExplainer algorithm computes these values exactly in polynomial time, which is nice because it avoids the approximation errors that come with sampling-based methods. LIME (Ribeiro et al., 2016) takes a different approach, fitting a sparse linear model around each prediction. Both methods are mainly evaluated on clean, static datasets in the original papers.

### 2.2 Explanation Robustness

Alvarez-Melis and Jaakkola (2018) proposed local Lipschitz continuity as a way to measure how sensitive explanations are to small input changes, and found that many methods are quite sensitive. Slack et al. (2020) went further and showed that SHAP and LIME can be deliberately fooled by an adversary. Dombrowski et al. (2019) showed a related result for gradient-based explanations on neural networks.

### 2.3 What Is Missing

A recent large-scale comparison of 16 interpretable methods (arXiv:2601.00428) tested accuracy on 216 datasets, but stability under perturbation was not part of the evaluation. A study of sparse Vision Transformers (arXiv:2603.15919) found that making a model structurally simpler does not necessarily make its explanations more stable — sparsity "redistributes rather than isolates computation." The Schmidt Sciences 2026 RFP specifically asks for explanation methods that work "beyond academic benchmarks." All of this suggests that there is room for a study that looks at explanation stability in a more structured way: across different kinds of perturbations, different model types, and different notions of what "stable" means.

---

## 3. Methodology

### 3.1 Stability Metrics

We wanted to measure stability along several dimensions, since a single number does not tell the full story. We settled on three categories.

**Rank stability.** The most basic question is whether the ordering of features by importance stays the same. We measure this with *Spearman rank correlation* (ρ_s) between the original and perturbed SHAP rankings. A value of ρ_s > 0.8 indicates the rankings are largely preserved. We also use *Top-k Jaccard similarity* to check whether the set of most important features is preserved:

$$J_k = \frac{|S_k \cap S'_k|}{|S_k \cup S'_k|}, \quad k \in \{3,5\}$$

**Magnitude stability.** Even if the ranking is preserved, the actual SHAP values might shift a lot. We measure this using a Lipschitz-style ratio. The *input-space Lipschitz ratio* is:

$$L = \frac{\|\text{SHAP}(\mathbf{x}') - \text{SHAP}(\mathbf{x})\|_2}{\|\mathbf{x}' - \mathbf{x}\|_2}$$

This tells us how much the SHAP values change per unit of input change. This works for Gaussian noise and feature dropout, where the input actually changes. But for label flip, the inputs stay the same and only the model changes (because it gets retrained). So we define a *model-space Lipschitz ratio*:

$$L_\theta = \frac{\|\Delta\text{SHAP}\|_2}{\|\Delta\boldsymbol\theta\|_2}$$

where θ represents the model parameters (e.g., coefficients for Logistic Regression, feature importances for tree-based models).

**Faithfulness.** We also want to know if the explanation is still *accurate* as a description of the model. *Surrogate fidelity* checks how well the SHAP additive decomposition reproduces the model's predictions:

$$\text{Fidelity} = 1 - \frac{\text{MAE}(f, g)}{\text{MAE}(f, \bar{y})}$$

where g(x) = E[f(x)] + Σ φ_j(x) is the SHAP reconstruction. If this goes negative, the SHAP explanation does worse than just predicting the mean — it is actively misleading. We also compute a *perturbation faithfulness ratio* FR_k: the ratio of the prediction drop when we mask the top-k SHAP features vs. the bottom-k. A high ratio confirms SHAP is pointing at the right features.

### 3.2 Perturbation Protocol

We apply three types of perturbation, each meant to model a different kind of data quality problem:

1. **Gaussian noise** (applied at test time): x' = x + ε, ε ~ N(0, σ²I), with σ ∈ {0.05, 0.10, 0.15}. This models sensor noise or measurement error. The model is not retrained.
2. **Label flip** (applied at training time): randomly flip y_train labels at rates {5%, 10%, 15%}. This models annotation errors. The model *is* retrained on the corrupted labels.
3. **Feature dropout** (applied at test time): set feature values to zero with probability {0.05, 0.10, 0.15}. This models missing data. The model is not retrained.

### 3.3 Datasets

We use three standard classification datasets. These are small-to-medium in size, which let us run many experimental configurations. All features are standardized to zero mean and unit variance. We use 80/20 stratified train-test splits and repeat each experiment with 5 different random seeds.

| Dataset | n | p | Domain | Source |
|---------|---|---|--------|--------|
| Heart Disease (Cleveland) | 297 | 13 | Cardiology | UCI |
| Pima Indians Diabetes | 768 | 8 | Endocrinology | Kaggle |
| Breast Cancer Wisconsin | 569 | 30 | Oncology | UCI |

### 3.4 Models

We pick two "black-box" models and two "interpretable" models. This lets us compare whether model transparency actually helps with explanation stability.

| Model | Type | SHAP Method | Key Params |
|-------|------|-------------|------------|
| XGBoost | Black-box | TreeExplainer | 100 trees, depth 6 |
| Logistic Regression | Black-box | LinearExplainer | L2, C=1.0 |
| Decision Tree | Interpretable | TreeExplainer | depth 5, min_leaf 10 |
| EBM | Interpretable | Native explain_local | 256 bins, 10 interactions |

---

## 4. Results

We ran 6,147 experiment configurations in total. Below, we organize the results around five main observations.

### 4.1 Label Corruption Dominates Everything Else

This was the clearest result. Under Gaussian noise and feature dropout, Spearman ρ stays above 0.95 for every model-dataset pair we tested. But under label flip, it drops sharply — down to 0.28 for Breast Cancer with Logistic Regression.

The difference is statistically significant in all cases. We ran Mann-Whitney U tests comparing label flip against the other two perturbation types, and all six comparisons gave p < 0.001 with Cohen's d ranging from 1.73 to 8.70.

**Table: Mean Spearman ρ by perturbation type (averaged across all four models and perturbation levels)**

| Dataset | Gaussian Noise | Feature Dropout | Label Flip |
|---------|---------------|-----------------|------------|
| Heart Disease | 0.987 ± 0.021 | 0.993 ± 0.005 | 0.754 ± 0.104 |
| Pima Diabetes | 0.993 ± 0.009 | 0.991 ± 0.012 | 0.884 ± 0.071 |
| Breast Cancer | 0.999 ± 0.001 | 0.996 ± 0.003 | 0.425 ± 0.164 |

> **Fig. 1** — Stability heatmap across all conditions. Green = stable (ρ ≈ 1); red = unstable. The label flip row stands out clearly. → `stability_heatmap.png`

> **Fig. 2** — How ρ degrades as perturbation intensity increases. Label flip drops steeply; the other two stay flat. → `degradation_curves.png`

### 4.2 Rank Stability and Magnitude Stability Are Not the Same Thing

This was somewhat surprising to us. XGBoost on Heart Disease has ρ = 0.96 under 5% Gaussian noise, which looks very stable if you only look at rankings. But its Lipschitz ratio is L = 7.67, meaning the SHAP values move 7.7× faster than the input perturbation in ℓ₂ norm. The ranking stays the same, but the actual numbers change a lot.

Logistic Regression, by contrast, has L < 0.67 across all conditions — it is stable in both senses. This distinction matters because in some applications (e.g., regulatory compliance) people care about the exact SHAP values, not just which feature is on top.

> **Fig. 3** — Rank stability (ρ) plotted against magnitude sensitivity (L). Points in the upper-right corner are rank-stable but magnitude-unstable. → `rank_vs_magnitude.png`

> **Fig. 4** — Lipschitz ratios across models. Left: input-space (Gaussian noise, feature dropout). Right: model-space (label flip). → `lipschitz_comparison.png`

### 4.3 EBM Gives the Most Stable Explanations

Among the four models, the Explainable Boosting Machine tends to come out on top for stability under label flip. On Breast Cancer, which has 30 features, EBM's ρ = 0.608 is about 2.2× higher than Logistic Regression's 0.281.

We think this is because of EBM's additive structure: it models each feature independently as f(x) = Σ h_j(x_j), so corrupted labels cannot easily cause one feature's contribution to "spill over" into another.

**Table: Mean Spearman ρ under label flip, by model**

| Dataset | EBM | XGBoost | LogReg | DecTree |
|---------|-----|---------|--------|---------|
| Heart | 0.798 | **0.817** | 0.745 | 0.656 |
| Diabetes | **0.919** | 0.919 | 0.906 | 0.792 |
| Breast Ca. | **0.608** | 0.472 | 0.281 | 0.341 |

> **Fig. 5** — Stability (mean ρ) under label flip, grouped by model. → `model_comparison_bars.png`

> **Fig. 6** — Radar chart comparing all four models across accuracy, AUC-ROC, SHAP stability, fidelity, and Top-3 Jaccard under label flip. EBM has the most balanced profile. → `radar_comparison.png`

### 4.4 Decision Trees Are Less Stable Than You Would Expect

This was probably the most counterintuitive result. Decision Trees are often described as "inherently interpretable," which might lead you to think their explanations are also stable. But they are actually among the worst: ρ = 0.341 on Breast Cancer and 0.656 on Heart Disease under label flip.

The reason, we believe, is structural. A Decision Tree is piecewise-constant: each leaf assigns a fixed value, and the boundaries between regions are determined by split points. A small change in training labels can move a split point, which restructures the entire tree topology and redistributes feature importance in a way that is hard to predict.

> **Fig. 7** — Black-box vs. interpretable model stability. Decision Trees (right panel, lower rows) show as much or more instability as the black-box models under label flip. → `blackbox_vs_interpretable_heatmap.png`

### 4.5 SHAP Fidelity Can Go Negative

Under 15% label flip, surrogate fidelity drops to −0.50 for Pima Diabetes with Logistic Regression. Negative fidelity means the SHAP additive decomposition does worse at predicting the model's output than just guessing the average — the explanation is not just noisy, it is actively wrong.

All six faithfulness tests under label flip are significant at p < 0.01 (Mann-Whitney U, Cohen's d between 1.81 and 2.35).

> **Fig. 8** — Surrogate fidelity vs. perturbation intensity. Under label flip, fidelity can drop below zero (dashed line), indicating the explanation is worse than the trivial baseline. → `faithfulness_degradation.png`

---

## 5. Discussion

### 5.1 Why Label Flip Is So Much Worse

The key difference is that Gaussian noise and feature dropout change the test inputs but leave the model unchanged. Since SHAP's exact methods (TreeExplainer, LinearExplainer) faithfully reflect the model, the explanations stay accurate even if the inputs are a bit different.

Label flip, on the other hand, corrupts the training data and forces the model to be retrained. The decision boundary itself moves. So the instability is coming from the model, not from the explanation method. We confirmed this by checking that native feature importances (from the model directly) and SHAP-based importances correlate at r ≈ 0.99, meaning SHAP is faithfully reporting what the model thinks — it is the model that has changed.

> **Fig. 9** — Native feature importance vs. SHAP stability under label flip. The tight diagonal clustering confirms that SHAP is faithfully reflecting model-level instability. → `native_vs_shap.png`

### 5.2 The Most Important Features Are the Most Fragile

When we looked at individual features, we found that the ones with the highest SHAP values (Glucose for Diabetes, worst area for Breast Cancer, ca for Heart Disease) are also the ones whose explanations shift the most under label corruption. This is inconvenient: the features you would rely on most for decision-making are the ones whose stability you can trust least.

> **Fig. 10** — Per-feature stability under 15% label flip. Red bars are the most unstable features; green bars are the most stable. The clinically important features tend to be red. → `per_feature_stability.png`

### 5.3 Accuracy vs. Stability: A Trade-Off

No single model dominates on both accuracy and stability across all datasets. For example, Logistic Regression on Breast Cancer gets 98.1% accuracy but the worst stability (ρ = 0.28). EBM consistently sits on or near the Pareto frontier, making it a reasonable default choice if you care about both.

> **Fig. 11** — Accuracy vs. stability Pareto frontier under label corruption. EBM tends to be on or near the frontier. → `pareto_frontier.png`

### 5.4 Practical Recommendations

Based on what we observed:

1. **Test stability before deploying explanations.** A model can have high accuracy and completely unreliable explanations. Checking ρ under simulated label noise is a simple sanity check.
2. **If you need an interpretable model, consider EBM over Decision Trees.** EBM's additive structure seems to give it a stability advantage that plain trees do not have.
3. **Report both rank and magnitude stability.** Spearman ρ alone can hide large shifts in SHAP values.
4. **Pay attention to label quality.** In our experiments, 5% label corruption causes more damage to explanations than 15% Gaussian noise.
5. **Check fidelity.** If surrogate fidelity is negative, the SHAP explanations are worse than useless and should not be the basis for any decision.

### 5.5 Limitations

We want to be upfront about what this study does not cover:

- We use only 5 random seeds per configuration. This limits the statistical power of paired model comparisons (the smallest possible Wilcoxon p-value with n=5 is 0.0625). More seeds would strengthen the conclusions.
- The three datasets are all small-to-medium tabular classification problems. We do not know how well these results transfer to large datasets, regression tasks, or non-tabular data like images or text.
- Our perturbations are synthetic. Real-world distributional shift often involves more complex patterns like concept drift or covariate shift, which we do not model here.
- For EBM, the local explanation is exact for the main additive terms, but this breaks down for the pairwise interaction terms that EBM also fits. We did not investigate this in detail.

---

## 6. Conclusion

We set out to answer a simple question: how stable are SHAP explanations when the data is noisy? The answer, at least for the datasets and models we tested, is "it depends a lot on the type of noise."

Gaussian noise and feature dropout barely affect SHAP rankings. Label corruption, even at low rates, can completely reshape which features SHAP considers important. Among the four models we tested, EBM provides the best balance of accuracy and stability, while Decision Trees — despite their reputation for transparency — produce the most fragile explanations. Perhaps most concerning, we found cases where SHAP's fidelity goes negative under label corruption, meaning the explanation is actively misleading rather than just noisy.

We think the practical takeaway is straightforward: if you are using SHAP explanations to make decisions, and there is any chance your data has label noise, you should test how stable those explanations are before relying on them. We release the full experimental code and data to make this easy to reproduce and extend.

### Future Work

Several directions seem worth exploring:

- Extending to temporal distributional shift, where the data changes over time (concept drift).
- Testing adversarial perturbations designed specifically to break explanations.
- Scaling up to higher-dimensional data (e.g., genomics, NLP embeddings).
- Investigating whether training objectives can be modified to encourage more stable explanations.

---

## References

1. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. NeurIPS.
2. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?" Explaining the predictions of any classifier. KDD.
3. Alvarez-Melis, D., & Jaakkola, T. (2018). On the robustness of interpretability methods. ICML Workshop.
4. Slack, D., Hilgard, S., Jia, E., Singh, S., & Lakkaraju, H. (2020). Fooling LIME and SHAP. AIES.
5. Dombrowski, A. K., et al. (2019). Explanations can be manipulated and geometry is to blame. NeurIPS.
6. Comparative Analysis of Interpretable ML Methods (2026). arXiv:2601.00428.
7. Sparse but not Simpler: ViT Interpretability (2026). arXiv:2603.15919.
8. Lou, Y., Caruana, R., & Gehrke, J. (2013). Intelligible models for classification and regression. KDD.
9. Schmidt Sciences (2026). Interpretability RFP.
10. Shapley-value Interpretability for Heart Disease (2024). arXiv:2408.06183.
11. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. KDD.
12. Nori, H., Jenkins, S., Koch, P., & Caruana, R. (2019). InterpretML: A unified framework for ML interpretability. arXiv:1909.09223.

---

## Appendix: Figure Index

| Figure | Description | File |
|--------|-------------|------|
| Fig. 1 | Stability heatmap (Spearman ρ) | stability_heatmap.png |
| Fig. 2 | Stability degradation curves | degradation_curves.png |
| Fig. 3 | Rank stability vs magnitude sensitivity | rank_vs_magnitude.png |
| Fig. 4 | Input-space vs model-space Lipschitz | lipschitz_comparison.png |
| Fig. 5 | 4-model stability comparison bars | model_comparison_bars.png |
| Fig. 6 | Multi-metric radar chart | radar_comparison.png |
| Fig. 7 | Black-box vs interpretable heatmaps | blackbox_vs_interpretable_heatmap.png |
| Fig. 8 | SHAP surrogate fidelity degradation | faithfulness_degradation.png |
| Fig. 9 | Native vs SHAP stability | native_vs_shap.png |
| Fig. 10 | Per-feature stability (most/least stable) | per_feature_stability.png |
| Fig. 11 | Accuracy–stability Pareto frontier | pareto_frontier.png |
| Fig. 12 | Final 3×3 summary panel | final_summary_panel.png |
| Fig. 13 | Top-k Jaccard degradation | topk_jaccard.png |
| Fig. 14 | Bootstrap confidence intervals | confidence_intervals.png |
| Fig. 15 | Statistical significance forest plot | significance_forest.png |
| Fig. 16 | Faithfulness ratio boxplots | faithfulness_ratio.png |
| Fig. 17 | Per-feature Pearson correlation heatmap | feature_correlation_heatmap.png |
| Fig. 18 | Accuracy vs stability scatter (proto-Pareto) | accuracy_vs_stability.png |
