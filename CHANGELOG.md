# Changelog

> Audit trail — one-line summary per file modification.

| Timestamp | File | Summary |
|-----------|------|---------|
| 2026-03-25 00:00 | CHANGELOG.md | Created changelog for audit trail |
| 2026-03-25 00:00 | research-notes/ (all) | Scaffolded full folder structure per .cursorrules |
| 2026-03-25 00:01 | research-notes/README.md | Created project overview with 5-day plan and navigation |
| 2026-03-25 00:01 | research-notes/niches/niche-1-explanation-stability.md | Niche 1 scoping: gap, math framework, task checklist |
| 2026-03-25 00:01 | research-notes/methods/datasets.md | Documented 3 UCI datasets with preprocessing notes |
| 2026-03-25 00:01 | research-notes/methods/metrics.md | Formalized 4 stability metrics with formulas |
| 2026-03-25 00:01 | research-notes/methods/models.md | Defined XGBoost, LogReg, DecisionTree, EBM configs |
| 2026-03-25 00:01 | research-notes/literature/papers.md | Added 5 key paper summaries with gap analysis |
| 2026-03-25 00:01 | research-notes/decisions/decisions-log.md | Logged DEC-001 through DEC-003 |
| 2026-03-25 00:01 | research-notes/drafts/paper-outline.md | Created working paper structure with 6 sections |
| 2026-03-25 00:02 | experiments/niche1/config.py | Central config: seeds, datasets, perturbation params |
| 2026-03-25 00:02 | experiments/niche1/data_loader.py | Dataset loaders for Heart, Diabetes, Breast Cancer |
| 2026-03-25 00:02 | experiments/niche1/train_baselines.py | Model training, evaluation, SHAP computation |
| 2026-03-25 00:02 | experiments/niche1/perturbations.py | Gaussian noise, label flip, feature dropout functions |
| 2026-03-25 00:02 | experiments/niche1/stability_metrics.py | Spearman, Lipschitz, Top-k Jaccard implementations |
| 2026-03-25 00:02 | experiments/niche1/run_experiment.py | Main experiment runner: full perturbation sweep |
| 2026-03-25 00:02 | experiments/niche1/requirements.txt | Python dependencies with version constraints |
| 2026-03-25 00:03 | research-notes/daily-log/2026-03-25.md | Day 1 experiment log: EXP-001, EXP-002 |
| 2026-03-25 00:04 | experiments/niche1/config.py | Removed deprecated use_label_encoder and penalty params |
| 2026-03-25 00:04 | experiments/niche1/results/stability_results.csv | 300 experiment records (auto-generated) |
| 2026-03-25 00:04 | experiments/niche1/results/feature_importances.json | Per-experiment SHAP importances (auto-generated) |
| 2026-03-25 00:04 | experiments/niche1/results/summary_stats.csv | Aggregated stability statistics (auto-generated) |
| 2026-03-25 00:05 | research-notes/daily-log/2026-03-25.md | Added EXP-002 results and key findings |
| 2026-03-25 00:05 | research-notes/niches/niche-1-explanation-stability.md | Updated task checklist with Day 1 completions |
| 2026-03-26 00:00 | experiments/niche1/feature_analysis.py | Per-feature stability + model-space Lipschitz script |
| 2026-03-26 00:00 | experiments/niche1/visualize.py | 7-plot visualization suite for publication figures |
| 2026-03-26 00:01 | experiments/niche1/config.py | Removed deprecated use_label_encoder and penalty params |
| 2026-03-26 00:01 | experiments/niche1/results/per_feature_stability.csv | 4590 per-feature stability records (auto-generated) |
| 2026-03-26 00:01 | experiments/niche1/results/model_lipschitz.csv | 90 model-space Lipschitz records (auto-generated) |
| 2026-03-26 00:02 | experiments/niche1/plots/ (7 files) | All publication-quality plots generated |
| 2026-03-26 00:03 | research-notes/daily-log/2026-03-26.md | Day 2 experiment log: EXP-003, EXP-004 |
| 2026-03-27 00:00 | experiments/niche1/faithfulness.py | Surrogate fidelity + perturbation faithfulness analysis |
| 2026-03-27 00:00 | experiments/niche1/statistical_tests.py | Wilcoxon, Mann-Whitney, Cohen's d, Bootstrap CIs |
| 2026-03-27 00:00 | experiments/niche1/visualize_day3.py | 4 Day 3 plots: faithfulness, forest, CI, ratio |
| 2026-03-27 00:01 | experiments/niche1/results/faithfulness_results.csv | 300 faithfulness records (auto-generated) |
| 2026-03-27 00:01 | experiments/niche1/results/statistical_tests.csv | 51 significance test records (auto-generated) |
| 2026-03-27 00:01 | experiments/niche1/results/confidence_intervals.csv | 216 bootstrap CI records (auto-generated) |
| 2026-03-27 00:02 | experiments/niche1/plots/ (4 new) | faithfulness_degradation, faithfulness_ratio, significance_forest, confidence_intervals |
| 2026-03-27 00:03 | research-notes/methods/metrics.md | Added metrics 5–7, statistical testing section |
| 2026-03-27 00:03 | research-notes/niches/niche-1-explanation-stability.md | Updated task checklist with Day 3 completions |
| 2026-03-27 00:04 | research-notes/daily-log/2026-03-27.md | Day 3 experiment log: EXP-005, EXP-006 |
| 2026-03-28 00:00 | experiments/niche1/config.py | Added EBM config + ALL_MODELS / INTERPRETABLE_MODELS constants |
| 2026-03-28 00:00 | experiments/niche1/train_baselines.py | Added EBM builder, native explain_local, 3D SHAP handling |
| 2026-03-28 00:00 | experiments/niche1/run_day4.py | Full 4-model experiment runner with faithfulness + native stability |
| 2026-03-28 00:00 | experiments/niche1/visualize_day4.py | 5 comparison plots: bars, Pareto, radar, native-vs-SHAP, heatmaps |
| 2026-03-28 00:01 | experiments/niche1/requirements.txt | Added interpret package |
| 2026-03-28 00:02 | experiments/niche1/results/day4_all_models.csv | 600 records across 4 models (auto-generated) |
| 2026-03-28 00:02 | experiments/niche1/plots/ (5 new) | model_comparison_bars, accuracy_vs_stability, radar, native_vs_shap, heatmap |
| 2026-03-28 00:03 | research-notes/daily-log/2026-03-28.md | Day 4 experiment log: EXP-007, EXP-008 |
| 2026-03-28 00:03 | research-notes/niches/niche-1-explanation-stability.md | Updated task checklist with Day 4 completions |
| 2026-03-29 00:00 | experiments/niche1/run_day5.py | Final analysis: Pareto frontier, summary tables, statistics |
| 2026-03-29 00:01 | experiments/niche1/results/table1_summary.csv | Publication Table 1: all models × datasets × metrics |
| 2026-03-29 00:01 | experiments/niche1/results/table2_perturbation.csv | Publication Table 2: ρ by perturbation type |
| 2026-03-29 00:01 | experiments/niche1/plots/pareto_frontier.png | Formal Pareto frontier (3 panels) |
| 2026-03-29 00:01 | experiments/niche1/plots/final_summary_panel.png | 3×3 summary panel (datasets × metrics) |
| 2026-03-29 00:02 | research-notes/drafts/paper-outline.md | Full paper draft: 6 sections, ~3500 words, 10 references |
| 2026-03-29 00:03 | research-notes/daily-log/2026-03-29.md | Day 5 experiment log: EXP-009, EXP-010, final summary |
| 2026-03-29 00:03 | research-notes/niches/niche-1-explanation-stability.md | All tasks marked complete |
| 2026-03-25 12:00 | latex/main.tex | Overleaf-ready LaTeX paper: 6 sections, 18 figures, 5 tables, appendix |
| 2026-03-25 12:00 | latex/references.bib | BibTeX bibliography with 16 entries |
| 2026-03-25 12:00 | latex/figures/*.png | Copied all 18 publication plots into LaTeX figures directory |
| 2026-03-25 12:30 | latex/main_expert_backup.tex | Backup of expert-tone version before undergrad rewrite |
| 2026-03-25 12:30 | latex/main.tex | Rewritten in undergraduate applied-math tone: less jargon, more direct |
