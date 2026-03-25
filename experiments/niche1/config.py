"""
Central configuration for Niche 1: Explanation Stability experiments.
"""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ROOT_DIR / "results"
PLOTS_DIR = ROOT_DIR / "plots"

RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

RANDOM_SEEDS = [0, 1, 2, 3, 4]
BASE_SEED = 42
TEST_SIZE = 0.2

DATASETS = ["heart", "diabetes", "breast_cancer"]

PERTURBATION_TYPES = ["gaussian", "label_flip", "feature_dropout"]
PERTURBATION_LEVELS = [0.05, 0.10, 0.15]

TOP_K_VALUES = [3, 5]

MODELS = {
    "xgboost": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "logloss",
    },
    "logistic_regression": {
        "C": 1.0,
        "l1_ratio": 0,
        "solver": "lbfgs",
        "max_iter": 1000,
    },
    "decision_tree": {
        "max_depth": 5,
        "min_samples_leaf": 10,
    },
    "ebm": {
        "max_bins": 256,
        "interactions": 10,
        "outer_bags": 8,
        "inner_bags": 0,
    },
}

ALL_MODELS = ["xgboost", "logistic_regression", "decision_tree", "ebm"]
BLACK_BOX_MODELS = ["xgboost", "logistic_regression"]
INTERPRETABLE_MODELS = ["decision_tree", "ebm"]
