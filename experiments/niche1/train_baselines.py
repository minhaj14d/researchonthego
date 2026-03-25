"""
Train baseline models and compute SHAP values.
"""
import numpy as np
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

from config import MODELS


def build_model(model_name, seed=42):
    params = {**MODELS[model_name], "random_state": seed}
    if model_name == "xgboost":
        return XGBClassifier(**params)
    elif model_name == "logistic_regression":
        return LogisticRegression(**params)
    elif model_name == "decision_tree":
        return DecisionTreeClassifier(**params)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "auc_roc": roc_auc_score(y_test, y_prob),
    }
    return model, metrics


def compute_shap_values(model, X_train, X_test, model_name):
    """Compute SHAP values using the appropriate explainer for the model type."""
    if model_name == "xgboost":
        explainer = shap.TreeExplainer(model)
    elif model_name == "logistic_regression":
        explainer = shap.LinearExplainer(model, X_train)
    elif model_name == "decision_tree":
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.KernelExplainer(model.predict_proba, X_train[:50])

    shap_values = explainer.shap_values(X_test)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    return shap_values


def get_feature_importance(shap_values):
    """Mean absolute SHAP value per feature."""
    return np.mean(np.abs(shap_values), axis=0)
