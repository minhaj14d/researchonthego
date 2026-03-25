"""
Train baseline models and compute SHAP values.
Supports: XGBoost, Logistic Regression, Decision Tree, EBM.
"""
import numpy as np
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from interpret.glassbox import ExplainableBoostingClassifier

from config import MODELS


def build_model(model_name, seed=42):
    params = {**MODELS[model_name], "random_state": seed}
    if model_name == "xgboost":
        return XGBClassifier(**params)
    elif model_name == "logistic_regression":
        return LogisticRegression(**params)
    elif model_name == "decision_tree":
        return DecisionTreeClassifier(**params)
    elif model_name == "ebm":
        return ExplainableBoostingClassifier(**params)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_and_evaluate(model, X_train, y_train, X_test, y_test, feature_names=None):
    import pandas as pd
    if feature_names is not None and isinstance(model, ExplainableBoostingClassifier):
        X_tr_df = pd.DataFrame(X_train, columns=feature_names)
        X_te_df = pd.DataFrame(X_test, columns=feature_names)
        model.fit(X_tr_df, y_train)
        y_pred = model.predict(X_te_df)
        y_prob = model.predict_proba(X_te_df)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "auc_roc": roc_auc_score(y_test, y_prob),
    }
    return model, metrics


def compute_shap_values(model, X_train, X_test, model_name, feature_names=None):
    """Compute SHAP values using the appropriate explainer for the model type."""
    if model_name == "ebm":
        return _ebm_local_importances(model, X_test, feature_names)

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
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    return shap_values


def _ebm_local_importances(model, X_test, feature_names=None):
    """
    Extract per-sample local explanations from EBM's native explain_local.
    EBM is inherently additive: f(x) = intercept + Σ h_j(x_j),
    so local explanations are exact and equivalent to SHAP for GAMs.
    """
    import pandas as pd
    if feature_names is not None:
        X_df = pd.DataFrame(X_test, columns=feature_names)
    else:
        X_df = pd.DataFrame(X_test)

    local_exp = model.explain_local(X_df)
    n_samples = X_test.shape[0]
    n_features = X_test.shape[1]
    shap_values = np.zeros((n_samples, n_features))
    for i in range(n_samples):
        sample_data = local_exp.data(i)
        scores = sample_data["scores"]
        for j in range(min(n_features, len(scores))):
            shap_values[i, j] = scores[j]
    return shap_values


def compute_native_importance(model, X_test, model_name):
    """
    Extract native (non-SHAP) feature importances for interpretable models.
    Returns array of shape (n_features,) or None if not applicable.
    """
    if model_name == "decision_tree":
        return model.feature_importances_
    elif model_name == "ebm":
        try:
            ebm_global = model.explain_global()
            data = ebm_global.data()
            scores = data["scores"]
            n_feat = X_test.shape[1]
            return np.array([np.mean(np.abs(s)) for s in scores[:n_feat]])
        except Exception:
            return None
    elif model_name == "logistic_regression":
        return np.abs(model.coef_.flatten())
    elif model_name == "xgboost":
        return model.feature_importances_
    return None


def get_feature_importance(shap_values):
    """Mean absolute SHAP value per feature."""
    return np.mean(np.abs(shap_values), axis=0)
