"""
Perturbation functions for testing explanation stability.
Three types: Gaussian noise, label flipping, feature dropout.
"""
import numpy as np


def add_gaussian_noise(X, level, rng):
    """Add Gaussian noise scaled by `level` * feature std."""
    noise = rng.normal(0, level, size=X.shape)
    return X + noise


def flip_labels(y, level, rng):
    """Randomly flip `level` fraction of labels."""
    y_perturbed = y.copy()
    n_flip = int(len(y) * level)
    flip_idx = rng.choice(len(y), size=n_flip, replace=False)
    y_perturbed[flip_idx] = 1 - y_perturbed[flip_idx]
    return y_perturbed


def dropout_features(X, level, rng):
    """Zero out `level` fraction of feature entries (element-wise)."""
    mask = rng.random(size=X.shape) < level
    X_perturbed = X.copy()
    X_perturbed[mask] = 0.0
    return X_perturbed


def apply_perturbation(X_train, y_train, X_test, perturbation_type, level, seed):
    """
    Apply a perturbation and return modified data.
    
    For 'gaussian' and 'feature_dropout': perturb X_test (explanation-time shift).
    For 'label_flip': perturb y_train (training-time shift, requires retraining).
    
    Returns (X_train', y_train', X_test', needs_retrain).
    """
    rng = np.random.default_rng(seed)

    if perturbation_type == "gaussian":
        X_test_p = add_gaussian_noise(X_test, level, rng)
        return X_train.copy(), y_train.copy(), X_test_p, False

    elif perturbation_type == "label_flip":
        y_train_p = flip_labels(y_train, level, rng)
        return X_train.copy(), y_train_p, X_test.copy(), True

    elif perturbation_type == "feature_dropout":
        X_test_p = dropout_features(X_test, level, rng)
        return X_train.copy(), y_train.copy(), X_test_p, False

    else:
        raise ValueError(f"Unknown perturbation: {perturbation_type}")
