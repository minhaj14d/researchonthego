"""
Dataset loading and preprocessing for Niche 1 experiments.
"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import TEST_SIZE


def load_heart_disease():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    columns = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
    ]
    df = pd.read_csv(url, header=None, names=columns, na_values="?")
    df.dropna(inplace=True)
    df["target"] = (df["target"] > 0).astype(int)
    X = df.drop("target", axis=1).values.astype(float)
    y = df["target"].values
    feature_names = [c for c in columns if c != "target"]
    return X, y, feature_names, "Heart Disease"


def load_pima_diabetes():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    columns = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigree", "Age", "Outcome"
    ]
    df = pd.read_csv(url, header=None, names=columns)
    zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[zero_cols] = df[zero_cols].replace(0, np.nan)
    df.fillna(df.median(), inplace=True)
    X = df.drop("Outcome", axis=1).values.astype(float)
    y = df["Outcome"].values
    feature_names = [c for c in columns if c != "Outcome"]
    return X, y, feature_names, "Pima Diabetes"


def load_breast_cancer_data():
    data = load_breast_cancer()
    X = data.data.astype(float)
    y = data.target
    feature_names = list(data.feature_names)
    return X, y, feature_names, "Breast Cancer"


LOADERS = {
    "heart": load_heart_disease,
    "diabetes": load_pima_diabetes,
    "breast_cancer": load_breast_cancer_data,
}


def get_dataset(name, seed=42):
    """Load, scale, and split a dataset. Returns dict with all components."""
    loader = LOADERS[name]
    X, y, feature_names, display_name = loader()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=TEST_SIZE, random_state=seed, stratify=y
    )
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_names,
        "display_name": display_name,
        "scaler": scaler,
    }
