"""
Case Study 1: Hospital Readmission Prediction
Logistic Regression with L2 regularization.

Expected input CSV columns (example):
readmitted_30d, age, heart_rate, systolic_bp, glucose, prior_visits,
diagnosis_code, gender

The script also works with a small built-in demonstration dataset if no CSV
is supplied. For a real submission, replace the demo data with your dataset.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_demo_data(n=1500, seed=42):
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 90, n)
    heart_rate = rng.normal(78, 12, n).clip(45, 140)
    systolic_bp = rng.normal(125, 18, n).clip(80, 210)
    glucose = rng.normal(115, 35, n).clip(55, 350)
    prior_visits = rng.poisson(2, n)
    diagnosis = rng.choice(["A", "B", "C", "D"], n)
    gender = rng.choice(["M", "F"], n)

    score = (
        -4.0 + 0.035 * age + 0.20 * prior_visits
        + 0.012 * (glucose - 100)
        + 0.010 * np.abs(systolic_bp - 120)
        + 0.008 * (heart_rate - 75)
        + (diagnosis == "D") * 0.7
    )
    p = 1 / (1 + np.exp(-score))
    y = rng.binomial(1, p)

    return pd.DataFrame({
        "age": age, "heart_rate": heart_rate, "systolic_bp": systolic_bp,
        "glucose": glucose, "prior_visits": prior_visits,
        "diagnosis_code": diagnosis, "gender": gender,
        "readmitted_30d": y
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, help="Path to patient CSV")
    args = parser.parse_args()

    if args.data:
        df = pd.read_csv(args.data)
        target = "readmitted_30d"
        if target not in df.columns:
            raise ValueError(f"CSV must contain '{target}' as the target column.")
    else:
        print("No dataset supplied. Running with a synthetic demonstration dataset.")
        df = make_demo_data()

    X = df.drop(columns=["readmitted_30d"])
    y = df["readmitted_30d"]

    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", categorical_pipe, categorical)
    ])

    # L2 regularization is selected explicitly.
    model = LogisticRegression(
        penalty="l2", C=1.0, solver="liblinear", max_iter=1000
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    pipeline.fit(X_train, y_train)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.50).astype(int)

    auc = roc_auc_score(y_test, probabilities)
    cm = confusion_matrix(y_test, predictions)

    print("\n=== Hospital Readmission Prediction ===")
    print(f"ROC-AUC: {auc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, zero_division=0))

    fpr, tpr, _ = roc_curve(y_test, probabilities)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - 30-Day Hospital Readmission")
    plt.legend()
    plt.tight_layout()
    plt.savefig("hospital_roc_curve.png", dpi=150)
    plt.close()

    ConfusionMatrixDisplay(cm).plot()
    plt.title("Hospital Readmission Confusion Matrix")
    plt.tight_layout()
    plt.savefig("hospital_confusion_matrix.png", dpi=150)
    plt.close()

    print("\nClinical interpretation:")
    print("- False Negative: patient is predicted low-risk but is readmitted.")
    print("- False Positive: patient is predicted high-risk but is not readmitted.")
    print("- In many clinical screening settings, false negatives can be costly")
    print("  because a missed high-risk patient may not receive extra follow-up.")
    print("- False positives can consume staff/resources and cause unnecessary")
    print("  follow-up. The appropriate threshold should therefore reflect clinical cost.")


if __name__ == "__main__":
    main()
