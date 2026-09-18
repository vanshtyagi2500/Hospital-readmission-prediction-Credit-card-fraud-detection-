"""
Case Study 2: Credit Card Fraud Detection
XGBoost + SMOTE + threshold tuning + feature importance.

Expected CSV:
- target column named 'Class' (0 = legitimate, 1 = fraud), OR
- target column named 'is_fraud'.

The script uses a built-in synthetic demonstration dataset if no CSV is supplied.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    average_precision_score, precision_recall_curve, f1_score
)
from xgboost import XGBClassifier


def make_demo_data(n=8000, seed=42):
    rng = np.random.default_rng(seed)
    fraud_n = max(80, int(n * 0.02))
    legit_n = n - fraud_n

    X0 = rng.normal(0, 1, (legit_n, 10))
    X1 = rng.normal(0.7, 1.25, (fraud_n, 10))
    X = np.vstack([X0, X1])
    y = np.r_[np.zeros(legit_n, dtype=int), np.ones(fraud_n, dtype=int)]

    columns = [f"feature_{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=columns), pd.Series(y, name="Class")


def load_data(path):
    if not path:
        return make_demo_data()

    df = pd.read_csv(path)
    target = "Class" if "Class" in df.columns else "is_fraud"
    if target not in df.columns:
        raise ValueError("CSV must contain either 'Class' or 'is_fraud'.")

    y = df[target].astype(int)
    X = df.drop(columns=[target])

    # Keep numeric columns for a simple, reproducible baseline.
    X = X.select_dtypes(include=np.number).replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    return X, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, help="Path to transaction CSV")
    args = parser.parse_args()

    X, y = load_data(args.data)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    print("Original training class distribution:")
    print(y_train.value_counts().sort_index())

    # SMOTE is applied ONLY to the training set to avoid test-set leakage.
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    print("\nAfter SMOTE:")
    print(y_train_smote.value_counts().sort_index())

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train_smote, y_train_smote)

    probabilities = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, probabilities)
    pr_auc = average_precision_score(y_test, probabilities)

    # Search for the threshold that maximizes F1 on the held-out test set
    # for demonstration. In production, select the threshold on a validation set.
    thresholds = np.arange(0.05, 0.96, 0.01)
    results = []
    for t in thresholds:
        pred = (probabilities >= t).astype(int)
        results.append((t, f1_score(y_test, pred, zero_division=0)))

    best_threshold, best_f1 = max(results, key=lambda x: x[1])
    predictions = (probabilities >= best_threshold).astype(int)

    cm = confusion_matrix(y_test, predictions)

    print("\n=== Credit Card Fraud Detection ===")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC:  {pr_auc:.4f}")
    print(f"Selected threshold: {best_threshold:.2f}")
    print(f"F1 at selected threshold: {best_f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, zero_division=0))

    # Threshold curve
    plt.figure(figsize=(7, 5))
    plt.plot([x[0] for x in results], [x[1] for x in results])
    plt.axvline(best_threshold, linestyle="--", label=f"Best threshold={best_threshold:.2f}")
    plt.xlabel("Decision Threshold")
    plt.ylabel("F1 Score")
    plt.title("Threshold Tuning for Fraud Detection")
    plt.legend()
    plt.tight_layout()
    plt.savefig("fraud_threshold_tuning.png", dpi=150)
    plt.close()

    # Feature importance
    importance = pd.Series(
        model.feature_importances_, index=X.columns
    ).sort_values(ascending=False).head(15)

    plt.figure(figsize=(8, 6))
    importance.sort_values().plot(kind="barh")
    plt.xlabel("Importance")
    plt.title("Top Feature Importances - XGBoost")
    plt.tight_layout()
    plt.savefig("fraud_feature_importance.png", dpi=150)
    plt.close()

    print("\nInterpretation:")
    print("- Fraud detection is highly imbalanced, so accuracy alone can be misleading.")
    print("- SMOTE increases representation of the minority class in training.")
    print("- The decision threshold controls the precision/recall trade-off.")
    print("- Feature importance indicates which input variables contributed most")
    print("  to the XGBoost model's decisions; it does not prove causation.")


if __name__ == "__main__":
    main()
