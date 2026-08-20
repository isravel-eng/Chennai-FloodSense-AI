"""Evaluate the saved Model 2 on the untouched 2022-2023 chronological test set."""

import pickle
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix

from preprocessing import FEATURE_ORDER, TARGET_COL

ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = ROOT / "data" / "processed" / "model2_features.csv"
MODELS_DIR = ROOT / "models"
TEST_START = "2022-01-01"


def main():
    with open(MODELS_DIR / "flood_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(MODELS_DIR / "flood_model_selection.pkl", "rb") as f:
        metadata = pickle.load(f)

    threshold = float(metadata["decision_threshold"])
    df = pd.read_csv(FEATURES_PATH, parse_dates=["date"])
    df = df[df["date"] >= pd.Timestamp(TEST_START)].sort_values("date")
    X_test = df[FEATURE_ORDER]
    y_test = df[TARGET_COL]

    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= threshold).astype(int)

    print(f"Model: {type(model).__name__}")
    print(f"Test period: {df.date.min().date()} -> {df.date.max().date()}")
    print(f"Test rows: {len(df)} (flood positives: {int(y_test.sum())})")
    print(f"PR-AUC: {average_precision_score(y_test, proba):.4f}")
    print(f"Decision threshold (validation): {threshold:.3f}")
    print("\nConfusion matrix [rows=actual, cols=predicted]:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["no_flood", "flood"], digits=3, zero_division=0))


if __name__ == "__main__":
    main()
