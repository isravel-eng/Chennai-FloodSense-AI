"""
evaluation.py
-------------
Loads the SAVED model2 artifact (models/flood_model.pkl) and reports a
full evaluation report on a fresh stratified holdout: confusion matrix,
precision/recall/F1 at the best-F1 threshold, and PR-AUC. Use this to
sanity-check the shipped model.pkl after it's been created by
02_evaluation.py (or retrieved from a fresh clone) - it does NOT retrain
anything.

Run standalone: python model_2_flood/evaluation.py
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessing import FEATURE_ORDER, TARGET_COL  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = ROOT / "data" / "processed" / "model2_features.csv"
MODELS_DIR = ROOT / "models"
RANDOM_STATE = 42


def main():
    model_path = MODELS_DIR / "flood_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} not found - run model_2_flood/02_evaluation.py first."
        )
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    df = pd.read_csv(FEATURES_PATH)
    df = df.sort_values("date")  # must match the row order used in 02_evaluation.py
    X = df[FEATURE_ORDER]
    y = df[TARGET_COL]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    proba = model.predict_proba(X_test)[:, 1]
    pr_auc = average_precision_score(y_test, proba)

    precision, recall, thresholds = precision_recall_curve(y_test, proba)
    f1 = np.divide(
        2 * precision * recall, precision + recall,
        out=np.zeros_like(precision), where=(precision + recall) != 0,
    )
    best_idx = int(np.argmax(f1))
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    y_pred = (proba >= best_threshold).astype(int)

    print(f"Model: {type(model).__name__}")
    print(f"PR-AUC (average precision): {pr_auc:.4f}")
    print(f"Best-F1 threshold: {best_threshold:.3f}")
    print("\nConfusion matrix (rows=actual, cols=predicted) [0=no-flood, 1=flood]:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["no_flood", "flood"], digits=3))

    print(
        "Interpretation note: flood_occurred_documented is a rare event "
        f"({int(y.sum())}/{len(y)} rows citywide across 1993-2023), so "
        "precision/recall at this threshold matter far more than raw "
        "accuracy - a model that always predicts 'no flood' would already "
        f"score {100*(1 - y.mean()):.1f}% accuracy while being useless."
    )


if __name__ == "__main__":
    main()
