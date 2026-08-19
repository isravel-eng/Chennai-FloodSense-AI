"""
02_evaluation.py
-----------------
Trains candidate Model 2 classifiers and selects the winner by PR-AUC
(precision-recall AUC), which is the appropriate metric here because
flood_occurred_documented is a rare-event / heavily imbalanced target
(176 positive rows out of 16,554 - ~1.1%). ROC-AUC would look
misleadingly good on a class this imbalanced; PR-AUC does not.

Candidates compared:
  - RandomForestClassifier (class_weight="balanced")
  - XGBClassifier (scale_pos_weight tuned for the imbalance)

Winner is saved to models/flood_model.pkl (pickled model object).
Run standalone: python model_2_flood/02_evaluation.py

Prerequisite: model_2_flood/preprocessing.py must have already been run
(this script imports FEATURE_ORDER from it).
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessing import FEATURE_ORDER, TARGET_COL  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = ROOT / "data" / "processed" / "model2_features.csv"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


def load_split():
    df = pd.read_csv(FEATURES_PATH)
    df = df.sort_values("date")
    X = df[FEATURE_ORDER]
    y = df[TARGET_COL]
    # Stratified split so the rare positive class is represented in both
    # train and test sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    return X_train, X_test, y_train, y_test


def evaluate(model, X_test, y_test, label):
    proba = model.predict_proba(X_test)[:, 1]
    pr_auc = average_precision_score(y_test, proba)
    precision, recall, thresholds = precision_recall_curve(y_test, proba)
    # Best F1 threshold on this holdout, for reference only.
    f1_scores = np.divide(
        2 * precision * recall, precision + recall,
        out=np.zeros_like(precision), where=(precision + recall) != 0,
    )
    best_idx = int(np.argmax(f1_scores))
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    print(f"{label}: PR-AUC = {pr_auc:.4f}  (best-F1 threshold ~= {best_threshold:.3f})")
    return pr_auc


def main():
    X_train, X_test, y_train, y_test = load_split()
    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    print(f"Train rows: {len(X_train)}  (positives: {n_pos}, negatives: {n_neg})")
    print(f"Test rows : {len(X_test)}  (positives: {int(y_test.sum())})")

    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=8,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_pr_auc = evaluate(rf, X_test, y_test, "RandomForest")

    scale_pos_weight = n_neg / max(n_pos, 1)
    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    xgb.fit(X_train, y_train)
    xgb_pr_auc = evaluate(xgb, X_test, y_test, "XGBoost")

    if xgb_pr_auc >= rf_pr_auc:
        winner, winner_name, winner_pr_auc = xgb, "XGBoost", xgb_pr_auc
    else:
        winner, winner_name, winner_pr_auc = rf, "RandomForest", rf_pr_auc

    print(f"\nSelected model: {winner_name} (PR-AUC = {winner_pr_auc:.4f})")

    model_path = MODELS_DIR / "flood_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(winner, f)
    print(f"Saved {model_path}")

    # Persist selection metadata for evaluation.py / README reporting.
    meta_path = MODELS_DIR / "flood_model_selection.pkl"
    with open(meta_path, "wb") as f:
        pickle.dump({
            "selected_model": winner_name,
            "pr_auc": round(float(winner_pr_auc), 4),
            "candidates": {
                "RandomForest": round(float(rf_pr_auc), 4),
                "XGBoost": round(float(xgb_pr_auc), 4),
            },
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_positive_train": n_pos,
            "n_positive_test": int(y_test.sum()),
        }, f)
    print(f"Saved {meta_path}")


if __name__ == "__main__":
    main()
