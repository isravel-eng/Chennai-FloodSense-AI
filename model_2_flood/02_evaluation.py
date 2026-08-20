"""
02_evaluation.py
-----------------
Train and evaluate Model 2 with a chronological split:

  TRAIN       1993-2017
  VALIDATION  2018-2021
  TEST        2022-2023

The validation period selects both the model configuration and the decision
threshold. The final test period remains untouched until the final report.

Candidates:
  - Random Forest baseline
  - XGBoost baseline
  - XGBoost tuned configuration

Selection metric: validation PR-AUC, appropriate for the rare flood class.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessing import FEATURE_ORDER, TARGET_COL  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = ROOT / "data" / "processed" / "model2_features.csv"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TRAIN_END = "2017-12-31"
VALIDATION_END = "2021-12-31"


def load_temporal_split():
    df = pd.read_csv(FEATURES_PATH, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    train = df[df["date"] <= pd.Timestamp(TRAIN_END)]
    validation = df[(df["date"] > pd.Timestamp(TRAIN_END)) & (df["date"] <= pd.Timestamp(VALIDATION_END))]
    test = df[df["date"] > pd.Timestamp(VALIDATION_END)]

    for name, part in (("train", train), ("validation", validation), ("test", test)):
        if part.empty or part[TARGET_COL].sum() == 0:
            raise ValueError(f"Temporal {name} partition is empty or has no positive flood events.")
    return train, validation, test


def xy(df):
    return df[FEATURE_ORDER], df[TARGET_COL]


def best_f1_threshold(y_true, proba):
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )
    idx = int(np.argmax(f1))
    threshold = float(thresholds[idx]) if idx < len(thresholds) else 0.5
    return threshold, float(f1[idx])


def evaluate(model, X, y):
    proba = model.predict_proba(X)[:, 1]
    pr_auc = average_precision_score(y, proba)
    threshold, f1 = best_f1_threshold(y, proba)
    return float(pr_auc), float(threshold), float(f1)


def make_models(scale_pos_weight: float):
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=400,
            max_depth=8,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost_baseline": XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost_tuned": XGBClassifier(
            n_estimators=600,
            max_depth=4,
            learning_rate=0.03,
            min_child_weight=3,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=2.0,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def build_final_model(name: str, scale_pos_weight: float):
    return make_models(scale_pos_weight)[name]


def main():
    train, validation, test = load_temporal_split()
    X_train, y_train = xy(train)
    X_val, y_val = xy(validation)
    X_test, y_test = xy(test)

    print("TIME-ORDERED SPLIT")
    print(f"Train      : {train.date.min().date()} -> {train.date.max().date()} ({len(train)} rows, positives: {int(y_train.sum())})")
    print(f"Validation : {validation.date.min().date()} -> {validation.date.max().date()} ({len(validation)} rows, positives: {int(y_val.sum())})")
    print(f"Test       : {test.date.min().date()} -> {test.date.max().date()} ({len(test)} rows, positives: {int(y_test.sum())})")

    scale_pos_weight = (len(y_train) - int(y_train.sum())) / max(int(y_train.sum()), 1)
    candidates = {}
    for name, model in make_models(scale_pos_weight).items():
        model.fit(X_train, y_train)
        pr_auc, threshold, f1 = evaluate(model, X_val, y_val)
        candidates[name] = {
            "model": model,
            "validation_pr_auc": pr_auc,
            "validation_threshold": threshold,
            "validation_f1": f1,
        }
        print(f"{name:18s} validation PR-AUC={pr_auc:.4f} threshold={threshold:.3f} F1={f1:.3f}")

    winner_name = max(candidates, key=lambda n: candidates[n]["validation_pr_auc"])
    winner = candidates[winner_name]
    selected_threshold = winner["validation_threshold"]
    print(f"\nSelected by validation PR-AUC: {winner_name} ({winner['validation_pr_auc']:.4f})")

    trainval = pd.concat([train, validation], ignore_index=True)
    X_trainval, y_trainval = xy(trainval)
    final_scale = (len(y_trainval) - int(y_trainval.sum())) / max(int(y_trainval.sum()), 1)
    final_model = build_final_model(winner_name, final_scale)
    final_model.fit(X_trainval, y_trainval)

    test_proba = final_model.predict_proba(X_test)[:, 1]
    test_pr_auc = average_precision_score(y_test, test_proba)
    test_pred = (test_proba >= selected_threshold).astype(int)
    tp = int(((test_pred == 1) & (y_test.to_numpy() == 1)).sum())
    predicted_positive = int((test_pred == 1).sum())
    actual_positive = int((y_test == 1).sum())
    precision = tp / max(predicted_positive, 1)
    recall = tp / max(actual_positive, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    print("\nFINAL UNTOUCHED TEST")
    print(f"Test PR-AUC : {test_pr_auc:.4f}")
    print(f"Precision   : {precision:.3f}")
    print(f"Recall      : {recall:.3f}")
    print(f"F1          : {f1:.3f}")
    print(f"Threshold   : {selected_threshold:.3f} (validation only)")

    with open(MODELS_DIR / "flood_model.pkl", "wb") as f:
        pickle.dump(final_model, f)

    metadata = {
        "model_version": "2.0",
        "selected_model": winner_name,
        "selection_metric": "validation_pr_auc",
        "validation_pr_auc": round(winner["validation_pr_auc"], 4),
        "validation_f1": round(winner["validation_f1"], 4),
        "decision_threshold": round(selected_threshold, 6),
        "final_test_pr_auc": round(float(test_pr_auc), 4),
        "final_test_precision": round(float(precision), 4),
        "final_test_recall": round(float(recall), 4),
        "final_test_f1": round(float(f1), 4),
        "candidates_validation_pr_auc": {
            name: round(info["validation_pr_auc"], 4) for name, info in candidates.items()
        },
        "split": {"train_end": TRAIN_END, "validation_end": VALIDATION_END, "test_start": "2022-01-01"},
        "feature_count": len(FEATURE_ORDER),
        "feature_order": FEATURE_ORDER,
        "n_train": len(train),
        "n_validation": len(validation),
        "n_test": len(test),
        "n_positive_train": int(y_train.sum()),
        "n_positive_validation": int(y_val.sum()),
        "n_positive_test": int(y_test.sum()),
    }
    with open(MODELS_DIR / "flood_model_selection.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print("Saved models/flood_model.pkl")
    print("Saved models/flood_model_selection.pkl")


if __name__ == "__main__":
    main()
