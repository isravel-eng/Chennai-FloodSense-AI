from pathlib import Path
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "model2_features.csv"
MODEL_PATH = ROOT / "models" / "flood_model.pkl"

FEATURES = [
    "rainfall_mm",
    "rainfall_3d_mm",
    "rainfall_7d_mm",
    "rainfall_30d_mm",
    "latitude",
    "longitude",
]
TARGET = "flood_occurred_documented"


def main():
    data = pd.read_csv(DATA_PATH)
    # Drop rows where target is NaN (e.g., backfilled data for 2024-2026)
    valid_data = data.dropna(subset=[TARGET])
    X = valid_data[FEATURES]
    y = valid_data[TARGET]

    # Convert date and perform a chronological split to avoid temporal leakage
    valid_data["date"] = pd.to_datetime(valid_data["date"])
    train_df = valid_data[valid_data["date"].dt.year < 2023]
    test_df = valid_data[valid_data["date"].dt.year >= 2023]

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]
    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print("--- METRICS ---")
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.3f}")
    print(f"Recall: {recall_score(y_test, y_pred, zero_division=0):.3f}")
    print(f"F1: {f1_score(y_test, y_pred, zero_division=0):.3f}")
    print(f"PR-AUC: {average_precision_score(y_test, y_prob):.3f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.3f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save model artifact
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
        
    # Save preprocessing artifact
    preprocessing_path = ROOT / "models" / "flood_preprocessing.pkl"
    with open(preprocessing_path, "wb") as f:
        pickle.dump({"feature_order": FEATURES}, f)

    print(f"Saved: {MODEL_PATH}")
    print("Features:", FEATURES)


if __name__ == "__main__":
    main()
