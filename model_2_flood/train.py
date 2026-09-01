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
    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
    print(classification_report(y_test, predictions, zero_division=0))

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print(f"Saved: {MODEL_PATH}")
    print("Features:", FEATURES)


if __name__ == "__main__":
    main()
