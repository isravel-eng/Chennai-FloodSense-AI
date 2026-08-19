"""
test_model1.py
---------------
Sanity checks for the saved SARIMA model (models/rainfall_model.pkl).
Run: python -m pytest tests/test_model1.py -v
  or: python tests/test_model1.py
"""

import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from statsmodels.tsa.statespace.sarimax import SARIMAXResults  # noqa: E402

MODELS_DIR = ROOT / "models"


def test_model_file_exists():
    assert (MODELS_DIR / "rainfall_model.pkl").exists(), (
        "rainfall_model.pkl missing - run model_1_rainfall/sarima.py first"
    )


def test_preprocessing_file_exists():
    assert (MODELS_DIR / "rainfall_preprocessing.pkl").exists()


def test_model_loads():
    model = SARIMAXResults.load(str(MODELS_DIR / "rainfall_model.pkl"))
    assert model is not None


def test_forecast_shape_and_sanity():
    model = SARIMAXResults.load(str(MODELS_DIR / "rainfall_model.pkl"))
    forecast = model.get_forecast(steps=6).predicted_mean
    assert len(forecast) == 6
    # Rainfall can't be negative in reality; SARIMA can technically predict
    # a small negative mean for very dry months - just check it's plausible
    # (not wildly out of range).
    assert forecast.min() > -50
    assert forecast.max() < 1000


def test_preprocessing_metadata_has_expected_keys():
    with open(MODELS_DIR / "rainfall_preprocessing.pkl", "rb") as f:
        meta = pickle.load(f)
    for key in ["order", "seasonal_order", "seasonal_period", "holdout_mae_mm"]:
        assert key in meta


if __name__ == "__main__":
    test_model_file_exists()
    test_preprocessing_file_exists()
    test_model_loads()
    test_forecast_shape_and_sanity()
    test_preprocessing_metadata_has_expected_keys()
    print("All Model 1 tests passed.")
