import json
import pickle
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
import numpy as np

from api.main import app
from model_2_flood.predict_flood import predict_flood, _load_artifacts
import live.live_features as live_features

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "flood_model.pkl"
PREPROC_PATH = MODELS_DIR / "flood_preprocessing.pkl"

client = TestClient(app)

def test_model_artifact_loading():
    """Test that the model artifacts load correctly without corruption (e.g. XGBoostError)."""
    assert MODEL_PATH.exists()
    assert PREPROC_PATH.exists()
    model, preprocessing = _load_artifacts()
    assert model is not None
    assert preprocessing is not None
    # Verify model has predict_proba
    assert hasattr(model, "predict_proba")
    # Verify classes are [0, 1] so index 1 is indeed the flood class
    assert list(model.classes_) == [0.0, 1.0]

def test_feature_ordering_and_shape():
    """Test feature shape and ordering are exactly as expected by the model."""
    _, preprocessing = _load_artifacts()
    order = preprocessing["feature_order"]
    assert order == [
        "rainfall_mm",
        "rainfall_3d_mm",
        "rainfall_7d_mm",
        "rainfall_30d_mm",
        "latitude",
        "longitude"
    ]
    
    # Try prediction with dummy data
    features = {f: 0.0 for f in order}
    features["rainfall_mm"] = 150.0
    result = predict_flood(features)
    assert "probability" in result
    assert "risk_band" in result
    assert 0.0 <= result["probability"] <= 1.0

def test_missing_feature_handling():
    """Test that predict_flood raises KeyError if a feature is missing."""
    with pytest.raises(KeyError):
        predict_flood({"rainfall_mm": 10.0, "latitude": 13.0, "longitude": 80.0})

def test_api_flood_risk_endpoint():
    """Test the main API endpoint for Sholinganallur."""
    response = client.get("/api/v1/flood-risk/Sholinganallur")
    assert response.status_code == 200
    data = response.json()
    assert "current" in data
    assert "next_24h" in data
    assert "locality" in data
    assert data["locality"].lower() == "sholinganallur"

def test_api_invalid_locality():
    """Test API behavior for an unknown locality."""
    response = client.get("/api/v1/flood-risk/UnknownCity")
    assert response.status_code == 404

def test_feature_parity():
    """Test that live_features does NOT leak today's rainfall into historical 30d rolling."""
    # live_features logic test
    weather = {"current_precipitation_mm": 100.0}
    history = {
        "rainfall_mm": 0.0,
        "rainfall_3d_mm": 10.0,
        "rainfall_7d_mm": 20.0,
        "rainfall_30d_mm": 50.0,
        "rainfall_lag_1": 5.0,
        "rainfall_lag_2": 2.0,
        "rainfall_lag_3": 3.0,
        "rainfall_lag_7": 0.0,
    }
    location = {"latitude": 13.0, "longitude": 80.0, "elevation_m_approx": 10.0}
    
    current = live_features.build_current_features(weather, history, location)
    
    assert current["rainfall_mm"] == 100.0
    # The crucial fix: live_features just copies history's 3d/7d/30d
    assert current["rainfall_3d_mm"] == 10.0
    assert current["rainfall_7d_mm"] == 20.0
    assert current["rainfall_30d_mm"] == 50.0
