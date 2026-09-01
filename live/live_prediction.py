from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from live.live_features import build_live_features
from model_2_flood.predict_flood import predict_flood


def predict_live_flood(locality: str) -> dict:
    features = build_live_features(locality)
    result = predict_flood(features)
    result["locality"] = locality
    return result
