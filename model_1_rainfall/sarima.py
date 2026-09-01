import pickle
import warnings
from pathlib import Path

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "monthly_rainfall_citywide.csv"
MODEL_PATH = ROOT / "models" / "rainfall_model.pkl"


def main():
    data = pd.read_csv(DATA_PATH, parse_dates=["month_start"])
    series = data.set_index("month_start")["avg_rainfall_mm"].asfreq("MS")
    series = series.interpolate().bfill().ffill()

    # Simple seasonal rainfall model: yearly seasonality (12 months).
    model = SARIMAX(
        series,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False)
    fitted.save(MODEL_PATH)

    print(f"Saved: {MODEL_PATH}")
    print("Model: SARIMA(1,1,1)(1,1,1,12)")


if __name__ == "__main__":
    main()
