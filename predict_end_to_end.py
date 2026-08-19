"""
predict_end_to_end.py
----------------------
Single CLI entry point for Chennai FloodSense AI.

Two modes:

  --live LOCALITY
      Fetches real current weather from Open-Meteo and runs the full live
      pipeline (weather API -> recent rainfall -> feature builder ->
      XGBoost). Requires internet access. Prints both the "current" and
      "next 24h" flood-risk predictions.

  --historical LOCALITY --date YYYY-MM-DD
      Looks up an existing row for LOCALITY on DATE in
      data/processed/model2_features.csv and runs it back through
      Model 2, for testing/demo/backtesting purposes (no internet
      required - useful in restricted-network environments).

  --forecast-rainfall
      Prints the SARIMA (Model 1) monthly rainfall research forecast for
      the next N months (default 6). Independent of Model 2 / live layer.

Examples:
    python predict_end_to_end.py --live Sholinganallur
    python predict_end_to_end.py --historical Alandur --date 2017-11-13
    python predict_end_to_end.py --forecast-rainfall --months 12
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def run_live(locality: str):
    from live.live_prediction import predict_live_flood, _pretty_print
    result = predict_live_flood(locality)
    _pretty_print(result)
    return result


def run_historical(locality: str, date_str: str):
    import pandas as pd
    from model_2_flood.predict_flood import predict_flood
    from model_2_flood.preprocessing import FEATURE_ORDER

    df = pd.read_csv(ROOT / "data" / "processed" / "model2_features.csv")
    match = df[(df["locality"].str.lower() == locality.lower()) & (df["date"] == date_str)]
    if match.empty:
        print(f"No record found for locality='{locality}' date='{date_str}'.")
        print("Check data/processed/model2_features.csv for available (locality, date) pairs.")
        return None

    row = match.iloc[0]
    features = {col: float(row[col]) for col in FEATURE_ORDER}
    result = predict_flood(features)

    print(f"\n{locality.upper()} - {date_str}")
    print(f"  actual flood documented: {bool(row['flood_occurred_documented'])}")
    print(f"  rainfall_mm            : {row['rainfall_mm']}")
    print(f"  predicted probability  : {result['probability']}")
    print(f"  predicted risk band    : {result['risk_band']}")
    return result


def run_forecast_rainfall(months: int):
    from statsmodels.tsa.statespace.sarimax import SARIMAXResults
    model = SARIMAXResults.load(str(ROOT / "models" / "rainfall_model.pkl"))
    forecast = model.get_forecast(steps=months)
    df = forecast.predicted_mean.round(1).to_frame("forecast_mm")
    ci = forecast.conf_int(alpha=0.05).round(1)
    df["lower_95"] = ci.iloc[:, 0]
    df["upper_95"] = ci.iloc[:, 1]
    print("\nModel 1 (SARIMA) - city-wide monthly rainfall research forecast:")
    print(df.to_string())
    print(
        "\nNote: this is a MONTHLY research forecast, not used directly in "
        "the live flood-risk prediction (see README.md)."
    )
    return df


def main():
    parser = argparse.ArgumentParser(description="Chennai FloodSense AI - end-to-end predictor")
    parser.add_argument("--live", metavar="LOCALITY", help="Run the live weather-API-backed prediction")
    parser.add_argument("--historical", metavar="LOCALITY", help="Run a historical backtest prediction")
    parser.add_argument("--date", metavar="YYYY-MM-DD", help="Date to use with --historical")
    parser.add_argument("--forecast-rainfall", action="store_true", help="Print Model 1 SARIMA forecast")
    parser.add_argument("--months", type=int, default=6, help="Months ahead for --forecast-rainfall")
    args = parser.parse_args()

    if args.live:
        run_live(args.live)
    elif args.historical:
        if not args.date:
            parser.error("--historical requires --date YYYY-MM-DD")
        run_historical(args.historical, args.date)
    elif args.forecast_rainfall:
        run_forecast_rainfall(args.months)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
