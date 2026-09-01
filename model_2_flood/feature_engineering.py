from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "master_dataset.csv"
OUT_PATH = ROOT / "data" / "processed" / "model2_features.csv"
MONTHLY_PATH = ROOT / "data" / "processed" / "monthly_rainfall_citywide.csv"


def main():
    df = pd.read_csv(RAW_PATH, parse_dates=["date"])
    df = df.sort_values(["locality", "date"]).copy()

    # Keep only features that are easy to explain.
    df["rainfall_3d_mm"] = df.groupby("locality")["rainfall_mm"].transform(
        lambda s: s.rolling(3, min_periods=1).sum()
    )
    df["rainfall_7d_mm"] = df.groupby("locality")["rainfall_mm"].transform(
        lambda s: s.rolling(7, min_periods=1).sum()
    )
    df["rainfall_30d_mm"] = df.groupby("locality")["rainfall_mm"].transform(
        lambda s: s.rolling(30, min_periods=1).sum()
    )

    columns = [
        "date",
        "locality",
        "latitude",
        "longitude",
        "rainfall_mm",
        "rainfall_3d_mm",
        "rainfall_7d_mm",
        "rainfall_30d_mm",
        "flood_occurred_documented",
    ]
    df[columns].to_csv(OUT_PATH, index=False)

    city = (
        df.groupby("date", as_index=False)["rainfall_mm"].mean()
        .rename(columns={"rainfall_mm": "avg_rainfall_mm"})
        .set_index("date")
        .resample("MS")
        .mean()
        .reset_index()
        .rename(columns={"date": "month_start"})
    )
    city.to_csv(MONTHLY_PATH, index=False)

    print(f"Saved {OUT_PATH}")
    print(f"Saved {MONTHLY_PATH}")


if __name__ == "__main__":
    main()
