"""
stationarity.py
----------------
Checks stationarity of the city-wide monthly rainfall series using the
Augmented Dickey-Fuller (ADF) test, and reports seasonal differencing
needed before fitting SARIMA in sarima.py.

Run standalone: python model_1_rainfall/stationarity.py
"""

import pandas as pd
from pathlib import Path
from statsmodels.tsa.stattools import adfuller

ROOT = Path(__file__).resolve().parent.parent
MONTHLY_PATH = ROOT / "data" / "processed" / "monthly_rainfall_citywide.csv"


def adf_report(series: pd.Series, label: str):
    result = adfuller(series.dropna(), autolag="AIC")
    stat, pvalue, used_lag, nobs, crit_values, _ = result
    print(f"\n--- ADF test: {label} ---")
    print(f"ADF statistic : {stat:.4f}")
    print(f"p-value       : {pvalue:.4f}")
    for key, val in crit_values.items():
        print(f"critical [{key}] : {val:.4f}")
    verdict = "STATIONARY" if pvalue < 0.05 else "NON-STATIONARY"
    print(f"Verdict       : {verdict} (alpha=0.05)")
    return pvalue < 0.05


def main():
    df = pd.read_csv(MONTHLY_PATH, parse_dates=["month_start"])
    series = df.set_index("month_start")["avg_rainfall_mm"].interpolate(method="time").bfill().ffill()

    is_stationary = adf_report(series, "raw monthly rainfall")

    if not is_stationary:
        diff1 = series.diff().dropna()
        adf_report(diff1, "first difference (d=1)")

        seasonal_diff = series.diff(12).dropna()
        adf_report(seasonal_diff, "seasonal difference (D=1, period=12)")

        combo = series.diff().diff(12).dropna()
        adf_report(combo, "d=1 + seasonal D=1 (period=12)")

    print(
        "\nRecommendation: Chennai rainfall is strongly seasonal (northeast "
        "monsoon Oct-Dec). Use SARIMA with a seasonal period of 12 months, "
        "and rely on seasonal_order differencing (D=1) rather than aggressive "
        "manual differencing - pmdarima/statsmodels auto search in sarima.py "
        "picks the final (p,d,q)(P,D,Q,12) order by AIC."
    )


if __name__ == "__main__":
    main()
