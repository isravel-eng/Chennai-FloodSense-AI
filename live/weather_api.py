"""
weather_api.py
--------------
Fetches REAL current + forecast weather for a Chennai locality from
Open-Meteo (https://open-meteo.com) - free, no API key required.

Why Open-Meteo: its forecast endpoint returns current conditions plus
hourly precipitation/temperature/humidity/wind for up to 16 days ahead,
and it accepts raw lat/lon directly (no geocoding step needed since we
already have coordinates in locality_lookup.csv).

NOTE ON SANDBOXED EXECUTION: this repo may be developed/reviewed inside an
environment with restricted internet egress. If `requests.get()` fails
here with a connection error, that is a network-policy issue, not a bug -
run this script on a machine with normal internet access (your laptop,
a server, GitHub Actions, etc). test_live_weather.py includes an offline
mode using a canned fixture for exactly this reason.

Run standalone: python live/weather_api.py "Sholinganallur"
"""

import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
LOOKUP_PATH = ROOT / "data" / "processed" / "locality_lookup.csv"

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

HOURLY_VARS = [
    "precipitation",
    "rain",
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "weather_code",
]
CURRENT_VARS = [
    "precipitation",
    "rain",
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
]


def get_locality_coords(locality: str) -> dict:
    lookup = pd.read_csv(LOOKUP_PATH)
    match = lookup[lookup["locality"].str.lower() == locality.lower()]
    if match.empty:
        available = ", ".join(sorted(lookup["locality"].tolist()))
        raise ValueError(
            f"Unknown locality '{locality}'. Known localities: {available}"
        )
    row = match.iloc[0]
    return {
        "locality": row["locality"],
        "latitude": float(row["latitude"]),
        "longitude": float(row["longitude"]),
        "elevation_m_approx": float(row["elevation_m_approx"]),
    }


def fetch_weather(latitude: float, longitude: float, forecast_days: int = 2) -> dict:
    """
    Returns raw Open-Meteo JSON with `current` conditions and an hourly
    forecast covering `forecast_days` days (default 2, enough to compute a
    clean next-24h precipitation total starting from "now").
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": ",".join(CURRENT_VARS),
        "hourly": ",".join(HOURLY_VARS),
        "forecast_days": forecast_days,
        "timezone": "Asia/Kolkata",
    }
    resp = requests.get(OPEN_METEO_URL, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def next_24h_precipitation_mm(weather_json: dict) -> float:
    """Sums hourly precipitation (mm) for the 24 hourly slots starting at
    the current hour, per Open-Meteo's `current.time`."""
    hourly = weather_json["hourly"]
    times = hourly["time"]
    precip = hourly["precipitation"]

    current_time = weather_json.get("current", {}).get("time", times[0])
    try:
        start_idx = times.index(current_time)
    except ValueError:
        start_idx = 0

    window = precip[start_idx: start_idx + 24]
    return float(sum(v for v in window if v is not None))


def get_weather_for_locality(locality: str) -> dict:
    """High-level convenience function used by live_prediction.py."""
    coords = get_locality_coords(locality)
    raw = fetch_weather(coords["latitude"], coords["longitude"])
    current = raw.get("current", {})
    return {
        "locality": coords["locality"],
        "latitude": coords["latitude"],
        "longitude": coords["longitude"],
        "elevation_m_approx": coords["elevation_m_approx"],
        "fetched_at": current.get("time"),
        "current_precipitation_mm": current.get("precipitation", 0.0) or 0.0,
        "current_temperature_c": current.get("temperature_2m"),
        "current_humidity_pct": current.get("relative_humidity_2m"),
        "current_wind_kmh": current.get("wind_speed_10m"),
        "forecast_next_24h_precipitation_mm": next_24h_precipitation_mm(raw),
        "raw": raw,
    }


if __name__ == "__main__":
    locality = sys.argv[1] if len(sys.argv) > 1 else "Sholinganallur"
    weather = get_weather_for_locality(locality)
    weather.pop("raw")
    for k, v in weather.items():
        print(f"{k}: {v}")
