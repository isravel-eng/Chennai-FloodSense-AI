import sys
from pathlib import Path
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
LOOKUP_PATH = ROOT / "data" / "processed" / "locality_lookup.csv"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
HOURLY_VARS = ["precipitation", "rain", "temperature_2m", "relative_humidity_2m", "wind_speed_10m", "weather_code"]
CURRENT_VARS = ["precipitation", "rain", "temperature_2m", "relative_humidity_2m", "wind_speed_10m"]
DAILY_VARS = ["precipitation_sum", "rain_sum", "weather_code"]

def get_locality_coords(locality: str) -> dict:
    lookup = pd.read_csv(LOOKUP_PATH)
    match = lookup[lookup["locality"].str.lower() == locality.lower()]
    if match.empty:
        raise ValueError(f"Unknown locality '{locality}'. Known localities: {', '.join(sorted(lookup['locality'].tolist()))}")
    row = match.iloc[0]
    return {"locality": row["locality"], "latitude": float(row["latitude"]), "longitude": float(row["longitude"]), "elevation_m_approx": float(row["elevation_m_approx"])}

def fetch_weather(latitude: float, longitude: float, forecast_days: int = 7) -> dict:
    params = {"latitude": latitude, "longitude": longitude, "current": ",".join(CURRENT_VARS), "hourly": ",".join(HOURLY_VARS), "daily": ",".join(DAILY_VARS), "forecast_days": forecast_days, "timezone": "Asia/Kolkata"}
    resp = requests.get(OPEN_METEO_URL, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()

def next_24h_precipitation_mm(weather_json: dict) -> float:
    hourly = weather_json["hourly"]
    times, precip = hourly["time"], hourly["precipitation"]
    current_time = weather_json.get("current", {}).get("time", times[0])
    try: start_idx = times.index(current_time)
    except ValueError: start_idx = 0
    return float(sum(v for v in precip[start_idx:start_idx + 24] if v is not None))

def daily_forecast(weather_json: dict) -> list[dict]:
    daily = weather_json.get("daily", {})
    dates, precipitation, rain = daily.get("time", []), daily.get("precipitation_sum", []), daily.get("rain_sum", [])
    result = []
    for i, day in enumerate(dates[:7]):
        p = precipitation[i] if i < len(precipitation) else None
        r = rain[i] if i < len(rain) else None
        result.append({"date": day, "rainfall_mm": float((r if r is not None else p) or 0.0)})
    return result

def get_weather_for_locality(locality: str) -> dict:
    coords = get_locality_coords(locality)
    raw = fetch_weather(coords["latitude"], coords["longitude"])
    current = raw.get("current", {})
    return {**coords, "fetched_at": current.get("time"), "current_precipitation_mm": current.get("precipitation", 0.0) or 0.0, "current_temperature_c": current.get("temperature_2m"), "current_humidity_pct": current.get("relative_humidity_2m"), "current_wind_kmh": current.get("wind_speed_10m"), "forecast_next_24h_precipitation_mm": next_24h_precipitation_mm(raw), "daily_forecast": daily_forecast(raw), "raw": raw}
