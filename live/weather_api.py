"""WeatherAPI.com adapter for Chennai FloodSense AI."""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LOOKUP_PATH = ROOT / "data" / "processed" / "locality_lookup.csv"

from live.weather_cache import fetch_weather_cached


def get_locality_coords(locality: str) -> dict:
    lookup = pd.read_csv(LOOKUP_PATH)
    match = lookup[lookup["locality"].str.lower() == locality.lower()]
    if match.empty:
        raise ValueError(
            f"Unknown locality '{locality}'. Known localities: "
            f"{', '.join(sorted(lookup['locality'].tolist()))}"
        )
    row = match.iloc[0]
    return {
        "locality": row["locality"],
        "latitude": float(row["latitude"]),
        "longitude": float(row["longitude"]),
        "elevation_m_approx": float(row["elevation_m_approx"]),
        "display_name": str(row.get("display_name", row["locality"])),
    }


def fetch_weather(latitude: float, longitude: float, forecast_days: int = 7) -> dict:
    return fetch_weather_cached(latitude, longitude, forecast_days)


def next_24h_precipitation_mm(weather_json: dict) -> float:
    hourly = weather_json.get("hourly", {})
    times = hourly.get("time", [])
    precip = hourly.get("precipitation", [])
    if not times or not precip:
        return 0.0

    current_time = weather_json.get("current", {}).get("time", times[0])
    try:
        current_ts = pd.Timestamp(current_time)
        hour_ts = pd.to_datetime(times)
        candidates = [i for i, ts in enumerate(hour_ts) if ts >= current_ts.floor("h")]
        start_idx = candidates[0] if candidates else 0
    except Exception:
        start_idx = 0

    return float(sum(v for v in precip[start_idx:start_idx + 24] if v is not None))


def daily_forecast(weather_json: dict) -> list[dict]:
    daily = weather_json.get("daily", {})
    dates = daily.get("time", [])
    precipitation = daily.get("precipitation_sum", [])
    rain = daily.get("rain_sum", [])
    result = []
    for i, day in enumerate(dates[:7]):
        p = precipitation[i] if i < len(precipitation) else None
        r = rain[i] if i < len(rain) else None
        result.append({
            "date": day,
            "rainfall_mm": float((r if r is not None else p) or 0.0),
        })
    return result


def get_weather_for_locality(locality: str) -> dict:
    coords = get_locality_coords(locality)
    raw = fetch_weather_cached(
        coords["latitude"],
        coords["longitude"],
        forecast_days=7,
        locality=locality,
    )
    current = raw.get("current", {})
    return {
        **coords,
        "weather_source": "weatherapi",
        "observation_date": str(current.get("last_updated", ""))[:10],
        "fetched_at": current.get("last_updated"),
        "current_precipitation_mm": current.get("precipitation", 0.0) or 0.0,
        "current_temperature_c": current.get("temperature_2m"),
        "current_humidity_pct": current.get("relative_humidity_2m"),
        "current_wind_kmh": current.get("wind_speed_10m"),
        "forecast_next_24h_precipitation_mm": next_24h_precipitation_mm(raw),
        "daily_forecast": daily_forecast(raw),
        "raw": raw,
    }
