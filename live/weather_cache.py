"""WeatherAPI.com caching, deduplication, and request retry logic."""

import os
import time
import logging
import threading
import requests
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

WEATHER_CACHE_TTL_SECONDS = int(os.environ.get("WEATHER_CACHE_TTL_SECONDS", "300"))
WEATHER_API_URL = "https://api.weatherapi.com/v1/forecast.json"
WEATHER_API_KEY_ENV = "WEATHER_API_KEY"

_cache: Dict[Tuple[float, float, int], Tuple[float, Any]] = {}
_cache_lock = threading.Lock()

class InFlightRequest:
    def __init__(self):
        self.event = threading.Event()
        self.result = None
        self.exception = None

_in_flight: Dict[Tuple[float, float, int], InFlightRequest] = {}
_in_flight_lock = threading.Lock()

def _get_cache_key(latitude: float, longitude: float, forecast_days: int) -> Tuple[float, float, int]:
    return (round(latitude, 4), round(longitude, 4), int(forecast_days))

def _normalize_weatherapi_response(data: dict) -> dict:
    location = data.get("location", {})
    current = data.get("current", {})
    forecast_days = data.get("forecast", {}).get("forecastday", [])

    hourly_times = []
    hourly_precip = []
    hourly_rain = []
    hourly_temp = []
    hourly_humidity = []
    hourly_wind = []
    hourly_code = []

    for day in forecast_days:
        for hour in day.get("hour", []):
            hourly_times.append(str(hour.get("time", "")))
            hourly_precip.append(float(hour.get("precip_mm") or 0.0))
            hourly_rain.append(float(hour.get("precip_mm") or 0.0))
            hourly_temp.append(hour.get("temp_c"))
            hourly_humidity.append(hour.get("humidity"))
            hourly_wind.append(hour.get("wind_kph"))
            hourly_code.append((hour.get("condition") or {}).get("code"))

    daily_time = []
    daily_precip = []
    daily_rain = []
    daily_code = []
    for day in forecast_days:
        day_info = day.get("day", {})
        daily_time.append(str(day.get("date")))
        daily_precip.append(float(day_info.get("totalprecip_mm") or 0.0))
        daily_rain.append(float(day_info.get("totalprecip_mm") or 0.0))
        daily_code.append((day_info.get("condition") or {}).get("code"))

    last_updated = current.get("last_updated", "")
    last_updated_epoch = current.get("last_updated_epoch")
    current_time = last_updated[:13] + ":00" if last_updated else ""
    return {
        "source": "weatherapi",
        "timezone": location.get("tz_id") or "Asia/Kolkata",
        "location": location,
        "current": {
            "time": current_time,
            "last_updated": last_updated,
            "last_updated_epoch": last_updated_epoch,
            "precipitation": float(current.get("precip_mm") or 0.0),
            "rain": float(current.get("precip_mm") or 0.0),
            "temperature_2m": current.get("temp_c"),
            "relative_humidity_2m": current.get("humidity"),
            "wind_speed_10m": current.get("wind_kph"),
            "weather_code": (current.get("condition") or {}).get("code"),
        },
        "hourly": {
            "time": hourly_times,
            "precipitation": hourly_precip,
            "rain": hourly_rain,
            "temperature_2m": hourly_temp,
            "relative_humidity_2m": hourly_humidity,
            "wind_speed_10m": hourly_wind,
            "weather_code": hourly_code,
        },
        "daily": {
            "time": daily_time,
            "precipitation_sum": daily_precip,
            "rain_sum": daily_rain,
            "weather_code": daily_code,
        },
    }

def _fetch_with_retries(latitude: float, longitude: float, forecast_days: int, locality: str) -> dict:
    api_key = os.environ.get(WEATHER_API_KEY_ENV, "").strip()
    if not api_key:
        raise RuntimeError("WEATHER_API_KEY is not configured on the backend.")

    params = {
        "key": api_key,
        "q": f"{latitude},{longitude}",
        "days": max(1, min(int(forecast_days), 14)),
        "aqi": "no",
        "alerts": "no",
    }

    max_retries = 2
    for attempt in range(max_retries + 1):
        start_time = time.time()
        try:
            resp = requests.get(WEATHER_API_URL, params=params, timeout=15)
            latency = time.time() - start_time

            if resp.status_code == 429:
                wait_time = min(2 ** attempt, 4)
                logger.warning(
                    "WeatherAPI rate limit for %s. Retrying in %ss (attempt %s/%s)",
                    locality, wait_time, attempt + 1, max_retries + 1
                )
                if attempt < max_retries:
                    time.sleep(wait_time)
                    continue

            resp.raise_for_status()
            logger.info(
                "WeatherAPI request successful for %s (attempt %s, latency %.2fs)",
                locality, attempt + 1, latency
            )
            return _normalize_weatherapi_response(resp.json())

        except requests.exceptions.RequestException as exc:
            latency = time.time() - start_time
            logger.warning(
                "WeatherAPI request failed for %s (attempt %s, latency %.2fs): %s",
                locality, attempt + 1, latency, exc
            )
            if attempt < max_retries:
                time.sleep(2 ** attempt)
                continue
            raise

def fetch_weather_cached(
    latitude: float,
    longitude: float,
    forecast_days: int = 7,
    locality: str = "Unknown",
) -> dict:
    key = _get_cache_key(latitude, longitude, forecast_days)

    with _cache_lock:
        hit = _cache.get(key)
        if hit:
            timestamp, data = hit
            if time.time() - timestamp < WEATHER_CACHE_TTL_SECONDS:
                logger.info("Weather cache HIT for %s %s", locality, key)
                return data
            del _cache[key]

    with _in_flight_lock:
        req = _in_flight.get(key)
        if req is not None:
            logger.info("Weather request DEDUPLICATED for %s %s", locality, key)
            is_leader = False
        else:
            req = InFlightRequest()
            _in_flight[key] = req
            is_leader = True

    if not is_leader:
        req.event.wait()
        if req.exception:
            raise req.exception
        return req.result

    logger.info("Weather cache MISS for %s %s. Fetching from WeatherAPI.com...", locality, key)
    try:
        data = _fetch_with_retries(latitude, longitude, forecast_days, locality)
        with _cache_lock:
            _cache[key] = (time.time(), data)
        req.result = data
        return data
    except Exception as exc:
        req.exception = exc
        raise
    finally:
        req.event.set()
        with _in_flight_lock:
            _in_flight.pop(key, None)
