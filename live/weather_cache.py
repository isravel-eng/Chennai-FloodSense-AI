"""Weather caching, deduplication, and rate-limiting logic."""

import os
import time
import logging
import threading
import requests
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

WEATHER_CACHE_TTL_SECONDS = int(os.environ.get("WEATHER_CACHE_TTL_SECONDS", "300"))
USER_AGENT = "Chennai-FloodSense-AI/1.0 (contact: admin@chennai-floodsense.ai)"

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
HOURLY_VARS = ["precipitation", "rain", "temperature_2m", "relative_humidity_2m", "wind_speed_10m", "weather_code"]
CURRENT_VARS = ["precipitation", "rain", "temperature_2m", "relative_humidity_2m", "wind_speed_10m"]
DAILY_VARS = ["precipitation_sum", "rain_sum", "weather_code"]

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
    # Round to 2 decimal places (~1.1km resolution), sufficient since Open-Meteo resolution is ~10km
    return (round(latitude, 2), round(longitude, 2), forecast_days)

def _fetch_with_retries(latitude: float, longitude: float, forecast_days: int, locality: str) -> dict:
    params = {
        "latitude": latitude, 
        "longitude": longitude, 
        "current": ",".join(CURRENT_VARS), 
        "hourly": ",".join(HOURLY_VARS), 
        "daily": ",".join(DAILY_VARS), 
        "forecast_days": forecast_days, 
        "timezone": "Asia/Kolkata"
    }
    headers = {"User-Agent": USER_AGENT}
    
    max_retries = 3
    for attempt in range(max_retries + 1):
        start_time = time.time()
        try:
            resp = requests.get(OPEN_METEO_URL, params=params, headers=headers, timeout=15)
            latency = time.time() - start_time
            
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                wait_time = int(retry_after) if retry_after and retry_after.isdigit() else (2 ** attempt)
                logger.warning(f"HTTP 429 Rate Limit for {locality}. Retrying in {wait_time}s (Attempt {attempt+1}/{max_retries})")
                if attempt < max_retries:
                    time.sleep(wait_time)
                    continue
                else:
                    resp.raise_for_status()
                    
            resp.raise_for_status()
            logger.info(f"Open-Meteo request successful for {locality} (attempt {attempt+1}, latency {latency:.2f}s)")
            return resp.json()
            
        except requests.exceptions.RequestException as e:
            latency = time.time() - start_time
            logger.warning(f"Request failed for {locality} (attempt {attempt+1}, latency {latency:.2f}s): {e}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)
                continue
            raise

def fetch_weather_cached(latitude: float, longitude: float, forecast_days: int = 7, locality: str = "Unknown") -> dict:
    key = _get_cache_key(latitude, longitude, forecast_days)
    
    with _cache_lock:
        if key in _cache:
            timestamp, data = _cache[key]
            if time.time() - timestamp < WEATHER_CACHE_TTL_SECONDS:
                logger.info(f"Weather cache HIT for {locality} {key}")
                return data
            else:
                del _cache[key]
                
    with _in_flight_lock:
        if key in _in_flight:
            logger.info(f"Weather request DEDUPLICATED for {locality} {key} (waiting for in-flight request)")
            req = _in_flight[key]
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
            
    logger.info(f"Weather cache MISS for {locality} {key}. Fetching from Open-Meteo...")
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
            if key in _in_flight:
                del _in_flight[key]
