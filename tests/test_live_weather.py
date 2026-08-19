"""
test_live_weather.py
---------------------
Tests for live/weather_api.py.

Two modes:
  1. OFFLINE (default, always runs): validates locality lookup and the
     next_24h_precipitation_mm() math against a canned Open-Meteo-shaped
     fixture, so this test suite works even with no internet access
     (e.g. inside a sandboxed CI runner).
  2. ONLINE (opt-in): set RUN_LIVE_NETWORK_TESTS=1 to also hit the real
     Open-Meteo API and check the response shape. Skipped by default.

Run: python -m pytest tests/test_live_weather.py -v
  or: python tests/test_live_weather.py
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from live.weather_api import get_locality_coords, next_24h_precipitation_mm  # noqa: E402

FIXTURE_WEATHER_JSON = {
    "current": {"time": "2026-08-19T09:00", "precipitation": 1.2},
    "hourly": {
        "time": [f"2026-08-19T{h:02d}:00" for h in range(9, 24)]
        + [f"2026-08-20T{h:02d}:00" for h in range(0, 9)],
        "precipitation": [1.2, 0.5, 0.0, 0.0, 2.0, 3.5, 0.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 6.5, 8.0, 2.0, 0.0,
                           0.0, 0.0, 0.0, 0.5],
    },
}


def test_get_locality_coords_known_locality():
    coords = get_locality_coords("Sholinganallur")
    assert coords["locality"].lower() == "sholinganallur"
    assert -90 <= coords["latitude"] <= 90
    assert -180 <= coords["longitude"] <= 180


def test_get_locality_coords_unknown_locality_raises():
    try:
        get_locality_coords("Not A Real Place")
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_next_24h_precipitation_sums_correct_window():
    total = next_24h_precipitation_mm(FIXTURE_WEATHER_JSON)
    expected = sum(FIXTURE_WEATHER_JSON["hourly"]["precipitation"][:24])
    assert abs(total - expected) < 1e-6


def test_online_fetch_optional():
    if os.environ.get("RUN_LIVE_NETWORK_TESTS") != "1":
        print("Skipping online Open-Meteo test (set RUN_LIVE_NETWORK_TESTS=1 to enable).")
        return
    from live.weather_api import get_weather_for_locality
    weather = get_weather_for_locality("Sholinganallur")
    assert weather["latitude"] is not None
    assert "forecast_next_24h_precipitation_mm" in weather


if __name__ == "__main__":
    test_get_locality_coords_known_locality()
    test_get_locality_coords_unknown_locality_raises()
    test_next_24h_precipitation_sums_correct_window()
    test_online_fetch_optional()
    print("All live weather tests passed.")
