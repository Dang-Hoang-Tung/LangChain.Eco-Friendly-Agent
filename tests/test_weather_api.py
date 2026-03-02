"""
Smoke tests for get_weather_forecast tool.
"""
import sys
import json

sys.path.insert(0, ".")
from src.tools import get_weather_forecast


def print_section(title: str) -> None:
    print(f"\n{'=' * 50}")
    print(f"  {title}")
    print('=' * 50)


def test_basic(location: str = "San Francisco, CA", days: int = 2) -> None:
    print_section(f"Basic forecast: {location}, {days} day(s)")
    result = get_weather_forecast.invoke({"location": location, "days": days})

    if "error" in result:
        print(f"ERROR: {result['error']}")
        return

    print(f"Resolved location : {result['location']}")
    print(f"Forecast days     : {result['forecast_days']}")
    print(f"Hourly entries    : {len(result['hourly'])}  (expected {days * 24})")
    print(f"Current conditions: {result['current']}")
    print("\nFirst 3 hourly entries:")
    for h in result["hourly"][:3]:
        print(f"  {h}")


def test_city_only(location: str = "Austin") -> None:
    print_section(f"City-only input: '{location}'")
    result = get_weather_forecast.invoke({"location": location, "days": 1})
    if "error" in result:
        print(f"ERROR: {result['error']}")
    else:
        print(f"Resolved to: {result['location']}")
        print(f"Current: {result['current']}")


def test_invalid_location() -> None:
    print_section("Invalid location (expect error)")
    result = get_weather_forecast.invoke({"location": "XYZNonexistentCity123", "days": 1})
    print(f"Result: {result}")
    assert "error" in result, "Expected an error key for invalid location"
    print("PASS: error key present as expected")


def test_days_clamping() -> None:
    print_section("Days clamping (requesting 10 days, API max is 7)")
    result = get_weather_forecast.invoke({"location": "New York", "days": 10})
    if "error" in result:
        print(f"ERROR: {result['error']}")
    else:
        # API returns at most 7 days worth of hours
        print(f"Hourly entries returned: {len(result['hourly'])}  (max 7*24=168)")
        assert len(result["hourly"]) <= 7 * 24, "Should not exceed 7 days"
        print("PASS: days clamped correctly")


if __name__ == "__main__":
    test_basic()
    test_city_only()
    test_invalid_location()
    test_days_clamping()
    print("\nAll tests complete.")
