"""
Smoke tests for get_electricity_prices tool.
"""
import sys

sys.path.insert(0, ".")
from src.tools import get_electricity_prices


def print_section(title: str) -> None:
    print(f"\n{'=' * 50}")
    print(f"  {title}")
    print("=" * 50)


def test_live_api(date: str = "2024-11-01") -> None:
    """Past date — should return real Agile prices from Octopus API."""
    print_section(f"Live Octopus API: {date}")
    result = get_electricity_prices.invoke({"date": date})

    assert "error" not in result
    assert result["currency"] == "GBP"
    assert len(result["hourly_rates"]) == 24

    print(f"Source       : {result['source']}")
    print(f"Pricing type : {result['pricing_type']}")
    print(f"Currency     : {result['currency']}")
    print()
    print(f"{'Hour':>5}  {'Period':<12}  {'p/kWh':>8}  {'GBP/kWh':>9}")
    print("-" * 44)
    for h in result["hourly_rates"]:
        print(
            f"{h['hour']:>5}  {h['period']:<12}  "
            f"{h['rate_pence_kwh']:>7.2f}p  £{h['rate']:>7.4f}"
        )
    print("PASS")


def test_fallback_future(date: str = "2030-01-01") -> None:
    """Future date — Agile prices not yet published, should use fallback TOU rates."""
    print_section(f"Fallback (future date): {date}")
    result = get_electricity_prices.invoke({"date": date})

    assert "error" not in result
    assert result["currency"] == "GBP"
    assert len(result["hourly_rates"]) == 24
    assert "fallback" in result["source"]

    print(f"Source       : {result['source']}")
    print(f"Pricing type : {result['pricing_type']}")
    sample_hours = [h for h in result["hourly_rates"] if h["hour"] in (0, 7, 16, 22)]
    for h in sample_hours:
        print(f"  Hour {h['hour']:>2}: {h['period']:<12} {h['rate_pence_kwh']}p/kWh")
    print("PASS")


def test_default_date() -> None:
    """No date argument — should default to today."""
    print_section("Default date (today)")
    result = get_electricity_prices.invoke({})

    assert "error" not in result
    assert "date" in result
    assert len(result["hourly_rates"]) == 24

    print(f"Resolved date: {result['date']}")
    print(f"Source       : {result['source']}")
    print("PASS")


def test_period_classification() -> None:
    """Verify TOU period labels are applied to the right hours."""
    print_section("Period classification check")
    result = get_electricity_prices.invoke({"date": "2030-01-01"})

    rates_by_hour = {h["hour"]: h["period"] for h in result["hourly_rates"]}

    # Off-peak: hours 0-6 and 23
    for h in list(range(7)) + [23]:
        assert rates_by_hour[h] == "off_peak", f"Hour {h} should be off_peak"

    # Peak: hours 16-18
    for h in range(16, 19):
        assert rates_by_hour[h] == "peak", f"Hour {h} should be peak"

    # Mid-peak: hours 7-15 and 19-22
    for h in list(range(7, 16)) + list(range(19, 23)):
        assert rates_by_hour[h] == "mid_peak", f"Hour {h} should be mid_peak"

    print("All period labels correct.")
    print("PASS")


def test_determinism() -> None:
    """Same date should always return the same rates (fallback path)."""
    print_section("Determinism check (fallback path)")
    r1 = get_electricity_prices.invoke({"date": "2030-06-15"})
    r2 = get_electricity_prices.invoke({"date": "2030-06-15"})

    assert r1["hourly_rates"] == r2["hourly_rates"], "Rates differ across calls for same date"
    print("Identical results for two calls on the same date.")
    print("PASS")


if __name__ == "__main__":
    test_live_api()
    test_fallback_future()
    test_default_date()
    test_period_classification()
    test_determinism()
    print("\nAll tests complete.")
