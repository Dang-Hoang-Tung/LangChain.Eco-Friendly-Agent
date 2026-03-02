# EcoHome Energy Advisor - Database Setup
# Sets up the SQLite database with:
#   - Energy usage data (consumption, device types, costs)
#   - Solar generation data (production, weather conditions)

# =============================================================================
# 1. Imports
# =============================================================================
import os
import sys
import random
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.energy import DatabaseManager

# =============================================================================
# 2. Initialize Database Manager
# =============================================================================
db_path = str(Path(__file__).resolve().parent / "data" / "energy_data.db")

# Remove existing DB so we always start with a clean slate (avoids duplicate records).
if os.path.exists(db_path):
    os.remove(db_path)
    print(f"Removed existing database: {db_path}")

db_manager = DatabaseManager(db_path)
print(f"Initialized DatabaseManager at {db_path}")

# =============================================================================
# 3. Create Database Tables
# =============================================================================
db_manager.create_tables()

# =============================================================================
# 4. Generate Sample Energy Usage Data (past 30 days)
# =============================================================================
# Device types with typical hourly consumption patterns and peak hours.
device_types = {
    'EV':        {'base_kwh': 10,  'variation': 5,   'peak_hours': [18, 19, 20, 21]},
    'HVAC':      {'base_kwh': 2,   'variation': 1,   'peak_hours': [12, 13, 14, 15, 16, 17]},
    'appliance': {'base_kwh': 1.5, 'variation': 0.5, 'peak_hours': [19, 20, 21, 22]},
}

device_names = {
    'EV':        'Tesla Model 3',
    'HVAC':      'Main AC Unit',
    'appliance': ['Dishwasher', 'Washing Machine', 'Dryer'],
}

start_date = datetime.now() - timedelta(days=30)
records_created = 0

# range(31) so day 30 = today
for day in range(31):
    current_date = start_date + timedelta(days=day)
    for hour in range(24):
        timestamp = current_date.replace(hour=hour, minute=0, second=0, microsecond=0)
        for device_type, config in device_types.items():
            variation = random.uniform(-config['variation'], config['variation'])
            peak_multiplier = 1.5 if hour in config['peak_hours'] else 0.8
            consumption = max(0, (config['base_kwh'] + variation) * peak_multiplier)

            price_per_kwh = 0.15 if hour in config['peak_hours'] else 0.10
            cost = consumption * price_per_kwh

            name = device_names[device_type]
            if isinstance(name, list):
                name = random.choice(name)

            db_manager.add_usage_record(
                timestamp=timestamp,
                consumption_kwh=consumption,
                device_type=device_type,
                device_name=name,
                cost_usd=cost,
            )
            records_created += 1

print(f"Created {records_created} energy usage records")

# =============================================================================
# 5. Generate Sample Solar Generation Data (past 30 days)
# =============================================================================
# Weather conditions affect generation via a multiplier; weather is chosen
# per-day with weighted probabilities.
weather_conditions = {
    'sunny':         {'multiplier': 1.0, 'probability': 0.4},
    'partly_cloudy': {'multiplier': 0.6, 'probability': 0.3},
    'cloudy':        {'multiplier': 0.3, 'probability': 0.2},
    'rainy':         {'multiplier': 0.1, 'probability': 0.1},
}

start_date = datetime.now() - timedelta(days=30)
generation_records = 0

for day in range(31):
    current_date = start_date + timedelta(days=day)

    weather_choice = random.choices(
        list(weather_conditions.keys()),
        weights=[w['probability'] for w in weather_conditions.values()],
    )[0]
    weather_multiplier = weather_conditions[weather_choice]['multiplier']

    for hour in range(24):
        # Solar only during daylight hours (6 AM–6 PM)
        if not (6 <= hour <= 18):
            continue

        timestamp = current_date.replace(hour=hour, minute=0, second=0, microsecond=0)

        hour_factor = 1 - abs(hour - 12) / 6  # peak at noon
        base_generation = 5.0 * hour_factor    # max 5 kWh at peak
        generation = max(0, base_generation * weather_multiplier * random.uniform(0.8, 1.2))

        base_temp = 20 + random.uniform(-5, 5)
        temp_factor = 1.0 if 15 <= base_temp <= 35 else 0.9
        irradiance = 800 * hour_factor * weather_multiplier if generation > 0 else 0

        db_manager.add_generation_record(
            timestamp=timestamp,
            generation_kwh=generation,
            weather_condition=weather_choice,
            temperature_c=base_temp * temp_factor,
            solar_irradiance=irradiance,
        )
        generation_records += 1

print(f"Created {generation_records} solar generation records")

# =============================================================================
# 6. Query and Analyze Data
# =============================================================================
recent_usage = db_manager.get_recent_usage(24)
recent_generation = db_manager.get_recent_generation(24)

print("\n=== Energy Usage Analysis ===")
print(f"Total records in last 24 hours: {len(recent_usage)}")

device_consumption: dict = {}
for record in recent_usage:
    device = record.device_type or 'unknown'
    if device not in device_consumption:
        device_consumption[device] = {'kwh': 0, 'cost': 0, 'records': 0}
    device_consumption[device]['kwh'] += record.consumption_kwh
    device_consumption[device]['cost'] += record.cost_usd or 0
    device_consumption[device]['records'] += 1

print("\nConsumption by device type:")
for device, data in device_consumption.items():
    print(f"  {device}: {data['kwh']:.2f} kWh, ${data['cost']:.2f}, {data['records']} records")

print("\n=== Solar Generation Analysis ===")
print(f"Total generation records in last 24 hours: {len(recent_generation)}")
total_generation = sum(r.generation_kwh for r in recent_generation)
print(f"Total generation: {total_generation:.2f} kWh")

weather_breakdown: dict = {}
for record in recent_generation:
    weather = record.weather_condition or 'unknown'
    if weather not in weather_breakdown:
        weather_breakdown[weather] = {'kwh': 0, 'records': 0}
    weather_breakdown[weather]['kwh'] += record.generation_kwh
    weather_breakdown[weather]['records'] += 1

print("\nGeneration by weather condition:")
for weather, data in weather_breakdown.items():
    print(f"  {weather}: {data['kwh']:.2f} kWh, {data['records']} records")

# =============================================================================
# 7. Test Database Tools
# =============================================================================
from tools import get_recent_energy_summary, query_energy_usage, query_solar_generation

end_date = datetime.now().strftime("%Y-%m-%d")
start_date_str = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

print("\n=== Testing Database Tools ===")
print(f"Querying data from {start_date_str} to {end_date}")

usage_data = query_energy_usage.invoke({"start_date": start_date_str, "end_date": end_date})
print(f"\nEnergy Usage Query Results:")
print(f"  Total records: {usage_data['total_records']}")
print(f"  Total consumption: {usage_data['total_consumption_kwh']} kWh")
print(f"  Total cost: ${usage_data['total_cost_usd']}")

generation_data = query_solar_generation.invoke({"start_date": start_date_str, "end_date": end_date})
print(f"\nSolar Generation Query Results:")
print(f"  Total records: {generation_data['total_records']}")
print(f"  Total generation: {generation_data['total_generation_kwh']} kWh")
print(f"  Average daily: {generation_data['average_daily_generation']} kWh")

summary = get_recent_energy_summary.invoke({"hours": 720})
print(f"\nRecent Energy Summary:")
print(f"  Usage: {summary['usage']['total_consumption_kwh']} kWh, ${summary['usage']['total_cost_usd']}")
print(f"  Generation: {summary['generation']['total_generation_kwh']} kWh")
print(f"  Weather: {summary['generation']['average_weather']}")
