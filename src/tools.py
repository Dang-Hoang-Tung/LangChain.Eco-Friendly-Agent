"""
Tools for EcoHome Energy Advisor Agent
"""
import os
import sys
import json
import random
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

_SRC_DIR = Path(__file__).resolve().parent
_DATA_DIR = _SRC_DIR / "data"
sys.path.insert(0, str(_SRC_DIR))

from models.energy import DatabaseManager

# Initialize database manager
db_manager = DatabaseManager()

@tool
def get_weather_forecast(location: str, days: int = 3) -> Dict[str, Any]:
    """
    Get weather forecast for a specific location and number of days using Open-Meteo API.

    Args:
        location (str): Location to get weather for (e.g., "San Francisco, CA")
        days (int): Number of days to forecast (1-7)

    Returns:
        Dict[str, Any]: Weather forecast data including temperature, conditions, and solar irradiance
    """
    import requests

    def _wmo_to_condition(code: int) -> str:
        """Map WMO weather code to a human-readable condition string."""
        if code == 0:
            return "sunny"
        elif code in (1, 2):
            return "partly_cloudy"
        elif code in (3, 45, 48):
            return "cloudy"
        else:  # 51-67 drizzle/rain, 71-77 snow, 80-86 showers, 95+ thunderstorm
            return "rainy"

    try:
        # Step 1: Geocode location string → lat/lng via Open-Meteo Geocoding API
        # Use only the city portion (before any comma) so the API resolves correctly
        city_name = location.split(",")[0].strip()
        geo_resp = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city_name, "count": 1, "language": "en", "format": "json"},
            timeout=10,
        )
        geo_resp.raise_for_status()
        geo_results = geo_resp.json().get("results")
        if not geo_results:
            return {"error": f"Location not found: '{location}'"}

        geo = geo_results[0]
        lat, lng = geo["latitude"], geo["longitude"]
        resolved = f"{geo['name']}, {geo.get('admin1', '')}, {geo.get('country', '')}".strip(", ")

        # Step 2: Fetch hourly forecast via Open-Meteo Forecast API
        fc_resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lng,
                "hourly": "temperature_2m,relativehumidity_2m,windspeed_10m,shortwave_radiation,weathercode",
                "forecast_days": min(max(days, 1), 7),
                "timezone": "auto",
            },
            timeout=10,
        )
        fc_resp.raise_for_status()
        hourly = fc_resp.json()["hourly"]

        times      = hourly["time"]
        temps      = hourly["temperature_2m"]
        humidity   = hourly["relativehumidity_2m"]
        wind       = hourly["windspeed_10m"]
        irradiance = hourly["shortwave_radiation"]
        codes      = hourly["weathercode"]

        # Current conditions from the first returned hour
        current = {
            "temperature_c": temps[0],
            "condition":     _wmo_to_condition(codes[0]),
            "humidity":      humidity[0],
            "wind_speed":    wind[0],
        }

        hourly_list = [
            {
                "datetime":        time_str,
                "hour":            int(time_str[11:13]),
                "temperature_c":   temps[i],
                "condition":       _wmo_to_condition(codes[i]),
                "solar_irradiance": irradiance[i],
                "humidity":        humidity[i],
                "wind_speed":      wind[i],
            }
            for i, time_str in enumerate(times)
        ]

        return {
            "location":     resolved,
            "forecast_days": days,
            "current":      current,
            "hourly":       hourly_list,
        }

    except requests.exceptions.RequestException as e:
        return {"error": f"Weather API request failed: {str(e)}"}
    except Exception as e:
        return {"error": f"Failed to get weather forecast: {str(e)}"}

@tool
def get_electricity_prices(date: Optional[str] = None) -> Dict[str, Any]:
    """
    Get electricity prices for a specific date or current day.

    Fetches real UK half-hourly prices from the Octopus Energy Agile tariff API
    (https://developer.octopus.energy/docs/api/) and aggregates them into hourly
    rates.  No API key required — the Agile product rates are publicly accessible.

    Agile prices are published ~4 pm UTC for the following day, so future dates
    fall back to typical UK TOU rates.

    Args:
        date (str): Date in YYYY-MM-DD format (defaults to today)

    Returns:
        Dict[str, Any]: Electricity pricing data with hourly rates
    """
    import requests

    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    # --- Octopus Energy Agile tariff (UK, no auth required) ---
    # Agile prices vary every 30 minutes, linked to day-ahead wholesale market.
    # Region A = Eastern England (representative UK average).
    # Docs: https://developer.octopus.energy/docs/api/#agile-octopus
    PRODUCT = "AGILE-24-10-01"
    TARIFF  = f"E-1R-{PRODUCT}-A"
    period_from = f"{date}T00:00Z"
    period_to   = (datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%dT%H:%MZ")

    def _period_label(hour: int) -> str:
        """Classify hour into a TOU period label."""
        if hour < 7 or hour >= 23:
            return "off_peak"
        elif 16 <= hour < 19:
            return "peak"
        else:
            return "mid_peak"

    try:
        resp = requests.get(
            f"https://api.octopus.energy/v1/products/{PRODUCT}/electricity-tariffs/{TARIFF}/standard-unit-rates/",
            params={"period_from": period_from, "period_to": period_to, "page_size": 50},
            timeout=10,
        )
        resp.raise_for_status()
        slots = resp.json().get("results", [])

        if slots:
            # Group the 30-minute slots by UTC hour, then average into hourly rates
            hour_buckets: Dict[int, list] = {h: [] for h in range(24)}
            for slot in slots:
                hour = int(slot["valid_from"][11:13])  # "2026-03-01T06:30:00Z" → 6
                hour_buckets[hour].append(slot["value_inc_vat"])  # pence/kWh inc. 5% VAT

            hourly_rates = []
            for hour in range(24):
                bucket = hour_buckets[hour]
                rate_pence = round(sum(bucket) / len(bucket), 2) if bucket else None
                hourly_rates.append({
                    "hour":            hour,
                    "rate":            round(rate_pence / 100, 4) if rate_pence is not None else None,
                    "rate_pence_kwh":  rate_pence,
                    "period":          _period_label(hour),
                    "demand_charge":   0.0,
                })

            return {
                "date":         date,
                "pricing_type": "agile_half_hourly",
                "source":       "Octopus Energy Agile tariff (UK)",
                "currency":     "GBP",
                "unit":         "per_kWh",
                "region":       "Eastern England",
                "hourly_rates": hourly_rates,
            }
    except Exception:
        pass

    # --- Fallback: typical UK TOU average rates (GBP/kWh, 2024) ---
    # Used when Agile prices are not yet published (e.g. future dates).
    UK_RATES = {"off_peak": 0.15, "mid_peak": 0.28, "peak": 0.40}
    rng = random.Random(date)

    hourly_rates = []
    for hour in range(24):
        period = _period_label(hour)
        base = UK_RATES[period]
        noise = rng.uniform(-0.03, 0.03) * base
        rate = round(base + noise, 4)
        hourly_rates.append({
            "hour":           hour,
            "rate":           rate,
            "rate_pence_kwh": round(rate * 100, 2),
            "period":         period,
            "demand_charge":  0.0,
        })

    return {
        "date":         date,
        "pricing_type": "time_of_use",
        "source":       "UK average TOU rates (fallback)",
        "currency":     "GBP",
        "unit":         "per_kWh",
        "hourly_rates": hourly_rates,
    }

@tool
def query_energy_usage(start_date: str, end_date: str, device_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Query energy usage data from the database for a specific date range.
    
    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        device_type (str): Optional device type filter (e.g., "EV", "HVAC", "appliance")
    
    Returns:
        Dict[str, Any]: Energy usage data with consumption details
    """
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
        
        records = db_manager.get_usage_by_date_range(start_dt, end_dt)
        
        if device_type:
            records = [r for r in records if r.device_type == device_type]
        
        usage_data = {
            "start_date": start_date,
            "end_date": end_date,
            "device_type": device_type,
            "total_records": len(records),
            "total_consumption_kwh": round(sum(r.consumption_kwh for r in records), 2),
            "total_cost_usd": round(sum(r.cost_usd or 0 for r in records), 2),
            "records": []
        }
        
        for record in records:
            usage_data["records"].append({
                "timestamp": record.timestamp.isoformat(),
                "consumption_kwh": record.consumption_kwh,
                "device_type": record.device_type,
                "device_name": record.device_name,
                "cost_usd": record.cost_usd
            })
        
        return usage_data
    except Exception as e:
        return {"error": f"Failed to query energy usage: {str(e)}"}

@tool
def query_solar_generation(start_date: str, end_date: str) -> Dict[str, Any]:
    """
    Query solar generation data from the database for a specific date range.
    
    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
    
    Returns:
        Dict[str, Any]: Solar generation data with production details
    """
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
        
        records = db_manager.get_generation_by_date_range(start_dt, end_dt)
        
        generation_data = {
            "start_date": start_date,
            "end_date": end_date,
            "total_records": len(records),
            "total_generation_kwh": round(sum(r.generation_kwh for r in records), 2),
            "average_daily_generation": round(sum(r.generation_kwh for r in records) / max(1, (end_dt - start_dt).days), 2),
            "records": []
        }
        
        for record in records:
            generation_data["records"].append({
                "timestamp": record.timestamp.isoformat(),
                "generation_kwh": record.generation_kwh,
                "weather_condition": record.weather_condition,
                "temperature_c": record.temperature_c,
                "solar_irradiance": record.solar_irradiance
            })
        
        return generation_data
    except Exception as e:
        return {"error": f"Failed to query solar generation: {str(e)}"}

@tool
def get_recent_energy_summary(hours: int = 24) -> Dict[str, Any]:
    """
    Get a summary of recent energy usage and solar generation.
    
    Args:
        hours (int): Number of hours to look back (default 24)
    
    Returns:
        Dict[str, Any]: Summary of recent energy data
    """
    try:
        usage_records = db_manager.get_recent_usage(hours)
        generation_records = db_manager.get_recent_generation(hours)
        
        summary = {
            "time_period_hours": hours,
            "usage": {
                "total_consumption_kwh": round(sum(r.consumption_kwh for r in usage_records), 2),
                "total_cost_usd": round(sum(r.cost_usd or 0 for r in usage_records), 2),
                "device_breakdown": {}
            },
            "generation": {
                "total_generation_kwh": round(sum(r.generation_kwh for r in generation_records), 2),
                "average_weather": "sunny" if generation_records else "unknown"
            }
        }
        
        # Calculate device breakdown
        for record in usage_records:
            device = record.device_type or "unknown"
            if device not in summary["usage"]["device_breakdown"]:
                summary["usage"]["device_breakdown"][device] = {
                    "consumption_kwh": 0,
                    "cost_usd": 0,
                    "records": 0
                }
            summary["usage"]["device_breakdown"][device]["consumption_kwh"] += record.consumption_kwh
            summary["usage"]["device_breakdown"][device]["cost_usd"] += record.cost_usd or 0
            summary["usage"]["device_breakdown"][device]["records"] += 1
        
        # Round the breakdown values
        for device_data in summary["usage"]["device_breakdown"].values():
            device_data["consumption_kwh"] = round(device_data["consumption_kwh"], 2)
            device_data["cost_usd"] = round(device_data["cost_usd"], 2)
        
        return summary
    except Exception as e:
        return {"error": f"Failed to get recent energy summary: {str(e)}"}

@tool
def search_energy_tips(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Search for energy-saving tips and best practices using RAG.
    
    Args:
        query (str): Search query for energy tips
        max_results (int): Maximum number of results to return
    
    Returns:
        Dict[str, Any]: Relevant energy tips and best practices
    """
    try:
        # Initialize vector store if it doesn't exist
        persist_directory = str(_DATA_DIR / "vectorstore")
        if not os.path.exists(persist_directory):
            os.makedirs(persist_directory)

        # Load documents if vector store doesn't exist
        if not os.path.exists(os.path.join(persist_directory, "chroma.sqlite3")):
            # Load documents
            documents = []
            for doc_path in [str(_DATA_DIR / "documents" / "tip_device_best_practices.txt"),
                             str(_DATA_DIR / "documents" / "tip_energy_savings.txt")]:
                if os.path.exists(doc_path):
                    loader = TextLoader(doc_path)
                    docs = loader.load()
                    documents.extend(docs)
            
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(documents)
            
            # Create vector store
            embeddings = OpenAIEmbeddings()
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                persist_directory=persist_directory
            )
        else:
            # Load existing vector store
            embeddings = OpenAIEmbeddings()
            vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings
            )
        
        # Search for relevant documents
        docs = vectorstore.similarity_search(query, k=max_results)
        
        results = {
            "query": query,
            "total_results": len(docs),
            "tips": []
        }
        
        for i, doc in enumerate(docs):
            results["tips"].append({
                "rank": i + 1,
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "relevance_score": "high" if i < 2 else "medium" if i < 4 else "low"
            })
        
        return results
    except Exception as e:
        return {"error": f"Failed to search energy tips: {str(e)}"}

@tool
def calculate_energy_savings(device_type: str, current_usage_kwh: float, 
                           optimized_usage_kwh: float, price_per_kwh: float = 0.12) -> Dict[str, Any]:
    """
    Calculate potential energy savings from optimization.
    
    Args:
        device_type (str): Type of device being optimized
        current_usage_kwh (float): Current energy usage in kWh
        optimized_usage_kwh (float): Optimized energy usage in kWh
        price_per_kwh (float): Price per kWh (default 0.12)
    
    Returns:
        Dict[str, Any]: Savings calculation results
    """
    savings_kwh = current_usage_kwh - optimized_usage_kwh
    savings_usd = savings_kwh * price_per_kwh
    savings_percentage = (savings_kwh / current_usage_kwh) * 100 if current_usage_kwh > 0 else 0
    
    return {
        "device_type": device_type,
        "current_usage_kwh": current_usage_kwh,
        "optimized_usage_kwh": optimized_usage_kwh,
        "savings_kwh": round(savings_kwh, 2),
        "savings_usd": round(savings_usd, 2),
        "savings_percentage": round(savings_percentage, 1),
        "price_per_kwh": price_per_kwh,
        "annual_savings_usd": round(savings_usd * 365, 2)
    }


TOOL_KIT = [
    get_weather_forecast,
    get_electricity_prices,
    query_energy_usage,
    query_solar_generation,
    get_recent_energy_summary,
    search_energy_tips,
    calculate_energy_savings
]
