import os
import math
from datetime import datetime, timezone
from typing import Optional
import httpx

ARCHIVE_URL = os.getenv("OPEN_METEO_ARCHIVE_URL", "https://archive-api.open-meteo.com/v1/archive")
FORECAST_URL = os.getenv("OPEN_METEO_FORECAST_URL", "https://api.open-meteo.com/v1/forecast")


async def fetch_meteorological_context(
    lat: Optional[float],
    lon: Optional[float],
    timestamp_iso: Optional[str],
) -> Optional[dict]:
    """
    Fetches temperature, dew point, humidity, shortwave radiation, and
    multi-level wind vectors (10m vs 80m/120m) to compute boundary-layer wind shear.
    """
    if lat is None or lon is None or timestamp_iso is None:
        return None

    dt = datetime.fromisoformat(timestamp_iso)
    date_str = dt.strftime("%Y-%m-%d")
    hour_idx = dt.hour

    # Use forecast endpoint for recent/today dates, archive for historical
    days_ago = (datetime.now(timezone.utc) - dt).days
    base_url = FORECAST_URL if days_ago <= 2 else ARCHIVE_URL

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date_str,
        "end_date": date_str,
        "hourly": ",".join([
            "temperature_2m",
            "dew_point_2m",
            "relative_humidity_2m",
            "shortwave_radiation",
            "wind_speed_10m",
            "wind_direction_10m",
            "wind_speed_80m",
            "wind_direction_80m",
            "cape",
        ]),
        "wind_speed_unit": "ms",
        "timezone": "UTC",
    }

    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(base_url, params=params)
            resp.raise_for_status()
            hourly = resp.json().get("hourly", {})

        # Compute vector wind shear between 10m and 80m
        u10 = -hourly["wind_speed_10m"][hour_idx] * math.sin(math.radians(hourly["wind_direction_10m"][hour_idx]))
        v10 = -hourly["wind_speed_10m"][hour_idx] * math.cos(math.radians(hourly["wind_direction_10m"][hour_idx]))
        u80 = -hourly["wind_speed_80m"][hour_idx] * math.sin(math.radians(hourly["wind_direction_80m"][hour_idx]))
        v80 = -hourly["wind_speed_80m"][hour_idx] * math.cos(math.radians(hourly["wind_direction_80m"][hour_idx]))
        shear_ms = round(math.sqrt((u80 - u10) ** 2 + (v80 - v10) ** 2), 2)

        return {
            "ambient_temp_c": hourly["temperature_2m"][hour_idx],
            "dew_point_c": hourly["dew_point_2m"][hour_idx],
            "relative_humidity_pct": hourly["relative_humidity_2m"][hour_idx],
            "shortwave_radiation_wm2": hourly["shortwave_radiation"][hour_idx],
            "wind_speed_10m_ms": hourly["wind_speed_10m"][hour_idx],
            "wind_direction_10m_deg": hourly["wind_direction_10m"][hour_idx],
            "boundary_layer_shear_ms": shear_ms,
            "cape_jkg": hourly.get("cape", [0])[hour_idx] or 0.0,
        }
    except Exception:
        return None
