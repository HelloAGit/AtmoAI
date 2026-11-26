import io
import math
from datetime import datetime, timezone
from typing import Optional
import exifread
from astral import LocationInfo
from astral.sun import elevation, azimuth


def _convert_to_degrees(value) -> float:
    d = float(value.values[0].num) / float(value.values[0].den)
    m = float(value.values[1].num) / float(value.values[1].den)
    s = float(value.values[2].num) / float(value.values[2].den)
    return d + (m / 60.0) + (s / 3600.0)


def extract_metadata_and_sun(
    image_bytes: bytes,
    override_lat: Optional[float] = None,
    override_lon: Optional[float] = None,
    override_timestamp_iso: Optional[str] = None,
) -> dict:
    tags = exifread.process_file(io.BytesIO(image_bytes), details=False)

    lat = override_lat
    lon = override_lon
    dt_utc = None

    # Extract GPS if not overridden
    if lat is None and "GPS GPSLatitude" in tags and "GPS GPSLatitudeRef" in tags:
        lat = _convert_to_degrees(tags["GPS GPSLatitude"])
        if str(tags["GPS GPSLatitudeRef"]) != "N":
            lat = -lat

    if lon is None and "GPS GPSLongitude" in tags and "GPS GPSLongitudeRef" in tags:
        lon = _convert_to_degrees(tags["GPS GPSLongitude"])
        if str(tags["GPS GPSLongitudeRef"]) != "E":
            lon = -lon

    # Extract Timestamp
    if override_timestamp_iso:
        dt_utc = datetime.fromisoformat(override_timestamp_iso.replace("Z", "+00:00")).astimezone(timezone.utc)
    elif "EXIF DateTimeOriginal" in tags:
        raw_dt = str(tags["EXIF DateTimeOriginal"])
        dt_utc = datetime.strptime(raw_dt, "%Y:%m:%d %H:%M:%S").replace(tzinfo=timezone.utc)

    # Calculate Exposure Value (EV_100) for irradiance cross-check
    ev_100 = None
    try:
        f_num = float(tags["EXIF FNumber"].values[0].num) / float(tags["EXIF FNumber"].values[0].den)
        exp_time = float(tags["EXIF ExposureTime"].values[0].num) / float(tags["EXIF ExposureTime"].values[0].den)
        iso = float(str(tags["EXIF ISOSpeedRatings"]))
        ev_100 = round(math.log2((f_num**2) / exp_time) - math.log2(iso / 100.0), 2)
    except Exception:
        pass

    # Calculate Sun Angle & Theoretical Clear-Sky Radiation
    sun_elev = None
    sun_azim = None
    clear_sky_wm2 = None

    if lat is not None and lon is not None and dt_utc is not None:
        loc = LocationInfo(latitude=lat, longitude=lon)
        sun_elev = round(elevation(loc.observer, dt_utc), 2)
        sun_azim = round(azimuth(loc.observer, dt_utc), 2)

        # Kasten-Czeplak clear-sky global horizontal irradiance approximation
        if sun_elev > 0:
            sin_el = math.sin(math.radians(sun_elev))
            clear_sky_wm2 = round(910.0 * sin_el - 30.0, 1)
        else:
            clear_sky_wm2 = 0.0

    return {
        "has_gps": lat is not None and lon is not None,
        "latitude": lat,
        "longitude": lon,
        "timestamp_utc": dt_utc.isoformat() if dt_utc else None,
        "exposure_value_100": ev_100,
        "elevation_deg": sun_elev,
        "azimuth_deg": sun_azim,
        "clear_sky_irradiance_wm2": clear_sky_wm2,
    }
