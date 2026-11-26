from typing import Optional
import cv2
import numpy as np

from core.exif_solar import extract_metadata_and_sun
from core.weather_fusion import fetch_meteorological_context
from core.physics_heuristics import synthesize_atmosphere
from models.sky_segmenter import segment_sky_and_surface
from models.cloud_classifier import classify_clouds_and_morphology
from models.optical_analyzer import analyze_optical_and_surface
from api.schemas import AtmosphericReport


async def run_atmospheric_analysis(
    image_bytes: bytes,
    override_lat: Optional[float] = None,
    override_lon: Optional[float] = None,
    override_timestamp_iso: Optional[str] = None,
) -> AtmosphericReport:
    # 1. Decode image buffer
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Unable to decode image bytes.")

    # 2. Extract EXIF GPS, Timestamp, Exposure & Solar Geometry
    meta = extract_metadata_and_sun(
        image_bytes=image_bytes,
        override_lat=override_lat,
        override_lon=override_lon,
        override_timestamp_iso=override_timestamp_iso,
    )

    # 3. Computer Vision Pipeline
    seg = segment_sky_and_surface(image_bgr)
    clouds = classify_clouds_and_morphology(image_bgr, seg["sky_mask"])
    optics = analyze_optical_and_surface(image_bgr, seg["sky_mask"], seg["ground_mask"])

    # 4. Meteorological Reanalysis / Forecast Anchor
    weather = await fetch_meteorological_context(
        lat=meta["latitude"],
        lon=meta["longitude"],
        timestamp_iso=meta["timestamp_utc"],
    )

    # 5. Synthesize Full Atmospheric Report
    return synthesize_atmosphere(meta, seg, clouds, optics, weather)
