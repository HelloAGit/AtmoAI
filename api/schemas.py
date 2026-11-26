from pydantic import BaseModel, Field
from typing import List, Optional


class ImageMetadata(BaseModel):
    has_gps: bool
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    timestamp_utc: Optional[str] = None
    exposure_value_100: Optional[float] = None


class SolarMetrics(BaseModel):
    elevation_deg: Optional[float] = None
    azimuth_deg: Optional[float] = None
    dominant_scattering: str
    clear_sky_irradiance_wm2: Optional[float] = None
    estimated_surface_irradiance_wm2: Optional[float] = None


class CloudAndDynamics(BaseModel):
    sky_fraction: float = Field(..., ge=0.0, le=1.0)
    cloud_cover_pct: float = Field(..., ge=0.0, le=100.0)
    detected_genera: List[str]
    atmospheric_instability: str
    turbulence_risk: str
    wind_shear_indicator: str
    surface_wind_speed_ms: Optional[float] = None
    wind_direction_deg: Optional[float] = None
    boundary_layer_shear_ms: Optional[float] = None


class Thermodynamics(BaseModel):
    ambient_temp_c: Optional[float] = None
    dew_point_c: Optional[float] = None
    relative_humidity_pct: Optional[float] = None
    visibility_estimate: str
    water_phases_present: List[str]


class AtmosphericReport(BaseModel):
    mode: str  # "FUSED_TELEMETRY" or "VISION_ONLY"
    metadata: ImageMetadata
    solar: SolarMetrics
    dynamics: CloudAndDynamics
    thermodynamics: Thermodynamics
    confidence_notes: List[str]
