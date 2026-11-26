from typing import Optional
from api.schemas import (
    AtmosphericReport,
    ImageMetadata,
    SolarMetrics,
    CloudAndDynamics,
    Thermodynamics,
)


def synthesize_atmosphere(
    meta: dict,
    seg: dict,
    clouds: dict,
    optics: dict,
    weather: Optional[dict],
) -> AtmosphericReport:
    notes = []
    mode = "FUSED_TELEMETRY" if weather else "VISION_ONLY"

    if not weather:
        notes.append("EXIF GPS/Timestamp missing or API unavailable: running in Vision-Only heuristic mode.")

    # 1. Radiation synthesis (attenuate clear-sky irradiance by cloud cover if telemetry lacks value)
    cloud_frac = clouds["cloud_cover_pct"] / 100.0
    est_irradiance = None
    if weather and weather.get("shortwave_radiation_wm2") is not None:
        est_irradiance = weather["shortwave_radiation_wm2"]
    elif meta.get("clear_sky_irradiance_wm2") is not None:
        # Kasten-Czeplak cloud attenuation formula: G = G_0 * (1 - 0.75 * N^3.4)
        est_irradiance = round(meta["clear_sky_irradiance_wm2"] * (1.0 - 0.75 * (cloud_frac**3.4)), 1)

    # 2. Instability & Turbulence synthesis
    genera_str = " ".join(clouds["genera"])
    cape = weather.get("cape_jkg", 0.0) if weather else 0.0

    if "Cumulonimbus" in genera_str or cape > 1000:
        instability = "High (Convective / Deep Instability)"
        turbulence = "Severe (Strong Updrafts & Wind Shear)"
    elif "Cumulus" in genera_str or cape > 250:
        instability = "Moderate (Boundary-Layer Thermal Mixing)"
        turbulence = "Light to Moderate (Thermal Bumps)"
    elif clouds["billow_wave_score"] > 0.65:
        instability = "Conditional (Kelvin-Helmholtz Shear Layer)"
        turbulence = "Moderate to Severe (Clear-Air / Wave Turbulence)"
    else:
        instability = "Low (Stratified / Stable Air Mass)"
        turbulence = "Minimal (Smooth Laminar Flow)"

    # 3. Wind Shear & Direction
    shear_val = weather["boundary_layer_shear_ms"] if weather else None
    if shear_val is not None:
        shear_label = f"{shear_val} m/s (10m–80m vector delta)"
    elif clouds["anisotropy_angle_deg"] is not None:
        shear_label = f"Visual cloud-streak alignment at {clouds['anisotropy_angle_deg']}° image axis"
    else:
        shear_label = "Indeterminate from static image"

    # 4. Water Phases
    phases = list(optics["water_phases_present"])
    if weather and weather["ambient_temp_c"] <= 0.0 and "Solid (Snow / Ice / Frost)" not in phases:
        phases.append("Sub-zero Environment (Ice Nuclei Favored)")

    return AtmosphericReport(
        mode=mode,
        metadata=ImageMetadata(
            has_gps=meta["has_gps"],
            latitude=meta["latitude"],
            longitude=meta["longitude"],
            timestamp_utc=meta["timestamp_utc"],
            exposure_value_100=meta["exposure_value_100"],
        ),
        solar=SolarMetrics(
            elevation_deg=meta["elevation_deg"],
            azimuth_deg=meta["azimuth_deg"],
            dominant_scattering=optics["dominant_scattering"],
            clear_sky_irradiance_wm2=meta["clear_sky_irradiance_wm2"],
            estimated_surface_irradiance_wm2=est_irradiance,
        ),
        dynamics=CloudAndDynamics(
            sky_fraction=seg["sky_fraction"],
            cloud_cover_pct=clouds["cloud_cover_pct"],
            detected_genera=clouds["genera"],
            atmospheric_instability=instability,
            turbulence_risk=turbulence,
            wind_shear_indicator=shear_label,
            surface_wind_speed_ms=weather["wind_speed_10m_ms"] if weather else None,
            wind_direction_deg=weather["wind_direction_10m_deg"] if weather else None,
            boundary_layer_shear_ms=shear_val,
        ),
        thermodynamics=Thermodynamics(
            ambient_temp_c=weather["ambient_temp_c"] if weather else None,
            dew_point_c=weather["dew_point_c"] if weather else None,
            relative_humidity_pct=weather["relative_humidity_pct"] if weather else None,
            visibility_estimate=optics["visibility_estimate"],
            water_phases_present=phases,
        ),
        confidence_notes=notes,
    )
