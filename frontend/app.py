import asyncio
import streamlit as st
from core.pipeline import run_atmospheric_analysis

st.set_page_config(page_title="AtmosVision", layout="wide")
st.title("AtmosVision — Outdoor Image Atmospheric Analyzer")

uploaded = st.file_uploader("Upload an outdoor environment photo (JPG/PNG)", type=["jpg", "jpeg", "png"])

with st.expander("Optional: Override GPS & Timestamp (if stripped by messaging apps)"):
    col1, col2, col3 = st.columns(3)
    lat_in = col1.text_input("Latitude (e.g., 52.3676)", "")
    lon_in = col2.text_input("Longitude (e.g., 4.9041)", "")
    time_in = col3.text_input("UTC Timestamp ISO (e.g., 2026-07-15T14:00:00Z)", "")

if uploaded is not None:
    image_bytes = uploaded.read()
    st.image(image_bytes, caption="Uploaded Environment", use_container_width=True)

    lat = float(lat_in) if lat_in.strip() else None
    lon = float(lon_in) if lon_in.strip() else None
    ts = time_in.strip() if time_in.strip() else None

    with st.spinner("Analyzing sky segmentation, scattering, solar angle, and atmospheric dynamics..."):
        report = asyncio.run(run_atmospheric_analysis(image_bytes, lat, lon, ts))

    st.info(f"**Analysis Mode:** `{report.mode}`")
    for note in report.confidence_notes:
        st.warning(note)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader("Solar & Radiation")
        st.metric("Sun Elevation / Azimuth", f"{report.solar.elevation_deg}° / {report.solar.azimuth_deg}°")
        st.metric("Surface Irradiance", f"{report.solar.estimated_surface_irradiance_wm2} W/m²")
        st.write(f"**Scattering:** {report.solar.dominant_scattering}")

    with c2:
        st.subheader("Clouds, Instability & Wind")
        st.metric("Cloud Cover", f"{report.dynamics.cloud_cover_pct}%")
        st.write(f"**Cloud Genera:** {', '.join(report.dynamics.detected_genera)}")
        st.write(f"**Instability:** {report.dynamics.atmospheric_instability}")
        st.write(f"**Turbulence:** {report.dynamics.turbulence_risk}")
        st.write(f"**Wind Shear:** {report.dynamics.wind_shear_indicator}")
        if report.dynamics.surface_wind_speed_ms is not None:
            st.write(f"**Surface Wind:** {report.dynamics.surface_wind_speed_ms} m/s @ {report.dynamics.wind_direction_deg}°")

    with c3:
        st.subheader("Thermodynamics & Moisture")
        if report.thermodynamics.ambient_temp_c is not None:
            st.metric("Temp / Dew Point", f"{report.thermodynamics.ambient_temp_c}°C / {report.thermodynamics.dew_point_c}°C")
            st.metric("Relative Humidity", f"{report.thermodynamics.relative_humidity_pct}%")
        st.write(f"**Visibility / Moisture:** {report.thermodynamics.visibility_estimate}")
        st.write(f"**Water Phases:** {', '.join(report.thermodynamics.water_phases_present)}")
