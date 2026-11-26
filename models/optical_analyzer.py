import cv2
import numpy as np


def analyze_optical_and_surface(
    image_bgr: np.ndarray,
    sky_mask: np.ndarray,
    ground_mask: np.ndarray,
) -> dict:
    """
    Evaluates Rayleigh vs. Mie scattering from sky color ratios,
    horizon haze (moisture/aerosol indicator), and snow/ice vs. liquid water on ground.
    """
    b, g, r = cv2.split(image_bgr.astype(np.float32) + 1e-5)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # 1. Rayleigh (Blue-dominant) vs Mie (White/Red aerosol/haze) scattering
    if np.sum(sky_mask > 0) > 100:
        blue_red_ratio = float(np.mean(b[sky_mask > 0] / r[sky_mask > 0]))
    else:
        blue_red_ratio = 1.0

    if blue_red_ratio > 1.35:
        scattering_type = "Rayleigh Dominant (Clean, Dry Air Mass)"
    elif blue_red_ratio > 1.05:
        scattering_type = "Mixed Rayleigh-Mie (Moderate Aerosols/Moisture)"
    else:
        scattering_type = "Mie Dominant (Hazy, High Moisture, or Overcast Scattering)"

    # 2. Horizon Haze / Visibility Proxy (Contrast decay near vertical center)
    h, w, _ = image_bgr.shape
    horizon_band = image_bgr[int(h * 0.4) : int(h * 0.6), :]
    horizon_sat = float(np.mean(cv2.cvtColor(horizon_band, cv2.COLOR_BGR2HSV)[:, :, 1]))

    if horizon_sat < 32.0:
        visibility_est = "Low / Hazy (< 5 km — High Near-Surface Moisture/Aerosol)"
    elif horizon_sat < 65.0:
        visibility_est = "Moderate (5–15 km)"
    else:
        visibility_est = "High / Crisp (> 15 km)"

    # 3. Surface Phase of Water Detection (Snow/Ice vs. Wet/Reflective Surfaces)
    phases = ["Water Vapor (Gas)"]
    if np.sum(sky_mask > 0) > 100 and np.mean(hsv[:, :, 1][sky_mask > 0]) < 65:
        phases.append("Liquid/Ice Cloud Condensate")

    if np.sum(ground_mask > 0) > 500:
        g_sat = hsv[:, :, 1][ground_mask > 0]
        g_val = hsv[:, :, 2][ground_mask > 0]
        # Bright, low-saturation ground pixels -> Snow / Frost
        snow_fraction = float(np.mean((g_sat < 35) & (g_val > 195)))
        if snow_fraction > 0.18:
            phases.append("Solid (Snow / Ice / Frost)")

    return {
        "blue_red_ratio": round(blue_red_ratio, 2),
        "dominant_scattering": scattering_type,
        "visibility_estimate": visibility_est,
        "water_phases_present": phases,
    }
