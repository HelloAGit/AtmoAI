import cv2
import numpy as np


def classify_clouds_and_morphology(image_bgr: np.ndarray, sky_mask: np.ndarray) -> dict:
    """
    Analyzes sky pixels for cloud cover percentage, vertical development,
    fibrous streaks (cirrus/shear), and billow periodicity (Kelvin-Helmholtz/turbulence).
    """
    sky_pixels = np.sum(sky_mask > 0)
    if sky_pixels < 500:
        return {
            "cloud_cover_pct": 0.0,
            "genera": ["Obscured / No Sky Visible"],
            "texture_entropy": 0.0,
            "anisotropy_angle_deg": None,
            "billow_wave_score": 0.0,
        }

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Low saturation + high brightness inside sky_mask indicates cloud vs clear blue sky
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]

    is_cloud = (sky_mask > 0) & (s_channel < 65) & (v_channel > 135)
    cloud_cover_pct = round(float(np.sum(is_cloud) / sky_pixels) * 100.0, 1)

    # Texture & Edge orientation inside sky for shear/streak detection
    masked_gray = cv2.bitwise_and(gray, gray, mask=sky_mask)
    sobelx = cv2.Sobel(masked_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(masked_gray, cv2.CV_64F, 0, 1, ksize=3)

    mag = np.sqrt(sobelx**2 + sobely**2)
    valid_edges = mag[sky_mask > 0]
    texture_entropy = float(np.std(valid_edges)) if len(valid_edges) > 0 else 0.0

    # Dominant streak orientation (proxy for upper-level wind shear vector)
    angles = (np.degrees(np.arctan2(sobely, sobelx)) + 180) % 180
    strong_angles = angles[(sky_mask > 0) & (mag > 25)]
    anisotropy_angle = round(float(np.median(strong_angles)), 1) if len(strong_angles) > 100 else None

    # Dark base detection (Cumulonimbus / Nimbostratus vertical thickness)
    dark_cloud_base = (sky_mask > 0) & (s_channel < 50) & (v_channel >= 65) & (v_channel <= 135)
    dark_ratio = float(np.sum(dark_cloud_base) / sky_pixels)

    # Heuristic Genus Mapping (Replaceable with timm ViT checkpoint)
    genera = []
    if cloud_cover_pct < 12.0:
        genera.append("Clear Sky (SKC)")
    elif dark_ratio > 0.25 and texture_entropy > 22.0:
        genera.append("Cumulonimbus / Towering Cumulus")
    elif cloud_cover_pct > 80.0 and texture_entropy < 12.0:
        genera.append("Stratus / Nimbostratus")
    elif 20.0 <= cloud_cover_pct <= 75.0 and texture_entropy >= 16.0:
        genera.append("Cumulus Humilis / Mediocris")
    else:
        genera.append("Cirrus / Altostratus")

    # High-frequency periodic bands indicate wave turbulence / Kelvin-Helmholtz shear
    billow_score = round(min(texture_entropy / 35.0, 1.0), 2)

    return {
        "cloud_cover_pct": cloud_cover_pct,
        "genera": genera,
        "texture_entropy": round(texture_entropy, 2),
        "anisotropy_angle_deg": anisotropy_angle,
        "billow_wave_score": billow_score,
    }
