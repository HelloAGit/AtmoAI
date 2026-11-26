import pytest
import numpy as np
import cv2
from core.pipeline import run_atmospheric_analysis


@pytest.mark.asyncio
async def test_vision_only_pipeline():
    # Generate a synthetic clear blue sky (top) + green terrain (bottom) image
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    img[:220, :] = (230, 150, 40)   # BGR blue sky
    img[220:, :] = (40, 140, 50)    # BGR green ground

    _, encoded = cv2.imencode(".jpg", img)
    report = await run_atmospheric_analysis(encoded.tobytes())

    assert report.mode == "VISION_ONLY"
    assert report.dynamics.sky_fraction > 0.4
    assert "Rayleigh" in report.solar.dominant_scattering
    assert report.dynamics.cloud_cover_pct < 15.0
