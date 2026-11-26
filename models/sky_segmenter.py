import cv2
import numpy as np


def segment_sky_and_surface(image_bgr: np.ndarray) -> dict:
    """
    Segments sky pixels from terrain/surface pixels using HSV color space,
    vertical spatial priors, and edge-gradient smoothness.
    (Can be swapped with a fine-tuned SegFormer model in production.)
    """
    h, w, _ = image_bgr.shape
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Sky regions typically have low texture variance (smooth gradients) and upper-frame bias
    laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
    smooth_mask = (laplacian < 18.0).astype(np.uint8)

    # Color criteria: Blue sky OR bright/overcast white-grey clouds
    lower_blue = np.array([90, 35, 70])
    upper_blue = np.array([135, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    lower_cloud = np.array([0, 0, 150])
    upper_cloud = np.array([180, 60, 255])
    cloud_mask = cv2.inRange(hsv, lower_cloud, upper_cloud)

    candidate_sky = cv2.bitwise_or(blue_mask, cloud_mask)
    candidate_sky = cv2.bitwise_and(candidate_sky, candidate_sky, mask=smooth_mask)

    # Apply vertical spatial prior (upper 75% of image weighted heavily)
    y_grid = np.linspace(1.0, 0.0, h)[:, None]
    spatial_weight = np.repeat(y_grid, w, axis=1)
    sky_prob = (candidate_sky / 255.0) * (spatial_weight > 0.25)

    sky_mask = (sky_prob > 0.5).astype(np.uint8) * 255

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_CLOSE, kernel)
    ground_mask = cv2.bitwise_not(sky_mask)

    sky_fraction = float(np.sum(sky_mask > 0) / (h * w))
    return {
        "sky_mask": sky_mask,
        "ground_mask": ground_mask,
        "sky_fraction": round(sky_fraction, 3),
    }
