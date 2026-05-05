"""
ClarifEye Color Verifier

Secondary confirmation step applied *after* neural-network detection.
Crops the bounding-box region, converts it to HSV, and counts pixels that
fall within the per-color ranges defined in ``config``.

Design notes
~~~~~~~~~~~~
* Frames arrive as **RGB** from ``CameraModule`` → converted with
  ``cv2.COLOR_RGB2HSV`` (not BGR2HSV).
* Red spans two disjoint hue arcs in OpenCV HSV (0-10° and 170-180°);
  both masks are combined before counting.
* Returns the color whose matching-pixel ratio exceeds
  ``config.COLOR_PIXEL_RATIO_THRESHOLD``, or ``None`` / ``"none"`` when no
  color is sufficiently dominant.
"""
import logging
from typing import Dict, Optional, Tuple

import numpy as np

try:
    from .. import config
except ImportError:
    import config  # type: ignore[no-redef]

try:
    import cv2  # type: ignore[import-untyped]
    _OPENCV_AVAILABLE = True
except ImportError:
    _OPENCV_AVAILABLE = False

logger = logging.getLogger("clarifeye.ai.color_verifier")


class ColorVerifier:
    """
    HSV pixel-counting color verifier for traffic light bounding boxes.

    Can also be used as a general-purpose dominant-color identifier for any
    region of interest in the frame.
    """

    def verify_traffic_light_color(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> Optional[str]:
        """
        Confirm the traffic light color inside *bbox* via pixel counting.

        Args:
            frame: Full-frame **RGB** ``numpy.ndarray`` from the camera module.
            bbox:  ``(x1, y1, x2, y2)`` bounding box in frame pixel coordinates.

        Returns:
            ``"red"``, ``"yellow"``, or ``"green"`` if one color is dominant
            above ``config.COLOR_PIXEL_RATIO_THRESHOLD``; ``None`` otherwise.
        """
        color, _ = self.get_dominant_color(frame, bbox)
        return color if color != "none" else None

    def get_dominant_color(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> Tuple[str, float]:
        """
        Return the dominant color and its pixel ratio inside *bbox*.

        Analyses red, yellow, and green simultaneously and reports whichever
        has the highest matching-pixel fraction — provided that fraction
        clears ``config.COLOR_PIXEL_RATIO_THRESHOLD``.

        Args:
            frame: Full-frame **RGB** ``numpy.ndarray``.
            bbox:  ``(x1, y1, x2, y2)`` bounding box.

        Returns:
            ``(color_name, ratio)`` where *color_name* is one of
            ``"red"``, ``"yellow"``, ``"green"``, or ``"none"`` and *ratio*
            is the best matching fraction in ``[0.0, 1.0]``.
        """
        if not _OPENCV_AVAILABLE:
            logger.error("OpenCV is unavailable — color verification disabled.")
            return "none", 0.0

        x1, y1, x2, y2 = bbox
        crop: np.ndarray = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return "none", 0.0

        total_pixels: int = crop.shape[0] * crop.shape[1]
        if total_pixels == 0:
            return "none", 0.0

        try:
            # Frame is RGB — use COLOR_RGB2HSV directly.
            hsv: np.ndarray = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)

            ratios: Dict[str, float] = {
                "red": self._count_red(hsv) / total_pixels,
                "yellow": self._count_single_range(
                    hsv,
                    config.HSV_YELLOW_LOWER,
                    config.HSV_YELLOW_UPPER,
                ) / total_pixels,
                "green": self._count_single_range(
                    hsv,
                    config.HSV_GREEN_LOWER,
                    config.HSV_GREEN_UPPER,
                ) / total_pixels,
            }

            best_color: str = max(ratios, key=lambda k: ratios[k])
            best_ratio: float = ratios[best_color]

            if best_ratio < config.COLOR_PIXEL_RATIO_THRESHOLD:
                return "none", best_ratio

            return best_color, round(best_ratio, 4)

        except Exception as exc:
            logger.error("get_dominant_color error: %s", exc)
            return "none", 0.0

    # ── HSV counting helpers ───────────────────────────────────────────────────

    def _count_red(self, hsv: np.ndarray) -> int:
        """
        Count red pixels by combining both HSV hue arcs for red.

        Red wraps around the 0°/180° boundary in OpenCV HSV space, requiring
        two separate ``cv2.inRange`` calls whose results are summed.

        Args:
            hsv: HSV image array (output of ``cv2.cvtColor(…, COLOR_RGB2HSV)``).

        Returns:
            Total number of red pixels in the image.
        """
        mask_lower = cv2.inRange(
            hsv,
            np.array(config.HSV_RED_LOWER_1, dtype=np.uint8),
            np.array(config.HSV_RED_UPPER_1, dtype=np.uint8),
        )
        mask_upper = cv2.inRange(
            hsv,
            np.array(config.HSV_RED_LOWER_2, dtype=np.uint8),
            np.array(config.HSV_RED_UPPER_2, dtype=np.uint8),
        )
        return int(cv2.countNonZero(mask_lower)) + int(cv2.countNonZero(mask_upper))

    def _count_single_range(
        self,
        hsv: np.ndarray,
        lower: Tuple[int, int, int],
        upper: Tuple[int, int, int],
    ) -> int:
        """
        Count pixels within a single contiguous HSV range.

        Args:
            hsv:   HSV image array.
            lower: ``(H_min, S_min, V_min)`` lower bound.
            upper: ``(H_max, S_max, V_max)`` upper bound.

        Returns:
            Number of pixels inside the range.
        """
        mask = cv2.inRange(
            hsv,
            np.array(lower, dtype=np.uint8),
            np.array(upper, dtype=np.uint8),
        )
        return int(cv2.countNonZero(mask))
