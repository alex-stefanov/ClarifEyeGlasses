"""
ClarifEye Low-Light Image Enhancer

Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to the
lightness channel of dark frames before they are passed to detection or OCR.

Why CLAHE on the L channel?
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Converting RGB → LAB decouples luminance (L) from chrominance (A, B).
Equalising only L boosts local contrast without shifting hues, so detection
models that rely on colour (traffic-light HSV checks, etc.) are not confused
by the enhancement.

Performance target
~~~~~~~~~~~~~~~~~~
< 5 ms per 640×640 frame on Raspberry Pi 4 (ARM Cortex-A72, CPU only).
Achieved by:
* Reusing a single ``cv2.CLAHE`` object across all calls.
* Never allocating unnecessary intermediate arrays.
* Skipping denoising (too slow for real-time on Pi).

All frames — input and output — are RGB ``(H, W, 3)`` ``ndarray``.
"""
import logging
import time
from typing import Tuple

import cv2
import numpy as np

try:
    from .. import config
except ImportError:
    import config  # type: ignore[no-redef]

logger = logging.getLogger("clarifeye.ai.low_light_enhancer")


class LowLightEnhancer:
    """
    Stateless CLAHE-based low-light image enhancer.

    The ``cv2.CLAHE`` object is created once in ``__init__`` and reused for
    every frame, avoiding repeated object construction overhead.

    All public methods accept and return RGB ``(H, W, 3)`` ``ndarray`` images,
    consistent with the rest of the ClarifEye pipeline.
    """

    def __init__(self) -> None:
        self._clahe = cv2.createCLAHE(
            clipLimit=config.CLAHE_CLIP_LIMIT,
            tileGridSize=config.CLAHE_TILE_GRID_SIZE,
        )
        logger.debug(
            "LowLightEnhancer ready (clipLimit=%.1f, tileGridSize=%s).",
            config.CLAHE_CLIP_LIMIT,
            config.CLAHE_TILE_GRID_SIZE,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def is_low_light(self, frame: np.ndarray) -> bool:
        """
        Check whether *frame* is too dark to be processed reliably.

        Converts the RGB frame to grayscale and compares the mean pixel
        brightness against ``config.LOW_LIGHT_BRIGHTNESS_THRESHOLD``.

        Args:
            frame: RGB ``(H, W, 3)`` ``ndarray``.

        Returns:
            ``True`` if the mean brightness is below the threshold.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        mean_brightness = float(np.mean(gray))
        is_dark = mean_brightness < config.LOW_LIGHT_BRIGHTNESS_THRESHOLD
        logger.debug(
            "Brightness check: mean=%.1f  threshold=%d  low_light=%s",
            mean_brightness,
            config.LOW_LIGHT_BRIGHTNESS_THRESHOLD,
            is_dark,
        )
        return is_dark

    def enhance(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE to the L channel of *frame* and return an enhanced copy.

        Processing pipeline::

            RGB  →  LAB  →  CLAHE(L)  →  LAB  →  RGB

        The original *frame* is never modified.

        Args:
            frame: RGB ``(H, W, 3)`` ``ndarray``.

        Returns:
            Enhanced RGB ``(H, W, 3)`` ``ndarray`` (always a new array).
        """
        t0 = time.monotonic()

        # RGB → LAB (OpenCV uses uint8 LAB: L ∈ [0, 255], A/B ∈ [0, 255])
        lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)

        # Split channels, equalise L only, recombine.
        l_ch, a_ch, b_ch = cv2.split(lab)
        l_eq = self._clahe.apply(l_ch)
        lab_eq = cv2.merge((l_eq, a_ch, b_ch))

        # LAB → RGB
        enhanced = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)

        elapsed_ms = (time.monotonic() - t0) * 1_000
        logger.debug(
            "CLAHE enhance: %.2f ms  L_before_mean=%.1f  L_after_mean=%.1f",
            elapsed_ms,
            float(np.mean(l_ch)),
            float(np.mean(l_eq)),
        )

        return enhanced

    def auto_enhance(self, frame: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Enhance *frame* only when it is detected as low-light.

        Combines :meth:`is_low_light` and :meth:`enhance` into a single
        call, so callers do not need to check brightness themselves.

        Args:
            frame: RGB ``(H, W, 3)`` ``ndarray``.

        Returns:
            ``(output_frame, enhanced)`` where ``enhanced`` is ``True`` if
            CLAHE was applied, ``False`` if the frame was returned as-is.
            When ``enhanced`` is ``False`` the returned array is the same
            object as *frame* (no copy is made).
        """
        if self.is_low_light(frame):
            return self.enhance(frame), True
        return frame, False
