"""
ClarifEye Currency Recognizer

ORB feature-matching against pre-downloaded reference banknote images.
Supports Bulgarian leva (BGN) and Euros (EUR).

Reference image layout expected under references_dir:
  bgn/<denom>_front.jpg  bgn/<denom>_back.jpg
  eur/<denom>_front.jpg  eur/<denom>_back.jpg
"""
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2  # type: ignore[import-untyped]
    _OPENCV_AVAILABLE = True
except ImportError:
    _OPENCV_AVAILABLE = False

try:
    from .. import config
except ImportError:
    import config  # type: ignore[no-redef]

logger = logging.getLogger("clarifeye.ai.currency_recognizer")

_AUDIO_KEY_MAP: Dict[Tuple[str, int], str] = {
    ("bgn", 5):   "currency_5_lev",
    ("bgn", 10):  "currency_10_lev",
    ("bgn", 20):  "currency_20_lev",
    ("bgn", 50):  "currency_50_lev",
    ("bgn", 100): "currency_100_lev",
    ("eur", 5):   "currency_5_eur",
    ("eur", 10):  "currency_10_eur",
    ("eur", 20):  "currency_20_eur",
    ("eur", 50):  "currency_50_eur",
    ("eur", 100): "currency_100_eur",
    ("eur", 200): "currency_200_eur",
}


class CurrencyRecognizer:
    """
    Recognizes banknotes by matching ORB descriptors against reference images.

    Loads all reference images at construction time (~80 MB RAM for ~22 images).
    Returns None from recognize() rather than raising when references are absent.
    """

    def __init__(self, references_dir: str) -> None:
        if not _OPENCV_AVAILABLE:
            logger.warning("cv2 not available — CurrencyRecognizer disabled.")
            self._orb = None
            self._matcher = None
            self._references = {}
            return
        self._orb = cv2.ORB_create(nfeatures=1000)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # (currency_code, denomination) → list of (keypoints, descriptors) per side
        self._references: Dict[Tuple[str, int], List[Tuple]] = {}
        self._load_references(references_dir)

    def _load_references(self, references_dir: str) -> None:
        if not _OPENCV_AVAILABLE:
            return
        if not os.path.isdir(references_dir):
            logger.warning(
                "Banknote references directory not found: %s — "
                "currency recognition will return no matches until images are downloaded.",
                references_dir,
            )
            return

        loaded = 0
        for currency_code in ("bgn", "eur"):
            currency_dir = os.path.join(references_dir, currency_code)
            if not os.path.isdir(currency_dir):
                continue
            for fname in sorted(os.listdir(currency_dir)):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                stem = os.path.splitext(fname)[0]
                denom_str = stem.split("_", 1)[0]
                try:
                    denom = int(denom_str)
                except ValueError:
                    logger.debug("Skipping unrecognised reference filename: %s", fname)
                    continue

                img_path = os.path.join(currency_dir, fname)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    logger.warning("Could not read reference image: %s", img_path)
                    continue

                img = _resize_long_edge(img, config.CURRENCY_RESIZE_WIDTH)
                kp, des = self._orb.detectAndCompute(img, None)
                if des is None or len(des) == 0:
                    logger.warning("No ORB descriptors extracted from: %s", img_path)
                    continue

                key = (currency_code, denom)
                self._references.setdefault(key, []).append((kp, des))
                loaded += 1

        logger.info(
            "CurrencyRecognizer: loaded %d reference image(s) for %d denomination(s).",
            loaded,
            len(self._references),
        )

    def recognize(
        self, frame: np.ndarray
    ) -> Optional[Tuple[str, int, float]]:
        """
        Match *frame* against all reference banknotes.

        Returns:
            (currency_code, denomination, confidence) or None if no match
            clears config.CURRENCY_MIN_MATCHES good-match threshold.
        """
        if not self._references or not _OPENCV_AVAILABLE:
            return None

        gray = (
            cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.ndim == 3 else frame
        )
        gray = _resize_long_edge(gray, config.CURRENCY_RESIZE_WIDTH)

        kp_query, des_query = self._orb.detectAndCompute(gray, None)
        if des_query is None or len(des_query) == 0:
            return None

        best_key: Optional[Tuple[str, int]] = None
        best_count: int = 0

        for (currency_code, denom), ref_list in self._references.items():
            for _ref_kp, ref_des in ref_list:
                try:
                    matches = self._matcher.match(des_query, ref_des)
                    good = [m for m in matches if m.distance < 50]
                    count = len(good)
                    if count > best_count:
                        best_count = count
                        best_key = (currency_code, denom)
                except Exception as exc:
                    logger.debug(
                        "ORB matching error for %s %s: %s", currency_code, denom, exc
                    )

        if best_count < config.CURRENCY_MIN_MATCHES:
            return None

        confidence = best_count / max(len(des_query), 1)
        currency_code, denom = best_key  # type: ignore[misc]
        return (currency_code, denom, confidence)

    @staticmethod
    def get_audio_key(currency_code: str, denomination: int) -> str:
        """
        Map a (currency_code, denomination) pair to its AUDIO_KEYS key.

        Raises:
            ValueError: if the combination is not in the registry.
        """
        key = (currency_code, denomination)
        if key not in _AUDIO_KEY_MAP:
            raise ValueError(
                f"Unknown currency combination: {currency_code!r} {denomination}"
            )
        return _AUDIO_KEY_MAP[key]


def _resize_long_edge(img: np.ndarray, target: int) -> np.ndarray:
    h, w = img.shape[:2]
    long_edge = max(h, w)
    if long_edge <= target:
        return img
    scale = target / long_edge
    return cv2.resize(
        img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
    )
