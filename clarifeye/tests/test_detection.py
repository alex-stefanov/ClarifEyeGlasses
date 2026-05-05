"""
tests/test_detection.py
=======================
Tests for the AI detection modules:

* ``ai/traffic_light_detector.py`` — TrafficLightDetector, _apply_nms, _compute_iou
* ``ai/object_detector.py``        — ObjectDetector, estimate_distance_cm
* ``ai/color_verifier.py``         — ColorVerifier

TFLite model tests gracefully handle missing model files (the detectors
initialise to an inert state and return ``[]`` when models are absent).
Color-verifier tests use synthetic solid-colour frames and require only
NumPy and OpenCV — no model file needed.
"""
import os
import sys
from typing import List

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from ai.detection import Detection
from ai.traffic_light_detector import TrafficLightDetector, _apply_nms, _compute_iou
from ai.object_detector import ObjectDetector, estimate_distance_cm
from ai.color_verifier import ColorVerifier


# ── Helpers ───────────────────────────────────────────────────────────────────

def _solid_rgb_frame(r: int, g: int, b: int, size: int = 100) -> np.ndarray:
    """
    Return a ``size×size×3`` uint8 ndarray filled with the given RGB colour.

    Used to create synthetic test frames where the dominant colour is known
    in advance so the ColorVerifier's HSV pixel-counting can be validated.
    """
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    frame[:, :, 0] = r
    frame[:, :, 1] = g
    frame[:, :, 2] = b
    return frame


def _make_detection(
    bbox=(10, 10, 100, 200),
    class_id=0,
    class_name="person",
    confidence=0.8,
) -> Detection:
    x1, y1, x2, y2 = bbox
    return Detection(
        bbox=bbox,
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
        center_x=(x1 + x2) // 2,
        center_y=(y1 + y2) // 2,
    )


# ════════════════════════════════════════════════════════════════════════════════
# TrafficLightDetector
# ════════════════════════════════════════════════════════════════════════════════

class TestTrafficLightDetector:
    """Tests for ``TrafficLightDetector`` and its NMS helpers."""

    def test_initialises_without_crash_when_model_missing(self):
        """
        TrafficLightDetector must not raise when the model file is absent.
        The ``_available`` flag should be ``False`` so ``detect()`` returns ``[]``.
        """
        detector = TrafficLightDetector(model_path="/nonexistent/model.tflite")
        assert detector is not None
        assert detector._available is False

    def test_detect_returns_empty_list_when_model_missing(self, sample_frame):
        """
        ``detect()`` must return an empty list (not raise) when the model was
        not loaded.  Verifies the graceful no-op path.
        """
        detector = TrafficLightDetector(model_path="/nonexistent/model.tflite")
        result = detector.detect(sample_frame)
        assert isinstance(result, list)
        assert len(result) == 0

    def test_detect_returns_list_of_detections_type(self, sample_frame):
        """
        When a real model is available ``detect()`` must return a
        ``List[Detection]``.  This test exercises the type contract and skips
        if the model file is absent (development machine without trained model).
        """
        if not os.path.exists(config.TRAFFIC_LIGHT_MODEL_PATH):
            pytest.skip("Traffic-light model not found — skipping live inference test.")
        detector = TrafficLightDetector()
        result = detector.detect(sample_frame)
        assert isinstance(result, list)
        for det in result:
            assert isinstance(det, Detection)

    # ── NMS helpers ───────────────────────────────────────────────────────────

    def test_compute_iou_identical_boxes(self):
        """IoU of two identical boxes must be exactly 1.0."""
        box = (0, 0, 100, 100)
        assert _compute_iou(box, box) == pytest.approx(1.0)

    def test_compute_iou_disjoint_boxes(self):
        """IoU of non-overlapping boxes must be 0.0."""
        assert _compute_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0

    def test_compute_iou_partial_overlap(self):
        """IoU of 50 % overlapping boxes must be between 0 and 1."""
        iou = _compute_iou((0, 0, 10, 10), (5, 0, 15, 10))
        assert 0.0 < iou < 1.0

    def test_compute_iou_symmetry(self):
        """IoU(A, B) must equal IoU(B, A)."""
        a, b = (10, 10, 60, 60), (30, 30, 80, 80)
        assert _compute_iou(a, b) == pytest.approx(_compute_iou(b, a))

    def test_nms_removes_overlapping_lower_confidence_box(self):
        """
        NMS must suppress the lower-confidence box when two boxes have
        IoU above the threshold, keeping only the highest-scoring one.
        """
        # Two nearly identical red-light boxes (IoU ≈ 0.73).
        high = _make_detection(bbox=(10, 10, 110, 110), confidence=0.90,
                               class_name="red")
        low  = _make_detection(bbox=(15, 15, 115, 115), confidence=0.60,
                               class_name="red")
        result = _apply_nms([high, low], iou_threshold=0.45)
        assert len(result) == 1
        assert result[0].confidence == pytest.approx(0.90)

    def test_nms_keeps_non_overlapping_boxes(self):
        """
        NMS must retain all boxes that do not overlap above the threshold,
        even if they have different confidence scores.
        """
        red   = _make_detection(bbox=(0, 0, 50, 50),     confidence=0.90, class_name="red")
        green = _make_detection(bbox=(300, 300, 400, 400), confidence=0.70, class_name="green")
        result = _apply_nms([red, green], iou_threshold=0.45)
        assert len(result) == 2

    def test_nms_empty_input_returns_empty(self):
        """NMS on an empty list must return an empty list without raising."""
        assert _apply_nms([], iou_threshold=0.45) == []

    def test_nms_result_sorted_by_confidence_descending(self):
        """NMS output must be sorted by descending confidence."""
        dets = [
            _make_detection(bbox=(i*100, 0, i*100+50, 50), confidence=0.5 + i * 0.1)
            for i in range(4)
        ]
        result = _apply_nms(dets, iou_threshold=0.45)
        confidences = [d.confidence for d in result]
        assert confidences == sorted(confidences, reverse=True)


# ════════════════════════════════════════════════════════════════════════════════
# ObjectDetector
# ════════════════════════════════════════════════════════════════════════════════

class TestObjectDetector:
    """Tests for ``ObjectDetector`` and ``estimate_distance_cm``."""

    def test_initialises_without_crash_when_model_missing(self):
        """ObjectDetector must not raise when the TFLite model file is absent."""
        detector = ObjectDetector(model_path="/nonexistent/model.tflite")
        assert detector is not None
        assert detector._available is False

    def test_detect_returns_empty_when_unavailable(self, sample_frame):
        """``detect()`` returns ``[]`` (not raises) when model is unavailable."""
        detector = ObjectDetector(model_path="/nonexistent/model.tflite")
        result = detector.detect(sample_frame)
        assert isinstance(result, list) and len(result) == 0

    def test_detect_live_returns_only_classes_of_interest(self, sample_frame):
        """
        Live inference must filter detections to ``config.OBJECT_CLASSES_OF_INTEREST``.
        Any returned class_id must be a key in that dict.
        """
        if not os.path.exists(config.OBJECT_MODEL_PATH):
            pytest.skip("Object detection model not found.")
        detector = ObjectDetector()
        result = detector.detect(sample_frame)
        for det in result:
            assert det.class_id in config.OBJECT_CLASSES_OF_INTEREST, (
                f"class_id {det.class_id} not in OBJECT_CLASSES_OF_INTEREST"
            )

    # ── estimate_distance_cm ──────────────────────────────────────────────────

    def test_estimate_distance_known_class_returns_float(self):
        """
        ``estimate_distance_cm`` must return a ``float`` for classes that have
        a known real-world height in ``config.KNOWN_HEIGHTS``.
        """
        det = _make_detection(
            bbox=(100, 160, 300, 480),   # height = 320 px
            class_name="person",
        )
        dist = estimate_distance_cm(det, frame_height=640)
        assert isinstance(dist, float), f"Expected float, got {type(dist)}"

    def test_estimate_distance_person_in_reasonable_range(self):
        """
        For a person bbox spanning 320 px in a 640-px frame the pinhole formula
        gives ≈ 265 cm.  The result must be in the physically plausible range.

        Formula: (1.7 m × 500 px focal) / 320 px = 2.656 m = 265.6 cm
        """
        det = _make_detection(
            bbox=(100, 160, 300, 480),   # height = 320 px
            class_name="person",
        )
        dist = estimate_distance_cm(det, frame_height=640)
        assert 50.0 <= dist <= 2000.0, f"Distance {dist} cm outside plausible range"

    def test_estimate_distance_car_large_bbox_is_close(self):
        """
        A very large car bounding box (nearly full frame) should yield a short
        estimated distance (object is nearby).
        """
        det = _make_detection(
            bbox=(0, 100, 640, 590),   # height = 490 px (nearly full frame)
            class_name="car",
        )
        dist = estimate_distance_cm(det, frame_height=640)
        # (1.5 m × 500 px) / 490 px × 100 = 153 cm
        assert dist is not None and dist < 300.0

    def test_estimate_distance_unknown_class_returns_none(self):
        """
        ``estimate_distance_cm`` must return ``None`` for classes not in
        ``config.KNOWN_HEIGHTS`` (e.g. ``"bench"``).
        """
        det = _make_detection(class_name="bench")
        dist = estimate_distance_cm(det, frame_height=640)
        assert dist is None

    def test_estimate_distance_zero_height_bbox_returns_none(self):
        """
        A bbox with zero height (degenerate detection) must yield ``None``
        rather than a division-by-zero error.
        """
        det = _make_detection(
            bbox=(10, 100, 200, 100),   # y1 == y2 → height = 0
            class_name="person",
        )
        dist = estimate_distance_cm(det, frame_height=640)
        assert dist is None


# ════════════════════════════════════════════════════════════════════════════════
# ColorVerifier
# ════════════════════════════════════════════════════════════════════════════════

class TestColorVerifier:
    """
    Tests for ``ColorVerifier`` using synthetic solid-colour frames.

    All tests are CPU-only (cv2 HSV conversion) and require no model files.
    The full-frame bbox ``(0, 0, size, size)`` ensures all pixels are inside
    the region of interest, guaranteeing the ratio threshold is met.
    """

    @pytest.fixture(scope="class")
    def verifier(self):
        return ColorVerifier()

    def _full_bbox(self, size: int = 100):
        return (0, 0, size, size)

    def test_red_frame_identified_as_red(self, verifier):
        """
        A solid red RGB frame (255, 0, 0) must be identified as ``"red"``.
        Red in OpenCV HSV lies at H≈0 with high saturation, within the
        two red hue arcs defined in ``config``.
        """
        frame = _solid_rgb_frame(255, 0, 0)
        result = verifier.verify_traffic_light_color(frame, self._full_bbox())
        assert result == "red", f"Expected 'red', got {result!r}"

    def test_yellow_frame_identified_as_yellow(self, verifier):
        """
        A solid yellow RGB frame (255, 220, 0) must be identified as
        ``"yellow"``.  Yellow in OpenCV HSV lies at H≈30 within the range
        [15, 35] configured in ``config``.
        """
        frame = _solid_rgb_frame(255, 220, 0)
        result = verifier.verify_traffic_light_color(frame, self._full_bbox())
        assert result == "yellow", f"Expected 'yellow', got {result!r}"

    def test_green_frame_identified_as_green(self, verifier):
        """
        A solid green RGB frame (0, 255, 0) must be identified as ``"green"``.
        Pure green in OpenCV HSV lies at H=60 within the range [35, 85].
        """
        frame = _solid_rgb_frame(0, 255, 0)
        result = verifier.verify_traffic_light_color(frame, self._full_bbox())
        assert result == "green", f"Expected 'green', got {result!r}"

    def test_grey_frame_returns_none(self, verifier):
        """
        A neutral grey frame has near-zero saturation in HSV and must not
        exceed any colour's pixel-ratio threshold.  ``verify_traffic_light_color``
        must return ``None`` for ambiguous / achromatic regions.
        """
        frame = _solid_rgb_frame(128, 128, 128)
        result = verifier.verify_traffic_light_color(frame, self._full_bbox())
        assert result is None, f"Expected None for grey, got {result!r}"

    def test_dark_frame_returns_none(self, verifier):
        """
        A very dark frame (near-black) has low V in HSV and must not trigger
        any colour detection (all HSV ranges require V ≥ 50).
        """
        frame = _solid_rgb_frame(20, 20, 20)
        result = verifier.verify_traffic_light_color(frame, self._full_bbox())
        assert result is None

    def test_partial_bbox_respected(self, verifier):
        """
        The verifier must analyse only the pixels inside the given ``bbox``,
        not the full frame.  Placing a red region in a small top-left corner
        and querying only the grey remainder must return ``None``.
        """
        frame = _solid_rgb_frame(128, 128, 128, size=100)
        frame[0:10, 0:10] = [255, 0, 0]   # small red patch — not dominant in grey region
        # Query the grey region (excluding the red patch).
        result = verifier.verify_traffic_light_color(frame, (10, 10, 100, 100))
        assert result is None

    def test_dominant_color_returns_name_and_ratio(self, verifier):
        """
        ``get_dominant_color`` must return a 2-tuple ``(color_name, ratio)``
        where ratio is a float in ``[0.0, 1.0]`` and color_name is one of
        ``"red"``, ``"yellow"``, ``"green"``, or ``"none"``.
        """
        frame = _solid_rgb_frame(0, 200, 0)
        color, ratio = verifier.get_dominant_color(frame, self._full_bbox())
        assert color in ("red", "yellow", "green", "none")
        assert 0.0 <= ratio <= 1.0

    def test_solid_green_ratio_above_threshold(self, verifier):
        """
        For a fully green frame the matching-pixel ratio must exceed the
        ``config.COLOR_PIXEL_RATIO_THRESHOLD`` (0.15).
        """
        frame = _solid_rgb_frame(0, 255, 0)
        _, ratio = verifier.get_dominant_color(frame, self._full_bbox())
        assert ratio >= config.COLOR_PIXEL_RATIO_THRESHOLD
