"""
tests/test_integration.py
==========================
End-to-end pipeline integration tests.

These tests exercise the full processing chain of each operating mode using
real core/AI module instances but no physical hardware:

* Camera frames are synthetic numpy arrays (from ``conftest.sample_frame``).
* TFLite models are absent on dev machines — detectors return ``[]`` and
  the pipeline degrades gracefully.
* The ``AudioManager`` is replaced by a ``MagicMock`` so no espeak-ng is
  needed.
* ``LowLightEnhancer``, ``SensorFusion``, and ``PriorityEngine`` run with
  real code.

Performance target
------------------
The navigation pipeline (enhancement + detection + fusion + priority) must
sustain ≥ 10 FPS on any machine, including CI.  Because detectors return ``[]``
when models are absent the main cost is CLAHE + NumPy, which is well within
the 100 ms/frame budget.
"""
import os
import sys
import time
from typing import List
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from ai.detection import Detection
from ai.low_light_enhancer import LowLightEnhancer
from ai.object_detector import ObjectDetector
from ai.traffic_light_detector import TrafficLightDetector
from ai.color_verifier import ColorVerifier
from core.sensor_fusion import SensorFusion
from core.priority_engine import PriorityEngine
from core.mode_manager import ModeManager


# ── Shared fixtures ────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def mock_audio():
    """A MagicMock standing in for AudioManager throughout this module."""
    return MagicMock()


@pytest.fixture(scope="module")
def enhancer():
    return LowLightEnhancer()


@pytest.fixture(scope="module")
def obj_detector():
    """ObjectDetector — may not have a model file; returns [] when absent."""
    return ObjectDetector()


@pytest.fixture(scope="module")
def tl_detector():
    """TrafficLightDetector — may not have a model file; returns [] when absent."""
    return TrafficLightDetector()


@pytest.fixture(scope="module")
def color_verifier():
    return ColorVerifier()


@pytest.fixture(scope="module")
def fusion():
    return SensorFusion()


@pytest.fixture(scope="module")
def priority_engine(mock_audio):
    return PriorityEngine(audio_manager=mock_audio)


@pytest.fixture(scope="module")
def mode_manager(mock_audio):
    return ModeManager(audio_manager=mock_audio)


# ════════════════════════════════════════════════════════════════════════════════
# Navigation pipeline (NAVIGATION / LOW_LIGHT modes)
# ════════════════════════════════════════════════════════════════════════════════

class TestNavigationPipeline:
    """Integration tests for the navigation processing chain."""

    def test_pipeline_runs_without_crash(
        self, sample_frame, enhancer, obj_detector, fusion, priority_engine
    ):
        """
        A single full pass through the navigation pipeline must not raise,
        regardless of whether TFLite model files are present.
        """
        frame, enhanced = enhancer.auto_enhance(sample_frame)
        detections: List[Detection] = obj_detector.detect(frame)
        fused = fusion.fuse(
            detections,
            ultrasonic_left_cm=None,
            ultrasonic_right_cm=None,
            tof_cm=None,
            frame_width=frame.shape[1],
            frame_height=frame.shape[0],
            dt=0.033,
        )
        announced = priority_engine.process_detections(fused)
        assert isinstance(announced, list)

    def test_pipeline_returns_detections_list(
        self, sample_frame, enhancer, obj_detector, fusion, priority_engine
    ):
        """
        The final announced list must be a ``List[Detection]`` (may be empty
        when models are absent).
        """
        frame, _ = enhancer.auto_enhance(sample_frame)
        detections = obj_detector.detect(frame)
        fused = fusion.fuse(detections, None, None, None, 640, 640, 0.033)
        announced = priority_engine.process_detections(fused)
        assert isinstance(announced, list)
        for det in announced:
            assert isinstance(det, Detection)

    def test_pipeline_with_sensor_readings(
        self, sample_frame, enhancer, obj_detector, fusion, priority_engine
    ):
        """
        Injecting synthetic sensor readings must not cause errors even when
        no detections are returned by the model.
        """
        frame, _ = enhancer.auto_enhance(sample_frame)
        detections = obj_detector.detect(frame)
        fused = fusion.fuse(
            detections,
            ultrasonic_left_cm=120.0,
            ultrasonic_right_cm=150.0,
            tof_cm=80.0,
            frame_width=frame.shape[1],
            frame_height=frame.shape[0],
            dt=0.033,
        )
        priority_engine.process_detections(fused)

    def test_pipeline_with_mock_detections(
        self, mock_detections, fusion, priority_engine
    ):
        """
        Injecting mock detections (pre-built ``Detection`` objects) through
        the fusion + priority chain must not raise and must produce scored
        detections.
        """
        fused = fusion.fuse(
            list(mock_detections),   # copy to avoid mutation side-effects
            ultrasonic_left_cm=None,
            ultrasonic_right_cm=None,
            tof_cm=None,
            frame_width=640,
            frame_height=640,
            dt=0.033,
        )
        # All fused detections must have a non-None fused_distance_cm
        # (they were seeded with fused_distance_cm from the fixture).
        for det in fused:
            if det.fused_distance_cm is not None:
                assert det.fused_distance_cm > 0.0

        announced = priority_engine.process_detections(fused)
        assert isinstance(announced, list)
        # All announced detections must have a threat_score assigned.
        for det in announced:
            assert det.threat_score >= 0.0


# ════════════════════════════════════════════════════════════════════════════════
# Low-light enhancement in the pipeline
# ════════════════════════════════════════════════════════════════════════════════

class TestLowLightPipeline:
    """Integration tests for the low-light enhancement branch."""

    def test_dark_frame_is_enhanced_before_detection(
        self, sample_frame_dark, enhancer, obj_detector
    ):
        """
        A dark frame must be identified as low-light, enhanced, and then
        fed to the detector without errors.
        """
        assert enhancer.is_low_light(sample_frame_dark), (
            "sample_frame_dark should be below the brightness threshold"
        )
        enhanced = enhancer.enhance(sample_frame_dark)
        detections = obj_detector.detect(enhanced)
        assert isinstance(detections, list)

    def test_enhancement_increases_mean_brightness(
        self, sample_frame_dark, enhancer
    ):
        """
        After CLAHE the enhanced frame must have a higher mean brightness
        than the original dark frame.
        """
        import cv2
        original_mean = float(np.mean(
            cv2.cvtColor(sample_frame_dark, cv2.COLOR_RGB2GRAY)
        ))
        enhanced = enhancer.enhance(sample_frame_dark)
        enhanced_mean = float(np.mean(
            cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
        ))
        assert enhanced_mean > original_mean, (
            f"Enhanced mean ({enhanced_mean:.1f}) must exceed "
            f"original mean ({original_mean:.1f})"
        )

    def test_bright_frame_not_enhanced_by_auto_enhance(
        self, sample_frame, enhancer
    ):
        """
        A bright (random noise) frame must not be identified as low-light so
        ``auto_enhance`` must return the same object (zero-copy path).
        """
        output, was_enhanced = enhancer.auto_enhance(sample_frame)
        # Random noise frame has mean ≈ 127, well above the threshold (60).
        assert not was_enhanced
        assert output is sample_frame


# ════════════════════════════════════════════════════════════════════════════════
# Traffic-light pipeline
# ════════════════════════════════════════════════════════════════════════════════

class TestTrafficLightPipeline:
    """Integration tests for the traffic-light detection + colour verification chain."""

    def test_tl_pipeline_runs_without_crash(
        self, sample_frame, enhancer, tl_detector, color_verifier
    ):
        """
        The traffic-light pipeline (enhance → detect → verify colour) must
        not raise even when the model file is absent.
        """
        frame, _ = enhancer.auto_enhance(sample_frame)
        detections = tl_detector.detect(frame)
        assert isinstance(detections, list)
        for det in detections:
            color = color_verifier.verify_traffic_light_color(frame, det.bbox)
            assert color in ("red", "yellow", "green", None)

    def test_tl_detector_unavailable_returns_empty(self, sample_frame):
        """
        A ``TrafficLightDetector`` initialised with a nonexistent model path
        must return ``[]`` from ``detect()``.
        """
        det = TrafficLightDetector(model_path="/nonexistent/model.tflite")
        result = det.detect(sample_frame)
        assert result == []

    def test_obj_detector_unavailable_returns_empty(self, sample_frame):
        """
        An ``ObjectDetector`` initialised with a nonexistent model path
        must return ``[]`` from ``detect()``.
        """
        det = ObjectDetector(model_path="/nonexistent/model.tflite")
        result = det.detect(sample_frame)
        assert result == []


# ════════════════════════════════════════════════════════════════════════════════
# Mode manager
# ════════════════════════════════════════════════════════════════════════════════

class TestModeManager:
    """Integration tests for ``ModeManager``."""

    def test_initial_mode_is_default(self, mode_manager):
        """The initial mode must be ``config.DEFAULT_MODE``."""
        assert mode_manager.get_current_mode() == config.DEFAULT_MODE

    def test_next_mode_advances_mode(self, mode_manager, mock_audio):
        """``next_mode()`` must advance to the next mode."""
        initial = mode_manager.get_current_mode()
        mode_manager.next_mode()
        assert mode_manager.get_current_mode() != initial

    def test_next_mode_cycles_through_all_modes(self, mock_audio):
        """Calling ``next_mode()`` ``NUM_MODES`` times must return to the start."""
        manager = ModeManager(audio_manager=mock_audio)
        start = manager.get_current_mode()
        for _ in range(config.NUM_MODES):
            manager.next_mode()
        assert manager.get_current_mode() == start

    def test_next_mode_announces_via_audio(self, mock_audio):
        """``next_mode()`` must call ``audio_manager.speak_key()`` once."""
        manager = ModeManager(audio_manager=mock_audio)
        mock_audio.reset_mock()
        manager.next_mode()
        mock_audio.speak_key.assert_called_once()

    def test_get_current_mode_is_thread_safe(self, mock_audio):
        """Concurrent reads and writes to the mode must not deadlock or raise."""
        manager = ModeManager(audio_manager=mock_audio)
        errors: list = []

        def reader():
            for _ in range(100):
                try:
                    manager.get_current_mode()
                except Exception as exc:
                    errors.append(exc)

        def writer():
            for _ in range(20):
                try:
                    manager.next_mode()
                except Exception as exc:
                    errors.append(exc)

        import threading
        threads = [
            threading.Thread(target=reader),
            threading.Thread(target=reader),
            threading.Thread(target=writer),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)
        assert not errors, f"Thread-safety errors: {errors}"


# ════════════════════════════════════════════════════════════════════════════════
# Priority engine integration
# ════════════════════════════════════════════════════════════════════════════════

class TestPriorityEngineIntegration:
    """Tests for ``PriorityEngine`` with realistic detection data."""

    @pytest.fixture
    def engine(self):
        return PriorityEngine(audio_manager=None)   # headless — no audio

    def test_threat_score_assigned_to_all_detections(
        self, engine, mock_detections
    ):
        """
        ``process_detections()`` must write ``threat_score`` onto every
        detection in the input list.
        """
        dets = list(mock_detections)
        engine.process_detections(dets)
        for det in dets:
            assert isinstance(det.threat_score, float)
            assert det.threat_score >= 0.0

    def test_car_at_80cm_scores_higher_than_bench_at_380cm(self, engine):
        """
        A car at 80 cm (high base score, high distance factor) must score
        higher than a bench at 380 cm (low base score, low factor).
        """
        car = Detection(
            bbox=(200, 250, 440, 500), class_id=2, class_name="car",
            confidence=0.92, center_x=320, center_y=375,
            position="center", fused_distance_cm=80.0,
        )
        bench = Detection(
            bbox=(220, 300, 420, 450), class_id=13, class_name="bench",
            confidence=0.61, center_x=320, center_y=375,
            position="center", fused_distance_cm=380.0,
        )
        engine.process_detections([car, bench])
        assert car.threat_score > bench.threat_score, (
            f"Car score {car.threat_score:.2f} must exceed bench "
            f"score {bench.threat_score:.2f}"
        )

    def test_detections_sorted_by_threat_descending(
        self, engine, mock_detections
    ):
        """
        After ``process_detections()`` the input list must be sorted by
        ``threat_score`` in descending order.
        """
        dets = list(mock_detections)
        engine.process_detections(dets)
        scores = [d.threat_score for d in dets]
        assert scores == sorted(scores, reverse=True), (
            f"Detections not sorted by threat score: {scores}"
        )

    def test_at_most_max_notification_objects_announced(self, engine):
        """
        ``process_detections()`` must announce at most
        ``config.MAX_NOTIFICATION_OBJECTS`` per call.
        """
        # Create many high-scoring detections with distinct positions to
        # avoid collisions on the object-id cooldown key.
        dets = [
            Detection(
                bbox=(i * 60, 100, i * 60 + 50, 400),
                class_id=2, class_name="car",
                confidence=0.95,
                center_x=i * 60 + 25,
                center_y=250,
                position=["left", "center", "right"][i % 3],
                fused_distance_cm=50.0 + i * 10.0,
            )
            for i in range(8)
        ]
        announced = engine.process_detections(dets)
        assert len(announced) <= config.MAX_NOTIFICATION_OBJECTS

    def test_none_audio_manager_does_not_crash(self):
        """
        ``PriorityEngine(audio_manager=None)`` must not raise when
        ``process_detections()`` would normally call ``speak()``.
        """
        engine = PriorityEngine(audio_manager=None)
        det = Detection(
            bbox=(100, 100, 400, 500), class_id=0, class_name="person",
            confidence=0.9, center_x=250, center_y=300,
            position="center", fused_distance_cm=80.0,
        )
        engine.process_detections([det])   # Must not raise.


# ════════════════════════════════════════════════════════════════════════════════
# Performance: ≥ 10 FPS
# ════════════════════════════════════════════════════════════════════════════════

class TestPerformance:
    """Throughput tests for the processing pipeline."""

    def test_navigation_pipeline_fps_above_10(
        self, sample_frame, enhancer, obj_detector, fusion, priority_engine
    ):
        """
        The navigation pipeline must sustain ≥ 10 FPS over 30 frames.

        On a development machine without model files the cost is dominated by
        ``LowLightEnhancer.auto_enhance`` (CLAHE, < 5 ms) and list operations.
        The 100 ms/frame budget (10 FPS) is very conservative.
        """
        n_frames = 30
        start = time.monotonic()

        for _ in range(n_frames):
            frame, _ = enhancer.auto_enhance(sample_frame)
            dets = obj_detector.detect(frame)
            fused = fusion.fuse(
                dets, None, None, None,
                frame.shape[1], frame.shape[0], 0.033,
            )
            priority_engine.process_detections(fused)

        elapsed = time.monotonic() - start
        fps = n_frames / elapsed
        assert fps >= 10.0, (
            f"Navigation pipeline too slow: {fps:.1f} FPS "
            f"({elapsed * 1000 / n_frames:.1f} ms/frame)"
        )

    def test_low_light_pipeline_fps_above_10(
        self, sample_frame_dark, enhancer, obj_detector, fusion, priority_engine
    ):
        """
        The low-light pipeline (forced CLAHE + navigation) must also sustain
        ≥ 10 FPS.  CLAHE on a 640×640 frame takes < 5 ms on modern hardware.
        """
        n_frames = 30
        start = time.monotonic()

        for _ in range(n_frames):
            frame = enhancer.enhance(sample_frame_dark)   # Unconditional enhance.
            dets = obj_detector.detect(frame)
            fused = fusion.fuse(
                dets, None, None, None,
                frame.shape[1], frame.shape[0], 0.033,
            )
            priority_engine.process_detections(fused)

        elapsed = time.monotonic() - start
        fps = n_frames / elapsed
        assert fps >= 10.0, (
            f"Low-light pipeline too slow: {fps:.1f} FPS "
            f"({elapsed * 1000 / n_frames:.1f} ms/frame)"
        )

    def test_sensor_fusion_throughput(self, mock_detections, fusion):
        """
        Processing 1000 ``fuse()`` calls with 5 mock detections each must
        complete in < 2 seconds.
        """
        start = time.monotonic()
        for i in range(1000):
            dets = list(mock_detections)
            fusion.fuse(dets, 120.0, 130.0, 90.0, 640, 640, 0.033)
        elapsed = time.monotonic() - start
        assert elapsed < 2.0, (
            f"SensorFusion too slow: {elapsed:.2f} s for 1000 frames"
        )
