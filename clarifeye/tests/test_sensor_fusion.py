"""
tests/test_sensor_fusion.py
============================
Tests for ``core/sensor_fusion.py``:

* ``SingleObjectKalman`` — per-object 1-D Kalman filter (distance + velocity).
* ``SensorFusion``       — multi-source fusion orchestrator.

No hardware required.  Sensor readings are injected as plain floats.
Stale-eviction tests manipulate ``_last_update_times`` directly because
we cannot sleep for ``_STALE_TIMEOUT_SEC`` (2 s) in a unit test.
"""
import os
import sys
import threading

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from ai.detection import Detection
from core.sensor_fusion import SingleObjectKalman, SensorFusion, _STALE_TIMEOUT_SEC


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_detection(
    class_name: str = "person",
    class_id: int = 0,
    center_x: int = 320,
    center_y: int = 320,
    bbox=(100, 160, 300, 480),
    confidence: float = 0.85,
) -> Detection:
    """Create a minimal Detection for sensor-fusion testing."""
    return Detection(
        bbox=bbox,
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
        center_x=center_x,
        center_y=center_y,
    )


# ════════════════════════════════════════════════════════════════════════════════
# SingleObjectKalman
# ════════════════════════════════════════════════════════════════════════════════

class TestSingleObjectKalman:
    """Tests for the per-object Kalman filter."""

    def test_initial_state_matches_seed(self):
        """Initial distance must equal the seed; initial velocity must be 0."""
        kalman = SingleObjectKalman(initial_distance=150.0)
        dist, vel = kalman.get_state()
        assert dist == pytest.approx(150.0)
        assert vel == pytest.approx(0.0)

    def test_predict_does_not_change_distance_with_zero_velocity(self):
        """
        With zero initial velocity a single predict step leaves the distance
        estimate approximately unchanged (constant-velocity model, v=0).
        """
        kalman = SingleObjectKalman(initial_distance=200.0)
        kalman.predict(dt=0.033)
        dist, _ = kalman.get_state()
        assert dist == pytest.approx(200.0, abs=1.0)

    def test_update_converges_toward_measurement(self):
        """
        Repeated predict+update cycles with a fixed measurement must drive
        the state estimate close to that measurement.
        """
        kalman = SingleObjectKalman(initial_distance=300.0)
        noise = config.KALMAN.measurement_noise_camera
        for _ in range(30):
            kalman.predict(dt=0.033)
            kalman.update(100.0, noise)
        dist, _ = kalman.get_state()
        assert dist == pytest.approx(100.0, abs=15.0), (
            f"Filter did not converge: {dist:.1f} cm (expected ≈ 100 cm)"
        )

    def test_velocity_negative_for_approaching_object(self):
        """
        Successive measurements with decreasing distance must produce a
        negative velocity estimate (object is approaching).
        """
        kalman = SingleObjectKalman(initial_distance=300.0)
        noise = config.KALMAN.measurement_noise_camera
        for i in range(20):
            kalman.predict(dt=0.033)
            kalman.update(300.0 - i * 10.0, noise)
        _, vel = kalman.get_state()
        assert vel < 0.0, f"Expected negative velocity for approaching object, got {vel:.2f}"

    def test_velocity_positive_for_receding_object(self):
        """
        Successive measurements with increasing distance must produce a
        positive velocity estimate (object is receding).
        """
        kalman = SingleObjectKalman(initial_distance=50.0)
        noise = config.KALMAN.measurement_noise_camera
        for i in range(20):
            kalman.predict(dt=0.033)
            kalman.update(50.0 + i * 10.0, noise)
        _, vel = kalman.get_state()
        assert vel > 0.0, f"Expected positive velocity for receding object, got {vel:.2f}"

    def test_distance_never_goes_negative(self):
        """
        The physical constraint ``distance ≥ 0`` must hold even when extreme
        negative measurements push the estimate below zero.
        """
        kalman = SingleObjectKalman(initial_distance=5.0)
        for _ in range(20):
            kalman.predict(dt=0.1)
            kalman.update(-50.0, config.KALMAN.measurement_noise_tof)
        dist, _ = kalman.get_state()
        assert dist >= 0.0, f"Distance must not go negative, got {dist:.3f}"

    def test_predict_only_stays_finite(self):
        """
        Calling ``predict`` 100 times without any ``update`` must yield a
        finite state (no divergence or NaN).
        """
        kalman = SingleObjectKalman(initial_distance=100.0)
        for _ in range(100):
            kalman.predict(dt=0.033)
        dist, vel = kalman.get_state()
        assert np.isfinite(dist)
        assert np.isfinite(vel)

    def test_update_ignores_negative_measurement(self):
        """``update`` with a negative value must be a no-op (state unchanged)."""
        kalman = SingleObjectKalman(initial_distance=100.0)
        kalman.update(-10.0, config.KALMAN.measurement_noise_camera)
        dist, _ = kalman.get_state()
        assert dist == pytest.approx(100.0)

    def test_update_ignores_nan_measurement(self):
        """``update`` with NaN must be silently discarded."""
        kalman = SingleObjectKalman(initial_distance=100.0)
        kalman.update(float("nan"), config.KALMAN.measurement_noise_camera)
        dist, _ = kalman.get_state()
        assert dist == pytest.approx(100.0)

    def test_update_ignores_inf_measurement(self):
        """``update`` with +inf must be silently discarded."""
        kalman = SingleObjectKalman(initial_distance=100.0)
        kalman.update(float("inf"), config.KALMAN.measurement_noise_camera)
        dist, _ = kalman.get_state()
        assert dist == pytest.approx(100.0)

    def test_zero_dt_clamped_safely(self):
        """
        ``predict(dt=0)`` must not cause division-by-zero or non-finite state.
        The implementation clamps dt to 1 µs internally.
        """
        kalman = SingleObjectKalman(initial_distance=100.0)
        kalman.predict(dt=0.0)
        dist, vel = kalman.get_state()
        assert np.isfinite(dist)
        assert np.isfinite(vel)

    def test_negative_dt_clamped_safely(self):
        """Negative dt must be clamped to the minimum, not cause errors."""
        kalman = SingleObjectKalman(initial_distance=100.0)
        kalman.predict(dt=-1.0)
        dist, vel = kalman.get_state()
        assert np.isfinite(dist)
        assert np.isfinite(vel)

    def test_tof_noise_lower_means_faster_convergence(self):
        """
        With lower measurement noise the ToF-driven filter must converge faster
        than the camera-driven filter starting from the same seed.
        """
        truth = 80.0
        k_tof = SingleObjectKalman(initial_distance=200.0)
        k_cam = SingleObjectKalman(initial_distance=200.0)
        for _ in range(10):
            k_tof.predict(dt=0.033)
            k_tof.update(truth, config.KALMAN.measurement_noise_tof)
            k_cam.predict(dt=0.033)
            k_cam.update(truth, config.KALMAN.measurement_noise_camera)
        dist_tof, _ = k_tof.get_state()
        dist_cam, _ = k_cam.get_state()
        assert abs(dist_tof - truth) <= abs(dist_cam - truth), (
            f"ToF filter ({dist_tof:.1f} cm) should be closer to truth ({truth} cm) "
            f"than camera filter ({dist_cam:.1f} cm)"
        )

    def test_get_state_returns_floats(self):
        """``get_state()`` must return a 2-tuple of Python floats."""
        kalman = SingleObjectKalman(initial_distance=50.0)
        dist, vel = kalman.get_state()
        assert isinstance(dist, float)
        assert isinstance(vel, float)


# ════════════════════════════════════════════════════════════════════════════════
# SensorFusion
# ════════════════════════════════════════════════════════════════════════════════

class TestSensorFusion:
    """Tests for the SensorFusion orchestrator."""

    @pytest.fixture
    def fusion(self) -> SensorFusion:
        return SensorFusion()

    # ── Position assignment ───────────────────────────────────────────────────

    def test_position_left(self, fusion):
        """center_x in the left third of the frame → ``position='left'``."""
        det = _make_detection(center_x=100, bbox=(10, 100, 190, 420))
        fusion.fuse([det], None, None, None, 640, 480, 0.033)
        assert det.position == "left"

    def test_position_center(self, fusion):
        """center_x in the middle third → ``position='center'``."""
        det = _make_detection(center_x=320, bbox=(200, 160, 440, 480))
        fusion.fuse([det], None, None, None, 640, 480, 0.033)
        assert det.position == "center"

    def test_position_right(self, fusion):
        """center_x in the right third → ``position='right'``."""
        det = _make_detection(center_x=540, bbox=(465, 100, 615, 420))
        fusion.fuse([det], None, None, None, 640, 480, 0.033)
        assert det.position == "right"

    def test_position_boundary_left_center(self, fusion):
        """
        center_x exactly at the left/centre boundary must be assigned
        ``"center"`` (boundary belongs to the upper zone).
        frame_third = 640 // 3 = 213.  center_x = 213 → center.
        """
        det = _make_detection(center_x=213, bbox=(150, 100, 276, 420))
        fusion.fuse([det], None, None, None, 640, 480, 0.033)
        assert det.position == "center"

    # ── Sensor selection ──────────────────────────────────────────────────────

    def test_tof_selected_for_center_close_range(self, fusion):
        """
        When the ToF reading is below ``tof_priority_range_cm`` and the
        detection is in the centre zone, ``sensor_distance_cm`` must equal
        the ToF value.
        """
        tof_cm = config.KALMAN.tof_priority_range_cm - 20.0  # e.g. 180 cm
        det = _make_detection(center_x=320, bbox=(200, 160, 440, 480))
        fusion.fuse([det], None, None, tof_cm, 640, 480, 0.033)
        assert det.sensor_distance_cm == pytest.approx(tof_cm)

    def test_tof_not_selected_above_priority_range(self, fusion):
        """
        A ToF reading above ``tof_priority_range_cm`` must not be selected;
        the fusion must fall back to the camera estimate.
        """
        tof_above = config.KALMAN.tof_priority_range_cm + 50.0  # e.g. 250 cm
        det = _make_detection(center_x=320, bbox=(200, 160, 440, 480))
        fusion.fuse([det], None, None, tof_above, 640, 480, 0.033)
        # sensor_distance_cm must differ from tof_above (camera estimate used).
        assert det.sensor_distance_cm != pytest.approx(tof_above)

    def test_tof_not_selected_for_left_zone(self, fusion):
        """
        ToF must only be a priority for centre-zone detections, not left/right.
        Even a close ToF reading must not override the left ultrasonic rule.
        """
        tof_cm = 50.0   # well within ToF range
        us_left = 80.0  # left ultrasonic
        det = _make_detection(center_x=80, bbox=(10, 100, 150, 420))
        fusion.fuse([det], us_left, None, tof_cm, 640, 480, 0.033)
        # For LEFT zone the ultrasonic should be selected, not ToF.
        assert det.sensor_distance_cm == pytest.approx(us_left)

    def test_ultrasonic_selected_for_left_zone(self, fusion):
        """
        Left ultrasonic within ``ultrasonic_priority_range_cm`` must be
        selected for a left-zone detection.
        """
        us_left = config.KALMAN.ultrasonic_priority_range_cm - 30.0
        det = _make_detection(center_x=80, bbox=(10, 100, 150, 420))
        fusion.fuse([det], us_left, None, None, 640, 480, 0.033)
        assert det.sensor_distance_cm == pytest.approx(us_left)

    def test_ultrasonic_selected_for_right_zone(self, fusion):
        """
        Right ultrasonic within range must be selected for a right-zone
        detection.
        """
        us_right = config.KALMAN.ultrasonic_priority_range_cm - 30.0
        det = _make_detection(center_x=540, bbox=(465, 100, 615, 420))
        fusion.fuse([det], None, us_right, None, 640, 480, 0.033)
        assert det.sensor_distance_cm == pytest.approx(us_right)

    def test_camera_used_when_no_hardware_sensor(self, fusion):
        """
        When no ultrasonic or ToF reading is available for a class with a
        known height, the camera pinhole estimate seeds the tracker and
        ``fused_distance_cm`` must be set.
        """
        det = _make_detection(
            class_name="person",
            center_x=320,
            bbox=(200, 160, 440, 480),   # height = 320 px → ≈ 266 cm
        )
        fusion.fuse([det], None, None, None, 640, 480, 0.033)
        assert det.estimated_distance_cm is not None
        assert det.estimated_distance_cm > 0.0
        assert det.fused_distance_cm is not None

    def test_unknown_class_without_sensor_is_skipped(self, fusion):
        """
        A detection with an unknown class name and no sensor reading has no
        seed distance; the tracker must not be created and
        ``fused_distance_cm`` must remain ``None``.
        """
        det = _make_detection(
            class_name="bench",  # Not in config.KNOWN_HEIGHTS
            center_x=320,
            bbox=(200, 300, 440, 450),
        )
        fusion.fuse([det], None, None, None, 640, 480, 0.033)
        assert det.fused_distance_cm is None

    def test_unknown_class_with_sensor_is_tracked(self, fusion):
        """
        An unknown-class detection supplied with a hardware sensor reading
        CAN be seeded and must have ``fused_distance_cm`` set.
        """
        det = _make_detection(
            class_name="bench",
            center_x=80,
            bbox=(10, 300, 150, 450),
        )
        fusion.fuse([det], 120.0, None, None, 640, 480, 0.033)
        assert det.fused_distance_cm is not None

    # ── Edge cases ────────────────────────────────────────────────────────────

    def test_empty_detections_returns_empty(self, fusion):
        """``fuse([])`` must return an empty list without raising."""
        result = fusion.fuse([], None, None, None, 640, 480, 0.033)
        assert result == []

    def test_zero_frame_dimensions_returns_unchanged(self, fusion):
        """``fuse()`` with zero frame dimensions must return detections
        unmodified (position not reassigned, no tracker created)."""
        det = _make_detection()
        original_fused = det.fused_distance_cm
        result = fusion.fuse([det], None, None, None, 0, 0, 0.033)
        assert result == [det]
        assert det.fused_distance_cm == original_fused

    def test_fuse_returns_same_list_object(self, fusion):
        """``fuse()`` must return the exact same list object (in-place)."""
        dets = [_make_detection(center_x=320, bbox=(200, 160, 440, 480))]
        result = fusion.fuse(dets, None, None, None, 640, 480, 0.033)
        assert result is dets

    def test_fused_distance_is_positive(self, fusion):
        """``fused_distance_cm`` must be > 0 when the tracker is seeded."""
        det = _make_detection(center_x=320, bbox=(200, 160, 440, 480))
        fusion.fuse([det], None, None, None, 640, 480, 0.033)
        if det.fused_distance_cm is not None:
            assert det.fused_distance_cm > 0.0

    # ── Multi-frame convergence ───────────────────────────────────────────────

    def test_multiple_frames_converge_with_tof(self, fusion):
        """
        Running 10 frames with a fixed ToF reading must drive
        ``fused_distance_cm`` close to that reading.
        """
        tof_cm = 120.0
        det = _make_detection(center_x=320, bbox=(200, 160, 440, 480))
        for _ in range(10):
            det.fused_distance_cm = None
            det.sensor_distance_cm = None
            fusion.fuse([det], None, None, tof_cm, 640, 480, 0.033)
        assert det.fused_distance_cm is not None
        assert abs(det.fused_distance_cm - tof_cm) < 30.0, (
            f"Fused {det.fused_distance_cm:.1f} cm should be near ToF {tof_cm:.1f} cm"
        )

    def test_predict_only_step_does_not_crash(self, fusion):
        """
        A fuse() call with no sensor readings for a previously seeded tracker
        must not raise (predict-only Kalman step).
        """
        det = _make_detection(center_x=320, bbox=(200, 160, 440, 480))
        # First frame: seed the tracker with ToF.
        fusion.fuse([det], None, None, 120.0, 640, 480, 0.033)
        # Second frame: no sensors — predict-only.
        det.fused_distance_cm = None
        fusion.fuse([det], None, None, None, 640, 480, 0.033)
        assert det.fused_distance_cm is not None

    # ── get_tracked_objects ───────────────────────────────────────────────────

    def test_get_tracked_objects_returns_dict(self, fusion):
        """``get_tracked_objects()`` must return a dict of float-pair tuples."""
        det = _make_detection(center_x=320, bbox=(200, 160, 440, 480))
        fusion.fuse([det], None, None, None, 640, 480, 0.033)
        tracked = fusion.get_tracked_objects()
        assert isinstance(tracked, dict)
        for oid, state in tracked.items():
            assert isinstance(oid, str)
            dist_cm, vel = state
            assert isinstance(dist_cm, float)
            assert isinstance(vel, float)

    def test_get_tracked_objects_is_thread_safe(self, fusion):
        """
        ``get_tracked_objects()`` must return without deadlock when called
        concurrently with ``fuse()``.
        """
        det = _make_detection(center_x=320, bbox=(200, 160, 440, 480))
        errors: list = []

        def reader():
            for _ in range(50):
                try:
                    fusion.get_tracked_objects()
                except Exception as exc:
                    errors.append(exc)

        def writer():
            for _ in range(50):
                try:
                    d = _make_detection(center_x=320, bbox=(200, 160, 440, 480))
                    fusion.fuse([d], None, None, None, 640, 480, 0.033)
                except Exception as exc:
                    errors.append(exc)

        threads = [threading.Thread(target=reader), threading.Thread(target=writer)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        assert not errors, f"Thread-safety errors: {errors}"

    def test_multiple_detections_create_separate_trackers(self, fusion):
        """
        Two detections with different class-position keys must produce two
        independent trackers in ``get_tracked_objects()``.
        """
        person = _make_detection(
            class_name="person", center_x=80, bbox=(10, 100, 150, 480),
        )
        car = _make_detection(
            class_name="car", center_x=540, bbox=(465, 100, 630, 480),
        )
        fusion.fuse(
            [person, car],
            120.0,   # left ultrasonic
            110.0,   # right ultrasonic
            None,
            640, 480, 0.033,
        )
        tracked = fusion.get_tracked_objects()
        assert len(tracked) >= 2

    # ── Stale tracker eviction ────────────────────────────────────────────────

    def test_stale_tracker_evicted_after_timeout(self, fusion):
        """
        Trackers not updated within ``_STALE_TIMEOUT_SEC`` must be evicted
        the next time ``fuse()`` is called.
        """
        det = _make_detection(center_x=320, bbox=(200, 160, 440, 480))
        fusion.fuse([det], None, None, None, 640, 480, 0.033)
        assert len(fusion.get_tracked_objects()) >= 1

        # Backdate all timestamps so the trackers appear stale.
        with fusion._lock:
            for oid in list(fusion._last_update_times):
                fusion._last_update_times[oid] -= (_STALE_TIMEOUT_SEC + 1.0)

        # An empty fuse() call triggers the eviction sweep.
        fusion.fuse([], None, None, None, 640, 480, 0.033)
        assert len(fusion.get_tracked_objects()) == 0

    def test_active_tracker_not_evicted_prematurely(self, fusion):
        """
        A tracker that receives regular updates must not be evicted between
        frames.
        """
        det = _make_detection(center_x=320, bbox=(200, 160, 440, 480))
        for _ in range(5):
            det.fused_distance_cm = None
            fusion.fuse([det], None, None, None, 640, 480, 0.033)
        assert len(fusion.get_tracked_objects()) >= 1

    def test_tracker_key_uses_class_and_position(self, fusion):
        """
        The tracker key must be ``f'{class_name}_{position}'`` so two
        objects of the same class in different zones are tracked independently.
        """
        person_left = _make_detection(
            class_name="person", center_x=80, bbox=(10, 100, 150, 480),
        )
        person_right = _make_detection(
            class_name="person", center_x=540, bbox=(465, 100, 615, 480),
        )
        fusion.fuse(
            [person_left, person_right],
            150.0,  # left ultrasonic
            140.0,  # right ultrasonic
            None,
            640, 480, 0.033,
        )
        tracked = fusion.get_tracked_objects()
        keys = set(tracked.keys())
        assert "person_left" in keys
        assert "person_right" in keys
