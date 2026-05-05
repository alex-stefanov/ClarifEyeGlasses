"""
ClarifEye Sensor Fusion Module

Combines per-object camera distance estimates with spatially-mapped ultrasonic
and Time-of-Flight sensor readings using a 1-D Kalman filter (constant-velocity
motion model) per tracked object.

Physical sensor geometry
~~~~~~~~~~~~~~~~~~~~~~~~
* Left ultrasonic  (HC-SR04, ~30° cone) — most representative of objects in
  the LEFT third of the frame.
* Right ultrasonic (HC-SR04, ~30° cone) — most representative of the RIGHT
  third.
* Centre ToF       (VL53L0X, ~25° cone) — highest precision for CENTRE
  detections within 200 cm.
* Camera pinhole estimate — per-object and class-aware, but imprecise (error
  grows quickly beyond ~3 m).

Sensor selection priority (evaluated per detection each frame)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. **ToF**          — detection in the CENTER zone AND reading is below
                      ``config.KALMAN.tof_priority_range_cm``.
2. **Ultrasonic**   — relevant side sensor below
                      ``config.KALMAN.ultrasonic_priority_range_cm``.
                      CENTER zone uses the minimum of both sides.
3. **Camera**       — always available when the class has a known real height.
4. **Predict-only** — Kalman dead-reckoning from the last valid update.

Object identity and tracker lifetime
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Trackers are keyed by ``f"{class_name}_{position}"`` (e.g. ``"person_center"``,
``"car_left"``).  Any tracker that receives no update for longer than
``_STALE_TIMEOUT_SEC`` seconds is automatically evicted.

Kalman filter (scratch implementation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
State  ``x  = [distance_cm, velocity_cm_s]``  (column vector)
Motion  ``F  = [[1, dt], [0, 1]]``            (constant-velocity model)
Measurement  ``H  = [[1, 0]]``               (observe distance only)
Process noise ``Q  = q · I₂``
Measurement noise ``R`` — scalar, varies per sensor (from ``config.KALMAN``).

Thread safety
~~~~~~~~~~~~~
``_trackers`` and ``_last_update_times`` are guarded by a ``threading.Lock``
so :meth:`SensorFusion.fuse` and :meth:`SensorFusion.get_tracked_objects`
can be called from different threads safely.
"""
import logging
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

# Max pixel distance (on a 640x640 frame) for associating a detection to an
# existing tracker of the same class.  Configured in config.ASSOCIATION_MAX_PX.

try:
    from .. import config
except ImportError:
    import config  # type: ignore[no-redef]

try:
    from .detection import Detection
except ImportError:
    try:
        from ai.detection import Detection  # type: ignore[no-redef]
    except ImportError:
        from detection import Detection  # type: ignore[no-redef]

logger = logging.getLogger("clarifeye.core.sensor_fusion")

# Trackers not updated within this window are evicted.
_STALE_TIMEOUT_SEC: float = 2.0


# ─── Per-object Kalman filter ─────────────────────────────────────────────────

class SingleObjectKalman:
    """
    1-D constant-velocity Kalman filter for a single object's distance.

    State vector ``x = [distance_cm, velocity_cm_s]``.

    All matrices use ``float64`` for numerical stability.  The distance
    component of the state is clamped to ``≥ 0`` after every step because a
    negative distance is physically meaningless.
    """

    def __init__(self, initial_distance: float) -> None:
        """
        Initialise state and covariance.

        Args:
            initial_distance: Best available distance estimate in centimetres
                              at the time the object is first detected.
        """
        self._x: np.ndarray = np.array(
            [float(initial_distance), 0.0], dtype=np.float64
        )
        # Initial covariance — high uncertainty in velocity, lower in position.
        self._P: np.ndarray = (
            np.eye(2, dtype=np.float64) * config.KALMAN.initial_covariance
        )
        self._q: float = float(config.KALMAN.process_noise)

    def predict(self, dt: float) -> None:
        """
        Propagate the state estimate forward by ``dt`` seconds.

        Uses a constant-velocity motion model:
        ``x_pred = F · x``,  ``P_pred = F · P · Fᵀ + Q``

        Args:
            dt: Elapsed time since the last predict/update, in seconds.
                Clamped to a minimum of 1 µs to guard against zero or
                negative values.
        """
        dt = max(float(dt), 1e-6)

        # State transition: constant velocity.
        F = np.array([[1.0, dt],
                      [0.0, 1.0]], dtype=np.float64)

        # Process noise: isotropic q·I₂.
        # The scalar q from config maps directly to the diagonal of Q, so
        # larger q → filter trusts motion model less and adapts to new
        # measurements faster.
        Q = np.eye(2, dtype=np.float64) * self._q

        self._x = F @ self._x
        self._P = F @ self._P @ F.T + Q

        # Physical constraint.
        self._x[0] = max(0.0, self._x[0])

    def update(self, measurement: float, measurement_noise: float) -> None:
        """
        Incorporate a new distance measurement (Kalman correction step).

        Equations (with H = [[1, 0]], so H·x = x[0]):

        * Innovation:          ``y = z − x[0]``
        * Innovation cov:      ``S = P[0,0] + R``
        * Kalman gain:         ``K = P[:,0] / S``    (shape [2])
        * State update:        ``x = x + K·y``
        * Covariance update:   ``P = (I − K⊗[1,0])·P``

        Args:
            measurement:       Sensor reading in centimetres.
            measurement_noise: Variance of this sensor type (from
                               ``config.KALMAN``).  Higher → trust the model
                               more; lower → trust the sensor more.
        """
        if not np.isfinite(measurement) or measurement < 0.0:
            return
        measurement_noise = max(float(measurement_noise), 1e-9)

        # ── Innovation ────────────────────────────────────────────────────────
        y: float = measurement - float(self._x[0])

        # ── Innovation covariance (scalar) ────────────────────────────────────
        S: float = float(self._P[0, 0]) + measurement_noise
        if S < 1e-10:
            return

        # ── Kalman gain (shape [2]) ───────────────────────────────────────────
        K: np.ndarray = self._P[:, 0] / S

        # ── State update ──────────────────────────────────────────────────────
        self._x = self._x + K * y

        # ── Covariance update: P = (I − K·H)·P ───────────────────────────────
        # K·H = outer(K, [1, 0]) = [[K[0], 0], [K[1], 0]]
        KH: np.ndarray = np.outer(K, np.array([1.0, 0.0]))
        self._P = (np.eye(2, dtype=np.float64) - KH) @ self._P

        # Symmetrise to prevent numerical drift from floating-point asymmetry.
        self._P = (self._P + self._P.T) * 0.5

        # Physical constraint.
        self._x[0] = max(0.0, self._x[0])

    def get_state(self) -> Tuple[float, float]:
        """
        Return the current state estimate.

        Returns:
            ``(distance_cm, velocity_cm_s)`` where velocity is positive when
            the object is receding and negative when approaching.
        """
        return float(self._x[0]), float(self._x[1])


# ─── Fusion orchestrator ──────────────────────────────────────────────────────

class SensorFusion:
    """
    Multi-source distance fusion layer for the ClarifEye pipeline.

    Maintains one :class:`SingleObjectKalman` tracker per detected object,
    keyed by ``f"{class_name}_{N}"`` (e.g. ``"person_0"``, ``"car_1"``).
    Trackers are associated frame-to-frame by spatial proximity (greedy
    nearest-neighbour within ``config.ASSOCIATION_MAX_PX`` pixels) rather than
    by position label, so a person walking from left to centre reuses the
    same tracker rather than spawning a new one.
    On each call to :meth:`fuse`, the trackers are predicted forward, updated
    with the best available sensor reading, and the resulting fused distance
    is written back onto each :class:`~ai.detection.Detection`.
    """

    def __init__(self) -> None:
        """Initialise the tracker store and threading primitives."""
        self._trackers: Dict[str, SingleObjectKalman] = {}
        self._last_update_times: Dict[str, float] = {}
        # Per-tracker last-known (center_x, center_y) for proximity association.
        self._tracker_centers: Dict[str, Tuple[int, int]] = {}
        # Per-class sequential counter for stable tracker key generation.
        self._tracker_id_counters: Dict[str, int] = {}
        self._lock = threading.Lock()

    # ── Public API ─────────────────────────────────────────────────────────────

    def fuse(
        self,
        detections: List[Detection],
        ultrasonic_left_cm: Optional[float],
        ultrasonic_right_cm: Optional[float],
        tof_cm: Optional[float],
        frame_width: int,
        frame_height: int,
        dt: float,
    ) -> List[Detection]:
        """
        Fuse all sensor readings into each detection and return the updated list.

        For every detection in ``detections`` the method:

        1. Computes ``estimated_distance_cm`` (camera pinhole) if not already set.
        2. Assigns ``position`` (``"left"`` / ``"center"`` / ``"right"``) from
           the detection's horizontal centre relative to ``frame_width``.
        3. Selects the highest-priority valid sensor reading for that zone.
        4. Predicts the existing Kalman tracker (or creates a new one) and
           incorporates the chosen measurement.
        5. Writes ``fused_distance_cm`` and ``sensor_distance_cm`` back onto
           the detection object.

        After all detections are processed, trackers not updated within
        ``_STALE_TIMEOUT_SEC`` seconds are evicted.

        Args:
            detections:          Detections from the current frame.  Modified
                                 in-place and returned.
            ultrasonic_left_cm:  Left HC-SR04 reading in cm, or ``None``.
            ultrasonic_right_cm: Right HC-SR04 reading in cm, or ``None``.
            tof_cm:              VL53L0X reading in cm, or ``None``.
            frame_width:         Width of the source frame in pixels.
            frame_height:        Height of the source frame in pixels.
            dt:                  Seconds elapsed since the previous call.

        Returns:
            The same ``detections`` list with fusion fields populated.
        """
        if frame_width <= 0 or frame_height <= 0:
            return detections

        now: float = time.time()
        frame_third: int = max(1, frame_width // 3)

        with self._lock:
            for det in detections:

                # ── 1. Camera distance (if not already set by detector) ───────
                if det.estimated_distance_cm is None:
                    det.estimated_distance_cm = self._pinhole_distance(det)

                # ── 2. Assign position from actual frame geometry ─────────────
                if det.center_x < frame_third:
                    det.position = "left"
                elif det.center_x < 2 * frame_third:
                    det.position = "center"
                else:
                    det.position = "right"

                # ── 3. Select best sensor reading for this zone ───────────────
                measurement, noise = self._select_measurement(
                    det,
                    ultrasonic_left_cm,
                    ultrasonic_right_cm,
                    tof_cm,
                )
                det.sensor_distance_cm = measurement

                # ── 4. Kalman predict + update ────────────────────────────────
                object_id = self._associate_detection(det)
                det.tracker_id = object_id
                self._tracker_centers[object_id] = (det.center_x, det.center_y)

                if object_id in self._trackers:
                    tracker = self._trackers[object_id]
                    tracker.predict(dt)
                    if measurement is not None:
                        tracker.update(measurement, noise)
                else:
                    # Seed the new tracker with the best available distance.
                    init_dist: Optional[float] = (
                        measurement
                        if measurement is not None
                        else det.estimated_distance_cm
                    )
                    if init_dist is None or init_dist <= 0.0:
                        # Cannot initialise without a seed distance; skip.
                        continue

                    tracker = SingleObjectKalman(init_dist)
                    self._trackers[object_id] = tracker
                    logger.debug(
                        "Kalman initialised: '%s'  seed=%.1f cm.", object_id, init_dist
                    )

                self._last_update_times[object_id] = now

                # ── 5. Write fused result back onto the detection ─────────────
                fused_dist, velocity = tracker.get_state()
                det.fused_distance_cm = round(fused_dist, 1)

                logger.debug(
                    "Fused %-14s %-6s  cam=%s  sensor=%s  fused=%.1f cm  vel=%+.1f cm/s",
                    det.class_name,
                    det.position,
                    f"{det.estimated_distance_cm:.0f} cm"
                    if det.estimated_distance_cm is not None
                    else "—",
                    f"{measurement:.0f} cm"
                    if measurement is not None
                    else "—",
                    fused_dist,
                    velocity,
                )

            # ── Evict trackers not seen recently ─────────────────────────────
            stale: List[str] = [
                oid
                for oid, last_t in self._last_update_times.items()
                if now - last_t > _STALE_TIMEOUT_SEC
            ]
            for oid in stale:
                self._trackers.pop(oid, None)
                self._last_update_times.pop(oid, None)
                self._tracker_centers.pop(oid, None)
                logger.debug("Tracker evicted (stale): '%s'.", oid)

        return detections

    def get_tracked_objects(self) -> Dict[str, Tuple[float, float]]:
        """
        Return a thread-safe snapshot of all active trackers.

        Returns:
            ``{object_id: (distance_cm, velocity_cm_s)}`` for every tracker
            currently in the active set.
        """
        with self._lock:
            return {
                oid: tracker.get_state()
                for oid, tracker in self._trackers.items()
            }

    # ── Private helpers ────────────────────────────────────────────────────────

    def _associate_detection(self, detection: Detection) -> str:
        """
        Greedy nearest-neighbour tracker association for a single detection.

        Searches existing trackers of the same class for the one whose
        last-known centre is closest to *detection*'s centre.  If the closest
        tracker is within ``config.ASSOCIATION_MAX_PX`` pixels it is reused;
        otherwise a new tracker key is created.

        Args:
            detection: Incoming detection with center_x / center_y set.

        Returns:
            Tracker key string (e.g. ``"person_0"``, ``"car_2"``).
        """
        class_name = detection.class_name
        det_cx = detection.center_x
        det_cy = detection.center_y
        max_px = config.ASSOCIATION_MAX_PX
        max_dist_sq = max_px * max_px

        best_key: Optional[str] = None
        best_dist_sq: float = float("inf")

        prefix = class_name + "_"
        for key, (cx, cy) in self._tracker_centers.items():
            if not key.startswith(prefix):
                continue
            dx = det_cx - cx
            dy = det_cy - cy
            dist_sq = float(dx * dx + dy * dy)
            if dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_key = key

        if best_key is not None and best_dist_sq <= max_dist_sq:
            return best_key

        # No close tracker found — create a new unique key.
        n = self._tracker_id_counters.get(class_name, 0)
        self._tracker_id_counters[class_name] = n + 1
        new_key = f"{class_name}_{n}"
        logger.debug("New tracker: '%s'  centre=(%d,%d).", new_key, det_cx, det_cy)
        return new_key

    def _pinhole_distance(self, det: Detection) -> Optional[float]:
        """
        Estimate object distance via the pinhole camera model.

        ``distance_m = (known_height_m × focal_length_px) / bbox_height_px``

        Args:
            det: Detection with a valid ``bbox`` in original-frame pixels and
                 a ``class_name`` that may appear in ``config.KNOWN_HEIGHTS``.

        Returns:
            Estimated distance in centimetres (1 dp), or ``None`` if the
            class has no reference height or the bounding-box height is zero.
        """
        known_h_m: Optional[float] = config.KNOWN_HEIGHTS.get(det.class_name)
        if known_h_m is None:
            return None

        pixel_h: int = det.bbox[3] - det.bbox[1]   # y2 − y1
        if pixel_h <= 0:
            return None

        distance_m = (known_h_m * config.CAMERA_FOCAL_LENGTH_PX) / pixel_h
        return round(distance_m * 100.0, 1)

    def _select_measurement(
        self,
        det: Detection,
        ultrasonic_left_cm: Optional[float],
        ultrasonic_right_cm: Optional[float],
        tof_cm: Optional[float],
    ) -> Tuple[Optional[float], float]:
        """
        Return ``(best_measurement_cm, noise_variance)`` for this detection.

        Sensor selection rules
        ~~~~~~~~~~~~~~~~~~~~~~
        * **CENTER zone**:  ToF (if < ``tof_priority_range_cm``) → ultrasonic
          minimum (if < ``ultrasonic_priority_range_cm``) → camera.
        * **LEFT zone**:    Left ultrasonic → camera.
        * **RIGHT zone**:   Right ultrasonic → camera.

        The returned ``noise_variance`` is one of the three scalars in
        ``config.KALMAN`` (``measurement_noise_tof``,
        ``measurement_noise_ultrasonic``, ``measurement_noise_camera``).

        Args:
            det:                 Current detection with ``position`` already set.
            ultrasonic_left_cm:  Left sensor reading or ``None``.
            ultrasonic_right_cm: Right sensor reading or ``None``.
            tof_cm:              ToF reading or ``None``.

        Returns:
            ``(measurement_cm, noise_variance)``  —  ``measurement_cm`` is
            ``None`` when no valid reading is available (predict-only step).
        """
        position = det.position

        # Identify the ultrasonic reading relevant to this zone.
        if position == "left":
            relevant_us: Optional[float] = ultrasonic_left_cm
        elif position == "right":
            relevant_us = ultrasonic_right_cm
        else:
            # CENTER: use the nearer of the two sensors — whichever is more
            # likely to correspond to the object the camera sees.
            us_candidates = [
                x for x in (ultrasonic_left_cm, ultrasonic_right_cm)
                if x is not None
            ]
            relevant_us = min(us_candidates) if us_candidates else None

        # ── Priority 1: ToF (best precision at close range, centre only) ─────
        if (
            position == "center"
            and tof_cm is not None
            and tof_cm < config.KALMAN.tof_priority_range_cm
        ):
            return tof_cm, config.KALMAN.measurement_noise_tof

        # ── Priority 2: Ultrasonic (reliable 2–400 cm) ────────────────────────
        if (
            relevant_us is not None
            and relevant_us < config.KALMAN.ultrasonic_priority_range_cm
        ):
            return relevant_us, config.KALMAN.measurement_noise_ultrasonic

        # ── Priority 3: Camera pinhole ────────────────────────────────────────
        camera_d = det.estimated_distance_cm
        if camera_d is not None and camera_d > 0.0:
            return camera_d, config.KALMAN.measurement_noise_camera

        # ── No valid measurement — caller will do a predict-only update ───────
        return None, config.KALMAN.measurement_noise_camera
