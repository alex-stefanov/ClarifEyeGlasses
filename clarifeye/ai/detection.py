"""
ClarifEye Shared Detection Dataclass

Single source of truth for the ``Detection`` type used by every detector,
the sensor-fusion layer, and the priority engine.

Field ownership
~~~~~~~~~~~~~~~
* **Detector**       fills: ``bbox``, ``class_id``, ``class_name``,
                            ``confidence``, ``center_x``, ``center_y``,
                            ``position``.
* **ObjectDetector** additionally fills: ``estimated_distance_cm`` via the
                     pinhole camera model.
* **SensorFusion**   fills: ``sensor_distance_cm``, ``fused_distance_cm``.
* **PriorityEngine** fills: ``threat_score``.

All sensor-fusion and priority fields default to ``None`` / ``0.0`` so
detectors can construct a ``Detection`` without knowing downstream state.
"""
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class Detection:
    """
    A detected object anchored to original-frame pixel coordinates.

    Bounding box convention: ``(x1, y1, x2, y2)`` with the origin at the
    top-left corner of the frame (OpenCV / standard image convention).
    """

    # ── Set by detector ───────────────────────────────────────────────────────
    bbox: Tuple[int, int, int, int]
    """``(x1, y1, x2, y2)`` in original-frame pixel space."""

    class_id: int
    """Model-specific class index (e.g. 0 = red for traffic lights,
    0 = person for COCO SSD)."""

    class_name: str
    """Human-readable label (e.g. ``"person"``, ``"car"``, ``"red"``)."""

    confidence: float
    """Detection confidence in ``[0.0, 1.0]``."""

    center_x: int
    """Horizontal centre of ``bbox`` in original-frame pixels."""

    center_y: int
    """Vertical centre of ``bbox`` in original-frame pixels."""

    position: str = "center"
    """Horizontal thirds of the frame: ``"left"`` | ``"center"`` | ``"right"``."""

    # ── Set by ObjectDetector (camera-only pinhole estimate) ──────────────────
    estimated_distance_cm: Optional[float] = None
    """Distance estimated from bounding-box height via the pinhole model."""

    # ── Set by SensorFusion ───────────────────────────────────────────────────
    sensor_distance_cm: Optional[float] = None
    """Nearest reading from the ultrasonic or ToF sensor for this object."""

    fused_distance_cm: Optional[float] = None
    """Kalman-filtered best-estimate distance (final output of fusion)."""

    # ── Set by SensorFusion (tracker association) ─────────────────────────────
    tracker_id: Optional[str] = None
    """Stable per-object identity key assigned by the fusion layer (e.g. 'car_0').
    Used by the priority engine for cooldown keying; falls back to class_name when None."""

    # ── Set by PriorityEngine ─────────────────────────────────────────────────
    threat_score: float = 0.0
    """Composite urgency score assigned by the priority engine."""
