"""
ClarifEye Test Configuration
=============================
Shared pytest fixtures available to every test file.

Pi detection
------------
Checks ``/proc/cpuinfo`` for the BCM2711 hardware identifier (Raspberry Pi 4).
Falls back to checking ``platform.machine()`` for ``aarch64``.

Fixtures
--------
``is_raspberry_pi``     — bool, session-scoped.
``sample_frame``        — 640×640 RGB uint8 ndarray, random pixel values.
``sample_frame_dark``   — 640×640 RGB uint8 ndarray, mean brightness < 30.
``mock_detections``     — List[Detection] with 5 varied objects/distances.
"""
import os
import platform
import sys

import numpy as np
import pytest

# ── Add project root to sys.path so all test files can import freely ──────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config
from ai.detection import Detection


# ── Pi detection ──────────────────────────────────────────────────────────────

def _detect_raspberry_pi() -> bool:
    """
    Return ``True`` when running on a Raspberry Pi 4 (BCM2711).

    Checks ``/proc/cpuinfo`` for the SoC identifier first, then falls back to
    checking the machine architecture.  Returns ``False`` on any I/O error
    (e.g. on Windows / macOS development machines).
    """
    # Primary: CPU info available on Linux.
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            cpuinfo = fh.read()
        if "BCM2711" in cpuinfo:
            return True
    except OSError:
        pass

    # Fallback: aarch64 + Linux is a good proxy for any modern Pi.
    return platform.machine() == "aarch64" and platform.system() == "Linux"


# ── Session-scoped fixtures ───────────────────────────────────────────────────

@pytest.fixture(scope="session")
def is_raspberry_pi() -> bool:
    """
    ``True`` when the test session is running on a Raspberry Pi 4.

    Used by hardware tests to skip when the physical hardware is unavailable.
    Session-scoped so the detection runs only once per test run.
    """
    return _detect_raspberry_pi()


# ── Frame fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def sample_frame() -> np.ndarray:
    """
    640×640×3 uint8 ndarray with random RGB pixel values [0, 255].

    Represents a typical daytime camera frame suitable for detection and
    enhancement modules.  Uses a fixed seed so tests are deterministic.
    """
    rng = np.random.default_rng(seed=42)
    return rng.integers(0, 256, size=(640, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_frame_dark() -> np.ndarray:
    """
    640×640×3 uint8 ndarray simulating a very dark scene.

    All pixels are drawn from [0, 25] so the mean brightness is well below
    ``config.LOW_LIGHT_BRIGHTNESS_THRESHOLD`` (60).  Appropriate for testing
    the LowLightEnhancer and the ``is_low_light()`` detection logic.
    """
    rng = np.random.default_rng(seed=99)
    return rng.integers(0, 26, size=(640, 640, 3), dtype=np.uint8)


# ── Detection fixture ─────────────────────────────────────────────────────────

@pytest.fixture
def mock_detections():
    """
    List of five ``Detection`` objects covering a range of threat levels,
    positions, and distances.

    Composition
    -----------
    1. ``car``          — center, 80 cm  (CRITICAL, within DISTANCE_CRITICAL_CM)
    2. ``person``       — left,   250 cm (MEDIUM, within DISTANCE_MEDIUM_CM)
    3. ``bicycle``      — right,  150 cm (HIGH, within DISTANCE_CLOSE_CM)
    4. ``bench``        — center, 380 cm (LOW, near DISTANCE_MEDIUM_CM)
    5. ``"red"``        — center, 600 cm (INFO traffic-light colour detection)
    """
    return [
        Detection(
            bbox=(200, 250, 440, 500),
            class_id=2,
            class_name="car",
            confidence=0.92,
            center_x=320,
            center_y=375,
            position="center",
            fused_distance_cm=80.0,
        ),
        Detection(
            bbox=(10, 150, 180, 550),
            class_id=0,
            class_name="person",
            confidence=0.84,
            center_x=95,
            center_y=350,
            position="left",
            fused_distance_cm=250.0,
        ),
        Detection(
            bbox=(450, 200, 620, 480),
            class_id=1,
            class_name="bicycle",
            confidence=0.76,
            center_x=535,
            center_y=340,
            position="right",
            fused_distance_cm=150.0,
        ),
        Detection(
            bbox=(220, 300, 420, 450),
            class_id=13,
            class_name="bench",
            confidence=0.61,
            center_x=320,
            center_y=375,
            position="center",
            fused_distance_cm=380.0,
        ),
        Detection(
            bbox=(280, 100, 360, 200),
            class_id=0,
            class_name="red",
            confidence=0.88,
            center_x=320,
            center_y=150,
            position="center",
            fused_distance_cm=600.0,
        ),
    ]
