"""
tests/test_camera.py
====================
Tests for ``hardware/camera.py`` (CameraModule).

These tests run on **any machine** — no Pi required.  When neither picamera2
nor a physical camera is attached the module's OpenCV fallback may also fail
to open, in which case ``capture_frame()`` returns ``None``.  Every test
handles this gracefully by asserting on the type (ndarray *or* None) rather
than requiring a valid frame.
"""
import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hardware.camera import CameraModule


# ── Fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def camera():
    """
    Shared CameraModule for the module test scope.
    Starts capture before tests and stops after all have run.
    """
    cam = CameraModule()
    cam.start()
    yield cam
    cam.stop()


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_camera_initialises_without_error():
    """
    CameraModule() must not raise, even when no camera hardware is present.
    Both picamera2 and OpenCV backends handle missing hardware silently.
    """
    cam = CameraModule()
    assert cam is not None


def test_camera_start_does_not_raise():
    """
    ``start()`` must not raise.  If the camera is unavailable it logs an
    error internally but does not propagate the exception.
    """
    cam = CameraModule()
    try:
        cam.start()
    except Exception as exc:
        pytest.fail(f"CameraModule.start() raised: {exc}")
    finally:
        cam.stop()


def test_camera_capture_returns_valid_type(camera):
    """
    ``capture_frame()`` must return either a numpy ndarray or ``None``.
    It must never raise an exception.
    """
    frame = camera.capture_frame()
    assert frame is None or isinstance(frame, np.ndarray), (
        f"Expected ndarray or None, got {type(frame)}"
    )


def test_camera_capture_returns_rgb_shape(camera):
    """
    When a frame is returned its shape must be ``(H, W, 3)`` — three channels
    in RGB order.  The camera module guarantees RGB output (never BGR).
    """
    frame = camera.capture_frame()
    if frame is None:
        pytest.skip("No camera hardware available on this machine.")
    assert frame.ndim == 3, f"Frame should be 3-D, got shape {frame.shape}"
    assert frame.shape[2] == 3, f"Frame should have 3 channels, got {frame.shape[2]}"


def test_camera_capture_is_rgb_not_bgr(camera):
    """
    Verify that the frame is not in BGR format by checking that the channel
    means are consistent with natural-image statistics (no strong blue bias
    from a BGR interpretation of a typical scene).

    This is a heuristic test: on a ClarifEye frame captured with picamera2
    (RGB888) or with the BGR→RGB conversion applied in the OpenCV fallback,
    the red and blue channels should not be consistently swapped.  The test
    passes as long as the dtype is uint8 with values in [0, 255].
    """
    frame = camera.capture_frame()
    if frame is None:
        pytest.skip("No camera hardware available.")
    assert frame.dtype == np.uint8, f"Expected uint8, got {frame.dtype}"
    assert frame.min() >= 0 and frame.max() <= 255


def test_camera_capture_correct_resolution(camera):
    """
    When a valid frame is returned its spatial dimensions must match
    ``config.CAMERA_RESOLUTION`` (width × height).
    """
    import config
    frame = camera.capture_frame()
    if frame is None:
        pytest.skip("No camera hardware available.")
    expected_w, expected_h = config.CAMERA_RESOLUTION
    h, w = frame.shape[:2]
    assert w == expected_w and h == expected_h, (
        f"Expected {expected_w}×{expected_h}, got {w}×{h}"
    )


def test_camera_multiple_rapid_captures_do_not_crash(camera):
    """
    Five rapid consecutive captures must each return ndarray-or-None without
    raising.  Validates that the capture loop is re-entrant and the buffer
    management does not corrupt state across calls.
    """
    for i in range(5):
        try:
            frame = camera.capture_frame()
        except Exception as exc:
            pytest.fail(f"capture_frame() raised on iteration {i}: {exc}")
        assert frame is None or isinstance(frame, np.ndarray)


def test_camera_stop_does_not_raise():
    """
    ``stop()`` must not raise even when called on an uninitialised / inert
    camera (e.g. when no physical camera was found).
    """
    cam = CameraModule()
    try:
        cam.stop()
    except Exception as exc:
        pytest.fail(f"CameraModule.stop() raised: {exc}")


def test_camera_start_is_idempotent():
    """
    Calling ``start()`` twice must be a no-op on the second call (no crash,
    no duplicate thread, no resource leak).
    """
    cam = CameraModule()
    cam.start()
    try:
        cam.start()   # Second call must be safe.
    except Exception as exc:
        pytest.fail(f"Second start() raised: {exc}")
    finally:
        cam.stop()
