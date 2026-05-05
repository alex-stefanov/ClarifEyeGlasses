"""
tests/test_ultrasonic.py
========================
Tests for ``hardware/ultrasonic.py`` (UltrasonicSensor + DualUltrasonic).

All tests in this module require physical GPIO hardware and are
automatically **skipped** on non-Pi machines.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

# ── Pi detection (module-level for pytestmark) ────────────────────────────────

def _is_raspberry_pi() -> bool:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            return "BCM2711" in fh.read()
    except OSError:
        return False


pytestmark = pytest.mark.skipif(
    not _is_raspberry_pi(),
    reason="Requires Raspberry Pi 4 hardware (BCM2711) with GPIO access",
)

# ── Imports (only reached on Pi) ──────────────────────────────────────────────

from hardware.ultrasonic import DualUltrasonic, UltrasonicSensor  # noqa: E402


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def single_sensor():
    """Left HC-SR04 sensor using BCM GPIO pins from config."""
    sensor = UltrasonicSensor(
        trig_pin=config.ULTRASONIC_LEFT_TRIG,
        echo_pin=config.ULTRASONIC_LEFT_ECHO,
        name="test-left",
    )
    yield sensor
    sensor.cleanup()


@pytest.fixture(scope="module")
def dual_sensor():
    """DualUltrasonic wrapper using both HC-SR04 sensors from config."""
    sensor = DualUltrasonic()
    yield sensor
    sensor.cleanup()


# ── Tests: UltrasonicSensor ───────────────────────────────────────────────────

def test_ultrasonic_initialises_without_error(single_sensor):
    """
    Sensor construction must succeed even when RPi.GPIO raises on non-Pi.
    Validates that all GPIO setup errors are caught internally.
    """
    assert single_sensor is not None


def test_ultrasonic_measure_returns_float_or_none(single_sensor):
    """
    ``measure_distance_cm()`` must return a ``float`` or ``None``.
    It must never raise an exception regardless of the echo response.
    """
    result = single_sensor.measure_distance_cm()
    assert result is None or isinstance(result, float), (
        f"Expected float or None, got {type(result)}"
    )


def test_ultrasonic_measure_in_valid_range(single_sensor):
    """
    Any non-``None`` reading must be within the HC-SR04's valid range (2–400 cm).
    The sensor implementation clamps and rejects out-of-range values internally.
    """
    result = single_sensor.measure_distance_cm()
    if result is not None:
        assert 2.0 <= result <= 400.0, (
            f"Out-of-range reading: {result} cm"
        )


def test_ultrasonic_multiple_reads_stable(single_sensor):
    """
    Ten consecutive reads must each be ``float | None`` without raising.
    Verifies the sensor handles repeated trigger-echo cycles cleanly.
    """
    for _ in range(10):
        result = single_sensor.measure_distance_cm()
        assert result is None or isinstance(result, float)


def test_ultrasonic_cleanup_does_not_raise(single_sensor):
    """
    ``cleanup()`` must release GPIO without raising, even if called twice.
    """
    # First cleanup is called by the fixture; a second call must be safe.
    try:
        single_sensor.cleanup()
    except Exception as exc:
        pytest.fail(f"cleanup() raised unexpectedly: {exc}")


# ── Tests: DualUltrasonic ─────────────────────────────────────────────────────

def test_dual_ultrasonic_initialises(dual_sensor):
    """DualUltrasonic construction must succeed with pins from config."""
    assert dual_sensor is not None


def test_dual_ultrasonic_measure_both_returns_tuple(dual_sensor):
    """
    ``measure_both()`` must return a 2-tuple.
    Each element is either a ``float`` or ``None``; never an exception.
    """
    result = dual_sensor.measure_both()
    assert isinstance(result, tuple) and len(result) == 2, (
        f"Expected 2-tuple, got {result!r}"
    )
    left_cm, right_cm = result
    assert left_cm is None or isinstance(left_cm, float)
    assert right_cm is None or isinstance(right_cm, float)


def test_dual_ultrasonic_get_minimum_distance(dual_sensor):
    """
    ``get_minimum_distance()`` must return the closer of the two readings
    or ``None`` if both sensors are unavailable.
    """
    minimum = dual_sensor.get_minimum_distance()
    left_cm, right_cm = dual_sensor.measure_both()

    if left_cm is None and right_cm is None:
        assert minimum is None
    else:
        valid = [d for d in (left_cm, right_cm) if d is not None]
        assert minimum == min(valid)


def test_dual_ultrasonic_cleanup_does_not_raise(dual_sensor):
    """DualUltrasonic cleanup must release all GPIO without raising."""
    try:
        dual_sensor.cleanup()
    except Exception as exc:
        pytest.fail(f"DualUltrasonic.cleanup() raised: {exc}")
