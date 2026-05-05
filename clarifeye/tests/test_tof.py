"""
tests/test_tof.py
=================
Tests for ``hardware/tof_sensor.py`` (VL53L0X ToF sensor).

All tests require the physical I2C sensor and are **skipped** on non-Pi
machines.  ``ToFSensor`` returns ``None`` gracefully when hardware is absent,
but the Pi-specific tests validate live I2C readings.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _is_raspberry_pi() -> bool:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            return "BCM2711" in fh.read()
    except OSError:
        return False


pytestmark = pytest.mark.skipif(
    not _is_raspberry_pi(),
    reason="Requires Raspberry Pi 4 with VL53L0X wired to I2C bus 1",
)

from hardware.tof_sensor import ToFSensor  # noqa: E402


# ── Fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tof():
    """Initialise a ToFSensor instance for the test module."""
    sensor = ToFSensor()
    yield sensor
    sensor.cleanup()


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_tof_initialises_without_error(tof):
    """
    ToFSensor construction must succeed (not raise) on a Pi with the VL53L0X
    connected.  After construction the sensor should report as available.
    """
    assert tof is not None


def test_tof_is_available_returns_true(tof):
    """
    ``is_available()`` performs a live I2C probe.  On a correctly wired Pi 4
    it must return ``True``.
    """
    assert tof.is_available() is True, (
        "ToF sensor not responding — check I2C wiring and address 0x29."
    )


def test_tof_measure_distance_returns_float_or_none(tof):
    """
    ``measure_distance_cm()`` must return a ``float`` or ``None``.
    It must never raise, even if the object is out of range.
    """
    result = tof.measure_distance_cm()
    assert result is None or isinstance(result, float), (
        f"Expected float or None, got {type(result)}"
    )


def test_tof_reading_in_valid_range(tof):
    """
    Any non-``None`` reading must be within the VL53L0X's reliable operating
    range (0 cm < d ≤ 200 cm).  The sensor implementation rejects values
    outside this window.
    """
    result = tof.measure_distance_cm()
    if result is not None:
        assert 0.0 < result <= 200.0, f"Reading out of range: {result} cm"


def test_tof_multiple_readings_stable(tof):
    """
    Five consecutive reads must all be ``float | None`` without raising.
    Validates that repeated I2C transactions do not leave the bus in a bad
    state.
    """
    for _ in range(5):
        result = tof.measure_distance_cm()
        assert result is None or isinstance(result, float)


def test_tof_cleanup_does_not_raise(tof):
    """
    ``cleanup()`` must release the I2C bus without raising, even if called
    on a sensor that was already cleaned up by the module fixture's teardown.
    """
    try:
        tof.cleanup()
    except Exception as exc:
        pytest.fail(f"ToFSensor.cleanup() raised: {exc}")
