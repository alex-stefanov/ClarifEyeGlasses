"""
ClarifEye Time-of-Flight Sensor Module

Wraps the Adafruit CircuitPython VL53L0X driver (adafruit-circuitpython-vl53l0x)
over I2C bus 1 on the Raspberry Pi.  Returns distances in centimetres with a
valid range of 0 cm < d ≤ 200 cm.
"""
import logging
from typing import Optional

try:
    from .. import config
except ImportError:
    import config  # type: ignore[no-redef]

logger = logging.getLogger("clarifeye.hardware.tof_sensor")

# ── Optional dependency guard ─────────────────────────────────────────────────
_TOF_AVAILABLE = False
try:
    import board  # type: ignore[import-untyped]
    import busio  # type: ignore[import-untyped]
    import adafruit_vl53l0x  # type: ignore[import-untyped]
    _TOF_AVAILABLE = True
except (ImportError, NotImplementedError):
    logger.warning(
        "adafruit_vl53l0x / busio / board not available "
        "— ToF sensor will always return None."
    )


class ToFSensor:
    """
    VL53L0X Time-of-Flight proximity sensor over I2C.

    Communicates on I2C bus 1 (``board.SCL`` / ``board.SDA``) at the default
    address ``0x29``.  The measurement timing budget is set from config so the
    caller controls the speed / accuracy trade-off.

    Valid output range: 0 cm < distance ≤ 200 cm.  Values outside this range,
    zero readings (sensor error code), and any I2C exceptions all return
    ``None``.
    """

    def __init__(self) -> None:
        """
        Initialise the I2C bus and configure the VL53L0X sensor.

        Sets ``measurement_timing_budget`` to ``config.TOF_TIMING_BUDGET_US``.
        If the hardware or libraries are not available the instance stays inert
        and all measurement methods return ``None`` / ``False``.
        """
        self._i2c: Optional["busio.I2C"] = None  # type: ignore[name-defined]
        self._sensor: Optional["adafruit_vl53l0x.VL53L0X"] = None  # type: ignore[name-defined]
        self._available: bool = False

        if not _TOF_AVAILABLE:
            return

        try:
            self._i2c = busio.I2C(board.SCL, board.SDA)
            self._sensor = adafruit_vl53l0x.VL53L0X(
                self._i2c,
                address=config.TOF_I2C_ADDRESS,
            )
            self._sensor.measurement_timing_budget = config.TOF_TIMING_BUDGET_US
            self._available = True
            logger.info(
                "ToFSensor ready  I2C-bus=%d  address=0x%02X  timing-budget=%d µs.",
                config.TOF_I2C_BUS,
                config.TOF_I2C_ADDRESS,
                config.TOF_TIMING_BUDGET_US,
            )
        except Exception as exc:
            logger.error("ToFSensor initialisation failed: %s", exc)
            self._available = False

    def measure_distance_cm(self) -> Optional[float]:
        """
        Read the current range from the VL53L0X.

        The sensor returns millimetres; this method converts to centimetres
        and applies the validity filter (0 mm is a sensor error; > 2000 mm
        is beyond the configured maximum range).

        Returns:
            Distance in centimetres (float, 1 decimal place), or ``None`` on
            any error or out-of-range condition.
        """
        if not self._available or self._sensor is None:
            return None

        try:
            range_mm: int = self._sensor.range
            if range_mm <= 0:
                return None
            distance_cm = range_mm / 10.0
            if distance_cm > 200.0:
                return None
            return round(distance_cm, 1)
        except Exception as exc:
            logger.error("ToFSensor measurement error: %s", exc)
            return None

    def is_available(self) -> bool:
        """
        Probe the sensor to confirm it is still responding on I2C.

        Performs a live range read rather than returning cached state so that
        transient I2C errors are detected.

        Returns:
            ``True`` if the sensor responds without raising an exception.
        """
        if not self._available or self._sensor is None:
            return False
        try:
            _ = self._sensor.range
            return True
        except Exception as exc:
            logger.warning("ToFSensor availability check failed: %s", exc)
            return False

    def cleanup(self) -> None:
        """De-initialise the I2C bus and release all sensor resources."""
        self._available = False
        self._sensor = None
        try:
            if self._i2c is not None:
                self._i2c.deinit()
                logger.info("ToFSensor I2C bus de-initialised.")
        except Exception as exc:
            logger.error("ToFSensor cleanup error: %s", exc)
        finally:
            self._i2c = None
