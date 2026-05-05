"""
ClarifEye Ultrasonic Sensor Module

Manages one or two HC-SR04 sensors via RPi.GPIO using BCM pin numbering.
The ECHO lines are already level-shifted to 3.3 V by hardware voltage dividers,
so no software handling is required beyond a normal GPIO input read.
"""
import logging
import time
from typing import Optional, Tuple

try:
    from .. import config
except ImportError:
    import config  # type: ignore[no-redef]

logger = logging.getLogger("clarifeye.hardware.ultrasonic")

# ── Optional dependency guard ─────────────────────────────────────────────────
_GPIO_AVAILABLE = False
try:
    import RPi.GPIO as GPIO  # type: ignore[import-untyped]
    _GPIO_AVAILABLE = True
except (ImportError, RuntimeError):
    logger.warning(
        "RPi.GPIO not available — ultrasonic sensors will always return None."
    )


class UltrasonicSensor:
    """
    Single HC-SR04 ultrasonic distance sensor.

    Sends a 10 µs trigger pulse and times the echo response to calculate
    the distance via the speed-of-sound formula:

        distance_cm = (echo_duration_s × 34 300) / 2

    Valid measurement range: 2 cm – 400 cm.  Readings outside this range or
    any GPIO/timing error return ``None``.
    """

    def __init__(self, trig_pin: int, echo_pin: int, name: str) -> None:
        """
        Configure GPIO pins for one HC-SR04 sensor.

        Args:
            trig_pin: BCM GPIO number connected to the TRIG pin.
            echo_pin: BCM GPIO number connected to the ECHO pin (via divider).
            name:     Human-readable label used in log messages.
        """
        self._trig = trig_pin
        self._echo = echo_pin
        self._name = name
        self._available = False

        if not _GPIO_AVAILABLE:
            return

        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self._trig, GPIO.OUT, initial=GPIO.LOW)
            GPIO.setup(self._echo, GPIO.IN)
            self._available = True
            logger.info(
                "UltrasonicSensor '%s' ready  TRIG=GPIO%d  ECHO=GPIO%d.",
                name,
                trig_pin,
                echo_pin,
            )
        except Exception as exc:
            logger.error(
                "UltrasonicSensor '%s' GPIO setup failed: %s", name, exc
            )

    def measure_distance_cm(self) -> Optional[float]:
        """
        Perform one distance measurement.

        Each call sends a single 10 µs trigger pulse and waits for the echo.
        Both the rising-edge and falling-edge waits have a 50 ms hard timeout
        to prevent the thread from blocking indefinitely on a missing echo.

        Returns:
            Distance in centimetres (float, 1 decimal place), or ``None`` if
            the reading is out of the valid 2–400 cm range or an error occurs.
        """
        if not self._available:
            return None

        try:
            # Ensure TRIG is settled LOW before sending a new pulse.
            GPIO.output(self._trig, GPIO.LOW)
            time.sleep(0.000002)  # 2 µs settling time

            # Send 10 µs trigger pulse.
            GPIO.output(self._trig, GPIO.HIGH)
            time.sleep(0.000010)  # 10 µs
            GPIO.output(self._trig, GPIO.LOW)

            # Wait for ECHO to go HIGH — record pulse start time.
            timeout = time.time() + 0.05
            pulse_start = time.time()
            while GPIO.input(self._echo) == GPIO.LOW:
                pulse_start = time.time()
                if pulse_start > timeout:
                    logger.debug("%s: timeout waiting for ECHO HIGH.", self._name)
                    return None

            # Wait for ECHO to go LOW — record pulse end time.
            timeout = time.time() + 0.05
            pulse_end = time.time()
            while GPIO.input(self._echo) == GPIO.HIGH:
                pulse_end = time.time()
                if pulse_end > timeout:
                    logger.debug("%s: timeout waiting for ECHO LOW.", self._name)
                    return None

            elapsed_s = pulse_end - pulse_start
            distance_cm = (elapsed_s * 34_300.0) / 2.0

            if distance_cm < 2.0 or distance_cm > 400.0:
                return None

            return round(distance_cm, 1)

        except Exception as exc:
            logger.error("%s measurement error: %s", self._name, exc)
            return None

    def cleanup(self) -> None:
        """Release the GPIO pins claimed by this sensor."""
        if not _GPIO_AVAILABLE or not self._available:
            return
        try:
            GPIO.cleanup([self._trig, self._echo])
            logger.info("UltrasonicSensor '%s' GPIO released.", self._name)
        except Exception as exc:
            logger.error("UltrasonicSensor '%s' cleanup error: %s", self._name, exc)


class DualUltrasonic:
    """
    Convenience wrapper that manages both the LEFT and RIGHT HC-SR04 sensors
    defined in ``config``.

    Measurements are sequential (left then right) to avoid acoustic
    cross-talk between the two sensors.
    """

    def __init__(self) -> None:
        """Initialise left and right sensors using pin assignments from config."""
        self._left = UltrasonicSensor(
            trig_pin=config.ULTRASONIC_LEFT_TRIG,
            echo_pin=config.ULTRASONIC_LEFT_ECHO,
            name="left",
        )
        self._right = UltrasonicSensor(
            trig_pin=config.ULTRASONIC_RIGHT_TRIG,
            echo_pin=config.ULTRASONIC_RIGHT_ECHO,
            name="right",
        )
        logger.info("DualUltrasonic initialised.")

    def measure_both(self) -> Tuple[Optional[float], Optional[float]]:
        """
        Measure distance from both sensors sequentially.

        Returns:
            ``(left_cm, right_cm)`` — either value may be ``None`` if the
            corresponding sensor read failed or returned an out-of-range value.
        """
        left_cm = self._left.measure_distance_cm()
        right_cm = self._right.measure_distance_cm()
        return left_cm, right_cm

    def get_minimum_distance(self) -> Optional[float]:
        """
        Return the closest reading across both sensors.

        Returns:
            Minimum distance in centimetres, or ``None`` if both sensors
            returned ``None``.
        """
        left_cm, right_cm = self.measure_both()
        valid = [d for d in (left_cm, right_cm) if d is not None]
        return min(valid) if valid else None

    def cleanup(self) -> None:
        """Release GPIO resources for both sensors."""
        self._left.cleanup()
        self._right.cleanup()
        logger.info("DualUltrasonic cleaned up.")
