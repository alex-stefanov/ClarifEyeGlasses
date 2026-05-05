"""
ClarifEye Button Handler Module

Registers GPIO interrupts for the three hardware buttons and dispatches to
caller-supplied callbacks with software debounce.

Buttons are wired active-LOW (press pulls pin to GND); internal pull-up
resistors are enabled in software so no external resistors are required.

Button roles
------------
NEXT MODE  (GPIO 24, Physical Pin 18) — cycle operating modes forward.
LANGUAGE   (GPIO 25, Physical Pin 22) — toggle English/Bulgarian.
ACTION     (GPIO  5, Physical Pin 29) — mode-specific action (OCR trigger,
                                        currency scan, scene description, etc.)

NEXT and LANGUAGE use a single FALLING-edge interrupt (simple click).
ACTION uses FALLING (press-start) + RISING (press-end) to measure duration,
which is forwarded to the on_action callback so handlers can distinguish a
short tap from a long press if needed.
"""
import logging
import time
from typing import Callable, Dict, Optional

try:
    from .. import config
except ImportError:
    import config  # type: ignore[no-redef]

logger = logging.getLogger("clarifeye.hardware.buttons")

# ── Optional dependency guard ─────────────────────────────────────────────────
_GPIO_AVAILABLE = False
try:
    import RPi.GPIO as GPIO  # type: ignore[import-untyped]
    _GPIO_AVAILABLE = True
except (ImportError, RuntimeError):
    logger.warning(
        "RPi.GPIO not available — buttons will not generate callbacks."
    )


class ButtonHandler:
    """
    Debounced GPIO button input with interrupt-driven callback dispatch.

    Three buttons are supported:

    * **NEXT MODE** (``config.BUTTON_NEXT_MODE``) — advance to the next mode.
    * **LANGUAGE**  (``config.BUTTON_LANGUAGE``)  — toggle English/Bulgarian.
    * **ACTION**    (``config.BUTTON_ACTION``)    — mode-specific action;
      callback receives the press duration in milliseconds.

    RPi.GPIO calls the registered callbacks on an internal interrupt thread.
    All callbacks are wrapped in ``try/except`` so an exception in user code
    never silently kills the interrupt handler.
    """

    def __init__(
        self,
        on_next_mode: Callable[[], None],
        on_language_toggle: Callable[[], None],
        on_action: Callable[[float], None],
    ) -> None:
        """
        Configure GPIO pins and register edge-detect interrupts.

        Args:
            on_next_mode:        Zero-argument callable — NEXT MODE pressed.
            on_language_toggle:  Zero-argument callable — LANGUAGE pressed.
            on_action:           One-argument callable(duration_ms: float) —
                                 ACTION released; receives press duration.
        """
        self._on_next_mode = on_next_mode
        self._on_language_toggle = on_language_toggle
        self._on_action = on_action

        # Per-pin last-accepted-FALLING-edge timestamp (milliseconds).
        self._last_press_ms: Dict[int, float] = {
            config.BUTTON_NEXT_MODE: 0.0,
            config.BUTTON_LANGUAGE:  0.0,
            config.BUTTON_ACTION:    0.0,
        }

        # Timestamp of the most recent FALLING edge on the ACTION pin.
        self._action_press_start_ms: Optional[float] = None

        self._available = False

        if not _GPIO_AVAILABLE:
            return

        try:
            GPIO.setmode(GPIO.BCM)

            for pin in (config.BUTTON_NEXT_MODE, config.BUTTON_LANGUAGE, config.BUTTON_ACTION):
                GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

            # Simple click buttons — FALLING edge only.
            GPIO.add_event_detect(
                config.BUTTON_NEXT_MODE,
                GPIO.FALLING,
                callback=self._on_next_pressed,
            )
            GPIO.add_event_detect(
                config.BUTTON_LANGUAGE,
                GPIO.FALLING,
                callback=self._on_language_pressed,
            )

            # ACTION button — BOTH edges so we can measure press duration.
            GPIO.add_event_detect(
                config.BUTTON_ACTION,
                GPIO.BOTH,
                callback=self._on_action_edge,
            )

            self._available = True
            logger.info(
                "ButtonHandler ready  NEXT=GPIO%d  LANG=GPIO%d  ACTION=GPIO%d  debounce=%d ms.",
                config.BUTTON_NEXT_MODE,
                config.BUTTON_LANGUAGE,
                config.BUTTON_ACTION,
                config.BUTTON_DEBOUNCE_MS,
            )
        except Exception as exc:
            logger.error("ButtonHandler initialisation failed: %s", exc)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _check_debounce(self, pin: int) -> bool:
        """
        Return True and update the timestamp if the debounce window has elapsed
        since the last accepted press on *pin*.
        """
        now_ms = time.time() * 1000.0
        elapsed_ms = now_ms - self._last_press_ms.get(pin, 0.0)
        if elapsed_ms >= config.BUTTON_DEBOUNCE_MS:
            self._last_press_ms[pin] = now_ms
            return True
        return False

    def _on_next_pressed(self, channel: int) -> None:
        """GPIO interrupt callback for the NEXT MODE button (FALLING edge)."""
        if self._check_debounce(channel):
            logger.debug("NEXT MODE button press accepted (GPIO%d).", channel)
            try:
                self._on_next_mode()
            except Exception as exc:
                logger.error("on_next_mode callback raised an exception: %s", exc)

    def _on_language_pressed(self, channel: int) -> None:
        """GPIO interrupt callback for the LANGUAGE button (FALLING edge)."""
        if self._check_debounce(channel):
            logger.debug("LANGUAGE button press accepted (GPIO%d).", channel)
            try:
                self._on_language_toggle()
            except Exception as exc:
                logger.error("on_language_toggle callback raised an exception: %s", exc)

    def _on_action_edge(self, channel: int) -> None:
        """
        GPIO interrupt callback for BOTH edges on the ACTION button.

        FALLING → record press-start time (with debounce).
        RISING  → compute duration, forward to on_action callback.
        """
        if not _GPIO_AVAILABLE:
            return

        pin_state = GPIO.input(channel)

        if pin_state == GPIO.LOW:
            # FALLING edge — button pressed.
            if self._check_debounce(channel):
                self._action_press_start_ms = time.time() * 1000.0
                logger.debug("ACTION button press-start (GPIO%d).", channel)
        else:
            # RISING edge — button released.
            if self._action_press_start_ms is None:
                return  # Spurious RISING with no recorded press-start; ignore.
            duration_ms = time.time() * 1000.0 - self._action_press_start_ms
            self._action_press_start_ms = None
            logger.debug(
                "ACTION button released (GPIO%d)  duration=%.0f ms.", channel, duration_ms
            )
            try:
                self._on_action(duration_ms)
            except Exception as exc:
                logger.error("on_action callback raised an exception: %s", exc)

    # ── Public API ────────────────────────────────────────────────────────────

    def cleanup(self) -> None:
        """Remove all event detection and release the button GPIO pins."""
        if not _GPIO_AVAILABLE or not self._available:
            return
        try:
            GPIO.remove_event_detect(config.BUTTON_NEXT_MODE)
            GPIO.remove_event_detect(config.BUTTON_LANGUAGE)
            GPIO.remove_event_detect(config.BUTTON_ACTION)
            GPIO.cleanup([config.BUTTON_NEXT_MODE, config.BUTTON_LANGUAGE, config.BUTTON_ACTION])
            logger.info("ButtonHandler GPIO released.")
        except Exception as exc:
            logger.error("ButtonHandler cleanup error: %s", exc)
