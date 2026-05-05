"""
ClarifEye Action Dispatcher

Routes ACTION button presses to the handler registered for the current mode.
The "action_received" audio cue plays immediately on every press regardless
of mode, giving the user instant haptic-equivalent feedback.  The actual
mode handler runs in a background daemon thread so GPIO is never blocked.
"""
import logging
import threading
from typing import Callable, Dict, Optional

try:
    from .. import config
except ImportError:
    import config  # type: ignore[no-redef]

logger = logging.getLogger("clarifeye.core.action_dispatcher")


class ActionDispatcher:
    """
    Dispatch ACTION button presses to per-mode handler callables.

    Usage::

        dispatcher = ActionDispatcher(mode_manager, audio, settings)
        dispatcher.register_handler(config.Mode.TEXT_READING, my_ocr_action)
        # Wire dispatcher.on_action_pressed to ButtonHandler's on_action.

    Only one handler runs at a time.  A second press while a handler is still
    running is acknowledged (audio plays) but the handler call is dropped.
    """

    def __init__(self, mode_manager, audio_manager, settings) -> None:
        self._mode_manager = mode_manager
        self._audio = audio_manager
        self._settings = settings

        self._handlers: Dict[config.Mode, Callable[[float], None]] = {}

        # Set while a handler thread is running; cleared when it finishes.
        self._handler_busy = threading.Event()

    def register_handler(self, mode: config.Mode, handler: Callable[[float], None]) -> None:
        """
        Register a callable for *mode*.

        Args:
            mode:    The operating mode this handler applies to.
            handler: Callable that receives press duration in milliseconds.
                     It runs in a daemon thread — must be thread-safe.
        """
        self._handlers[mode] = handler
        logger.debug("Action handler registered for mode %s.", mode.name)

    def on_action_pressed(self, duration_ms: float) -> None:
        """
        Called by ButtonHandler when the ACTION button is released.

        Always plays "action_received" at HIGH priority with no cooldown so
        the user gets instant feedback.  Then dispatches to the mode handler
        (if one is registered and no handler is already running).

        Args:
            duration_ms: How long the button was held, in milliseconds.
        """
        # Immediate feedback — always plays, even if busy.
        self._audio.speak_key(
            "action_received",
            config.AUDIO_PRIORITY_HIGH,
            cooldown=None,
        )

        mode = self._mode_manager.get_current_mode()
        handler = self._handlers.get(mode)

        if handler is None:
            logger.debug(
                "No action handler registered for mode %s — press acknowledged only.",
                mode.name,
            )
            return

        if self._handler_busy.is_set():
            logger.warning(
                "Action handler still running (mode=%s) — new press ignored.", mode.name
            )
            return

        self._handler_busy.set()

        def _run() -> None:
            try:
                handler(duration_ms)
            except Exception as exc:
                logger.error(
                    "Action handler for mode %s raised an exception: %s",
                    mode.name, exc, exc_info=True,
                )
            finally:
                self._handler_busy.clear()

        t = threading.Thread(target=_run, daemon=True, name="clarifeye-action")
        t.start()
