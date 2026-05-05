"""
ClarifEye Mode Manager

Owns the current operating mode and exposes thread-safe transitions.
Clears the audio queue on every mode switch so old messages don't bleed
into the new mode.
"""
import logging
import threading

try:
    from .. import config
except ImportError:
    import config

logger = logging.getLogger("clarifeye.core.mode_manager")

# Maps each Mode value to its audio key in the registry.
_MODE_KEYS = {
    config.Mode.TRAFFIC_LIGHT: "mode_traffic_light",
    config.Mode.NAVIGATION:    "mode_navigation",
    config.Mode.TEXT_READING:  "mode_text_reading",
}


class ModeManager:
    def __init__(self, audio_manager) -> None:
        self._mode = config.DEFAULT_MODE
        self._lock = threading.Lock()
        self._audio = audio_manager

    def get_current_mode(self):
        with self._lock:
            return self._mode

    def next_mode(self) -> None:
        with self._lock:
            new_index = (int(self._mode) + 1) % config.NUM_MODES
            self._mode = config.Mode(new_index)
            new_mode = self._mode

        self._flush_audio()
        self._announce_mode(new_mode)
        logger.info("Mode change -> %s", new_mode.name)

    def get_mode_name(self) -> str:
        with self._lock:
            return config.MODE_NAMES.get(self._mode, "Unknown")

    def announce_current_mode(self, priority: int = config.AUDIO_PRIORITY_NORMAL) -> None:
        """Announce the current mode using speak_key (lets AudioManager choose language)."""
        with self._lock:
            mode = self._mode
        self._announce_mode(mode, priority)

    def _announce_mode(
        self, mode, priority: int = config.AUDIO_PRIORITY_HIGH
    ) -> None:
        key = _MODE_KEYS.get(mode)
        if key is not None:
            self._audio.speak_key(key, priority)
        else:
            # Fallback for future modes not yet in the registry.
            name = config.MODE_NAMES.get(mode, "Unknown")
            self._audio.speak_text(name, priority)

    def _flush_audio(self) -> None:
        try:
            self._audio.flush()
        except Exception as exc:
            logger.error("Audio flush error: %s", exc)
