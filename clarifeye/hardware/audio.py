"""
ClarifEye Audio Manager

Three-backend TTS system (priority order):
  1. WAV file player  — pre-recorded human-voice files for fixed vocabulary.
  2. Piper TTS        — natural synthesis for dynamic content / missing WAVs.
  3. espeak-ng        — emergency fallback; always available on Pi OS.

Public API:
  speak_key(key, priority, language=None)      — fixed vocabulary via key registry
  speak_text(text, priority, language=None)    — dynamic content via Piper/espeak-ng
  speak_sequence(keys, priority, language=None)— play multiple keys as one unit
  speak(text, priority, language=None)         — backward-compat alias for speak_text
  flush()                                      — drain queue, interrupt playback

If language=None the current language is read from the Settings instance.
"""
import logging
import queue
import subprocess
import threading
import time
from typing import Dict, List, Optional

try:
    from .. import config
except ImportError:
    import config  # type: ignore[no-redef]

from hardware.audio_keys import AUDIO_KEYS, get_canonical_text
from hardware.wav_player import WavPlayer
from hardware.piper_tts import PiperTTS

logger = logging.getLogger("clarifeye.hardware.audio")


class AudioManager:
    """
    Priority-queue based audio manager with three fallback backends.

    Queue item format: (priority: int, timestamp: float, payload: dict)

    Payload keys:
      type     — "key" | "text" | "sequence"
      key      — audio registry key (type="key")
      text     — free-form text (type="text")
      keys     — list of registry keys (type="sequence")
      language — "en" | "bg" | None (None → read from Settings)
      cooldown — seconds (None disables duplicate suppression)
    """

    def __init__(self, settings=None) -> None:
        self._settings = settings

        self._queue: queue.PriorityQueue = queue.PriorityQueue()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._kill_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        self._current_process: Optional[subprocess.Popen] = None

        # Cooldown tracking: cooldown_key → time.time() of last utterance.
        self._recent_messages: Dict[str, float] = {}
        # Priority of the item currently being played (None when idle).
        self._current_priority: Optional[int] = None

        self._wav_player = WavPlayer(config.AUDIO_DIR)
        self._piper = PiperTTS(config.VOICES_DIR)

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return
        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._speak_worker,
            daemon=True,
            name="clarifeye-audio",
        )
        self._worker_thread.start()
        logger.info("AudioManager started.")

    def stop(self) -> None:
        self._stop_event.set()
        self._kill_event.set()
        with self._lock:
            if (
                self._current_process is not None
                and self._current_process.poll() is None
            ):
                self._current_process.kill()
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=2.0)
        logger.info("AudioManager stopped.")

    def speak_key(
        self,
        key: str,
        priority: int = config.AUDIO_PRIORITY_NORMAL,
        language: Optional[str] = None,
        cooldown: Optional[float] = config.NOTIFICATION_COOLDOWN_SEC,
    ) -> None:
        """Enqueue a fixed-vocabulary key for WAV playback (Piper/espeak-ng fallback)."""
        payload = {"type": "key", "key": key, "language": language, "cooldown": cooldown}
        self._enqueue(priority, payload)

    def speak_text(
        self,
        text: str,
        priority: int = config.AUDIO_PRIORITY_NORMAL,
        language: Optional[str] = None,
        cooldown: Optional[float] = config.NOTIFICATION_COOLDOWN_SEC,
    ) -> None:
        """Enqueue free-form text for Piper TTS (espeak-ng fallback)."""
        payload = {"type": "text", "text": text, "language": language, "cooldown": cooldown}
        self._enqueue(priority, payload)

    def speak_sequence(
        self,
        keys: List[str],
        priority: int = config.AUDIO_PRIORITY_NORMAL,
        language: Optional[str] = None,
        cooldown: Optional[float] = config.NOTIFICATION_COOLDOWN_SEC,
    ) -> None:
        """Enqueue a sequence of keys played back-to-back with no gap."""
        payload = {"type": "sequence", "keys": keys, "language": language, "cooldown": cooldown}
        self._enqueue(priority, payload)

    def speak(
        self,
        text: str,
        priority: int = config.AUDIO_PRIORITY_NORMAL,
        language: Optional[str] = None,
        cooldown: Optional[float] = config.NOTIFICATION_COOLDOWN_SEC,
    ) -> None:
        # Deprecated: use speak_key() for fixed vocabulary or speak_text() for dynamic content.
        self.speak_text(text, priority, language, cooldown)

    def flush(self) -> None:
        """
        Drain the pending queue, interrupt current playback, and clear cooldown
        so the next mode's first announcement plays immediately.
        """
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

        self._kill_event.set()

        with self._lock:
            if (
                self._current_process is not None
                and self._current_process.poll() is None
            ):
                self._current_process.kill()
                self._current_process = None
            self._recent_messages.clear()

        logger.debug("AudioManager flushed.")

    def is_speaking(self) -> bool:
        with self._lock:
            return (
                self._current_process is not None
                and self._current_process.poll() is None
            )

    # ── Internal enqueue ──────────────────────────────────────────────────────

    def _enqueue(self, priority: int, payload: dict) -> None:
        self._queue.put((priority, time.time(), payload))

        with self._lock:
            current = self._current_priority
            # Interrupt if the new item has strictly higher priority (lower number)
            # than what is currently playing, or unconditionally for CRITICAL.
            should_interrupt = (
                (current is not None and priority < current)
                or priority == config.AUDIO_PRIORITY_CRITICAL
            )
            if should_interrupt:
                self._kill_event.set()
                if (
                    self._current_process is not None
                    and self._current_process.poll() is None
                ):
                    self._current_process.kill()
                    logger.debug(
                        "Interrupted priority %s audio for incoming priority %d.",
                        current,
                        priority,
                    )

    # ── Worker thread ─────────────────────────────────────────────────────────

    def _speak_worker(self) -> None:
        while not self._stop_event.is_set():
            try:
                priority, timestamp, payload = self._queue.get(
                    timeout=config.THREAD_QUEUE_TIMEOUT_SEC
                )
            except queue.Empty:
                continue

            lang = payload.get("language") or (
                self._settings.get_language() if self._settings else "en"
            )

            # ── Cooldown check ─────────────────────────────────────────────────
            cooldown_sec = payload.get("cooldown", config.NOTIFICATION_COOLDOWN_SEC)
            if cooldown_sec is not None:
                cooldown_key = self._cooldown_key(payload)
                now = time.time()
                with self._lock:
                    last_t = self._recent_messages.get(cooldown_key, 0.0)
                    if now - last_t < cooldown_sec:
                        logger.debug("Cooldown suppressed: %r", cooldown_key)
                        continue
                    self._recent_messages[cooldown_key] = now
                    self._recent_messages = {
                        k: v for k, v in self._recent_messages.items()
                        if now - v < config.NOTIFICATION_COOLDOWN_SEC
                    }

            # ── Dispatch ───────────────────────────────────────────────────────
            self._kill_event.clear()
            ptype = payload["type"]

            with self._lock:
                self._current_priority = priority
            try:
                if ptype == "key":
                    self._dispatch_key(payload["key"], lang)
                elif ptype == "text":
                    self._dispatch_text(payload["text"], lang)
                elif ptype == "sequence":
                    self._dispatch_sequence(payload["keys"], lang)
            except Exception as exc:
                logger.error("Audio dispatch error (%s): %s", ptype, exc, exc_info=True)
            finally:
                with self._lock:
                    self._current_priority = None

    @staticmethod
    def _cooldown_key(payload: dict) -> str:
        if payload["type"] == "key":
            return f"key:{payload['key']}"
        if payload["type"] == "sequence":
            return f"seq:{','.join(payload['keys'])}"
        return f"text:{payload.get('text', '')}"

    # ── Dispatch methods ──────────────────────────────────────────────────────

    def _dispatch_key(self, key: str, lang: str) -> None:
        # 1. Try pre-recorded WAV.
        proc = self._wav_player.play(key, lang)
        if proc is not None:
            self._wait_process(proc)
            return

        # 2. Synthesize via Piper.
        try:
            canonical = get_canonical_text(key, lang)
        except (KeyError, ValueError) as exc:
            logger.error("Audio key lookup failed: %s", exc)
            return

        if canonical and self._piper.is_available(lang):
            if self._piper.speak(canonical, lang, self._kill_event):
                return

        # 3. Fall back to espeak-ng.
        if canonical:
            self._espeak(canonical, lang)

    def _dispatch_text(self, text: str, lang: str) -> None:
        # 1. Try Piper.
        if self._piper.is_available(lang):
            if self._piper.speak(text, lang, self._kill_event):
                return
        # 2. Fall back to espeak-ng.
        self._espeak(text, lang)

    def _dispatch_sequence(self, keys: List[str], lang: str) -> None:
        # Try WAV sequence if all files present.
        if all(self._wav_player.has(k, lang) for k in keys):
            self._wav_player.play_sequence(keys, lang, self._kill_event)
            return
        # Fall back: play each key individually.
        for key in keys:
            if self._kill_event.is_set():
                break
            self._dispatch_key(key, lang)

    # ── Playback helpers ──────────────────────────────────────────────────────

    def _wait_process(self, proc: subprocess.Popen) -> None:
        with self._lock:
            self._current_process = proc
        try:
            while proc.poll() is None:
                if self._kill_event.is_set():
                    proc.kill()
                    break
                self._kill_event.wait(timeout=0.02)
        finally:
            with self._lock:
                if self._current_process is proc:
                    self._current_process = None

    def _espeak(self, text: str, lang: str) -> None:
        if not text:
            return
        voice = config.TTS_VOICE_BG if lang == "bg" else config.TTS_VOICE_EN
        cmd = [config.TTS_ENGINE, "-v", voice, "-s", str(config.TTS_SPEED), text]
        process: Optional[subprocess.Popen] = None
        try:
            process = subprocess.Popen(cmd)
            with self._lock:
                self._current_process = process
            while process.poll() is None:
                if self._kill_event.is_set():
                    process.kill()
                    break
                self._kill_event.wait(timeout=0.02)
        except FileNotFoundError:
            logger.error(
                "espeak-ng not found. Install with: sudo apt install espeak-ng"
            )
        except Exception as exc:
            logger.error("TTS subprocess error: %s", exc)
        finally:
            with self._lock:
                if self._current_process is process:
                    self._current_process = None
