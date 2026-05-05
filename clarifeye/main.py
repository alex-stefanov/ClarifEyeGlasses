"""
ClarifEye — Main Orchestrator
Raspberry Pi 4 wearable vision-assistance system for visually impaired people.

Thread model
~~~~~~~~~~~~
* **Camera thread**     — captures frames from the camera and drops them into
                          a bounded queue (oldest frame discarded when full).
* **Sensor thread**     — reads left/right ultrasonic and centre ToF sensors
                          in a tight 30 ms loop and stores the latest readings
                          in a shared dict.
* **Processing thread** — pulls frames from the queue, selects the AI pipeline
                          for the current mode, and drives the audio output.

All hardware calls are wrapped in try/except so a failing sensor never
crashes the system.  Every blocking queue/sleep operation uses a timeout so
threads can respond to the ``running`` event within one sleep cycle.

Run with: python main.py
"""
import logging
import logging.handlers
import os
import queue
import signal
import sys
import threading
import time
from typing import Dict, List, Optional

import numpy as np

# Ensure project root is on the path regardless of working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config

from hardware.audio import AudioManager
from hardware.buttons import ButtonHandler
from hardware.camera import CameraModule
from hardware.tof_sensor import ToFSensor
from hardware.ultrasonic import DualUltrasonic

from ai.color_verifier import ColorVerifier
from ai.currency_recognizer import CurrencyRecognizer
from ai.low_light_enhancer import LowLightEnhancer
from ai.object_detector import ObjectDetector
from ai.scene_describer import SceneDescriber
from ai.text_reader import TextReader
from ai.traffic_light_detector import TrafficLightDetector
from ai.translator import Translator

from core.action_dispatcher import ActionDispatcher
from core.mode_manager import ModeManager
from core.priority_engine import PriorityEngine
from core.sensor_fusion import SensorFusion
from core.settings import Settings

logger = logging.getLogger("clarifeye.main")


# ─── ClarifEyeSystem ─────────────────────────────────────────────────────────

class ClarifEyeSystem:
    """
    Top-level orchestrator that owns all hardware and AI modules, manages the
    three worker threads, and handles graceful startup and shutdown.
    """

    # ── Initialisation ────────────────────────────────────────────────────────

    def __init__(self) -> None:
        """
        Initialise all subsystems in dependency order.

        Failures in individual hardware components are caught and logged;
        the system continues with degraded capability rather than aborting.
        """
        self._stopping = threading.Event()

        # ── 1. Logging ─────────────────────────────────────────────────────────
        self._setup_logging()
        logger.info("=" * 60)
        logger.info("ClarifEye initialising …")

        # ── 2. Settings (before audio — AudioManager reads language from it) ───
        self._settings = Settings(config.SETTINGS_PATH)

        # ── 3. Audio (must be running before any mode announcements) ───────────
        self._audio = AudioManager(settings=self._settings)
        self._audio.start()

        # ── 4. Camera ──────────────────────────────────────────────────────────
        self._camera = CameraModule()

        # ── 5. Dual ultrasonic ─────────────────────────────────────────────────
        self._ultrasonic = DualUltrasonic()

        # ── 6. ToF sensor ──────────────────────────────────────────────────────
        self._tof = ToFSensor()

        # ── 7. Mode manager ────────────────────────────────────────────────────
        self._mode_manager = ModeManager(self._audio)

        # ── 8. Action dispatcher ──────────────────────────────────────────────
        self._action_dispatcher = ActionDispatcher(
            self._mode_manager, self._audio, self._settings
        )

        # ── 9. Button handler ─────────────────────────────────────────────────
        self._buttons = ButtonHandler(
            on_next_mode=self._mode_manager.next_mode,
            on_language_toggle=self._on_language_toggle,
            on_action=self._action_dispatcher.on_action_pressed,
        )

        # ── 10-12. Detection models ───────────────────────────────────────────
        self._tl_detector = TrafficLightDetector()
        self._obj_detector = ObjectDetector()
        self._color_verifier = ColorVerifier()

        # ── 13-15. Text / translation / enhancement ───────────────────────────
        self._text_reader = TextReader()
        self._translator = Translator(self._settings)
        self._low_light = LowLightEnhancer()

        # ── 16. Currency recognizer ───────────────────────────────────────────
        try:
            self._currency_recognizer: Optional[CurrencyRecognizer] = CurrencyRecognizer(
                os.path.join(config.DATA_DIR, "banknotes")
            )
        except Exception as exc:
            logger.warning("CurrencyRecognizer init failed: %s — currency mode disabled.", exc)
            self._currency_recognizer = None

        # ── 17. Scene describer (SmolVLM) ─────────────────────────────────────
        try:
            self._scene_describer: Optional[SceneDescriber] = SceneDescriber(
                config.SCENE_MODEL_PATH,
                config.SCENE_TOKENIZER_PATH,
            )
        except Exception as exc:
            logger.warning("SceneDescriber init failed: %s — scene mode disabled.", exc)
            self._scene_describer = None

        # ── Register action handlers ───────────────────────────────────────────
        self._action_dispatcher.register_handler(
            config.Mode.TEXT_READING, self._on_text_reading_action
        )
        self._action_dispatcher.register_handler(
            config.Mode.CURRENCY, self._on_currency_action
        )
        self._action_dispatcher.register_handler(
            config.Mode.SCENE, self._on_scene_action
        )

        # ── 16-17. Fusion + priority ──────────────────────────────────────────
        self._sensor_fusion = SensorFusion()
        self._priority_engine = PriorityEngine(self._audio)

        # ── 16. Threading primitives ───────────────────────────────────────────
        self._frame_queue: queue.Queue[np.ndarray] = queue.Queue(
            maxsize=config.FRAME_QUEUE_MAXSIZE
        )
        self._running = threading.Event()

        # Latest ultrasonic / ToF readings; written by sensor thread every 30 ms.
        self._sensor_data: Dict = {
            "left_cm": None,
            "right_cm": None,
            "tof_cm": None,
            "timestamp": 0.0,
        }
        self._sensor_lock = threading.Lock()

        self._threads: List[threading.Thread] = []

        # Register OS-level shutdown signals.
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        logger.info("All subsystems initialised.")

    # ── Startup / Shutdown ────────────────────────────────────────────────────

    def start(self) -> None:
        """
        Start all worker threads and block until a shutdown signal is received.

        Announces startup and the initial mode in Bulgarian, starts the three
        daemon threads, then spins in a 100 ms sleep loop until
        ``_running`` is cleared by :meth:`stop` or a signal handler.
        """
        self._audio.speak_key("system_starting", config.AUDIO_PRIORITY_HIGH)

        try:
            self._camera.start()
        except Exception as exc:
            logger.error("Camera start failed: %s", exc)

        self._running.set()

        for name, target in (
            ("camera",     self._camera_thread),
            ("processing", self._processing_thread),
            ("sensor",     self._sensor_thread),
        ):
            t = threading.Thread(
                target=target, name=f"clarifeye-{name}", daemon=True
            )
            t.start()
            self._threads.append(t)
            logger.info("Thread '%s' started.", name)

        # Brief pause so the startup announcement finishes before mode name.
        time.sleep(0.3)
        self._mode_manager.announce_current_mode(config.AUDIO_PRIORITY_NORMAL)

        logger.info("ClarifEye running.  Ctrl+C or SIGTERM to stop.")

        # Block main thread; shutdown happens via _running.clear() elsewhere.
        while self._running.is_set():
            time.sleep(0.1)

    def stop(self) -> None:
        """
        Graceful shutdown: announce stopping, drain worker threads, release
        all hardware.

        Idempotent — safe to call from a signal handler *and* a ``finally``
        block without double-execution.
        """
        if self._stopping.is_set():
            return
        self._stopping.set()

        logger.info("ClarifEye shutting down …")
        self._running.clear()

        # Announce before stopping the audio worker so the message is heard.
        try:
            self._audio.speak_key("system_stopping", config.AUDIO_PRIORITY_HIGH)
            time.sleep(2.0)   # Allow TTS time to play the announcement.
        except Exception as exc:
            logger.error("Shutdown announcement error: %s", exc)

        # Wait for worker threads to finish their current iteration.
        for t in self._threads:
            t.join(timeout=5.0)
            if t.is_alive():
                logger.warning("Thread '%s' did not stop within 5 s.", t.name)

        # Release hardware in reverse initialisation order.
        for label, fn in (
            ("audio",      self._audio.stop),
            ("buttons",    self._buttons.cleanup),
            ("tof_sensor", self._tof.cleanup),
            ("ultrasonic", self._ultrasonic.cleanup),
            ("camera",     self._camera.stop),
        ):
            try:
                fn()
            except Exception as exc:
                logger.error("Cleanup of '%s' failed: %s", label, exc)

        logger.info("ClarifEye shutdown complete.")
        logger.info("=" * 60)

    # ── Button callbacks ──────────────────────────────────────────────────────

    def _on_language_toggle(self) -> None:
        """
        Toggle the app language and announce the new selection.

        Settings.toggle_language() atomically flips "en"↔"bg" and returns the
        new value.  speak_key with language=None then reads from Settings, so
        it automatically picks the correct variant of "language_switched"
        (English announces in English, Bulgarian announces in Bulgarian).
        """
        new_lang = self._settings.toggle_language()
        self._audio.flush()
        self._audio.speak_key("language_switched", config.AUDIO_PRIORITY_HIGH)
        logger.info("Language toggled → %s.", new_lang)

    # ── Signal handler ────────────────────────────────────────────────────────

    def _handle_signal(self, signum: int, _frame: object) -> None:
        """
        Handle SIGINT / SIGTERM from the OS.

        Clears ``_running`` so the main thread's sleep loop in :meth:`start`
        exits cleanly; the ``finally`` block in ``__main__`` then calls
        :meth:`stop`.
        """
        logger.info("Signal %d received — initiating shutdown.", signum)
        self._running.clear()

    # ── Worker threads ────────────────────────────────────────────────────────

    def _camera_thread(self) -> None:
        """
        Capture loop: grab frames from the camera and push to ``_frame_queue``.

        The queue is bounded (``config.FRAME_QUEUE_MAXSIZE``).  When it is
        full the oldest frame is discarded so the processing thread always
        operates on the most recent image.  Capture failures trigger a 10 ms
        pause before retrying.
        """
        logger.debug("Camera thread started.")

        while self._running.is_set():
            try:
                frame = self._camera.capture_frame()
            except Exception as exc:
                logger.error("Camera capture error: %s", exc)
                time.sleep(0.01)
                continue

            if frame is None:
                time.sleep(0.01)
                continue

            # Drop oldest frame to make room for the new one.
            if self._frame_queue.full():
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    pass

            try:
                self._frame_queue.put_nowait(frame)
            except queue.Full:
                pass   # Harmless race: processing thread cleared the slot first.

        logger.debug("Camera thread exiting.")

    def _sensor_thread(self) -> None:
        """
        Sensor polling loop: read ultrasonic and ToF every 30 ms.

        HC-SR04 sensors need ~25 ms between consecutive trigger pulses to
        avoid acoustic echo from the previous measurement, so the 30 ms sleep
        satisfies the hardware requirement while keeping readings fresh.

        Failures on individual sensors are caught and logged; the other
        sensor continues to function.
        """
        logger.debug("Sensor thread started.")

        while self._running.is_set():
            left_cm: Optional[float] = None
            right_cm: Optional[float] = None
            tof_cm: Optional[float] = None

            try:
                left_cm, right_cm = self._ultrasonic.measure_both()
            except Exception as exc:
                logger.error("Ultrasonic read error: %s", exc)

            try:
                tof_cm = self._tof.measure_distance_cm()
            except Exception as exc:
                logger.error("ToF read error: %s", exc)

            with self._sensor_lock:
                self._sensor_data = {
                    "left_cm": left_cm,
                    "right_cm": right_cm,
                    "tof_cm": tof_cm,
                    "timestamp": time.monotonic(),
                }

            time.sleep(0.030)

        logger.debug("Sensor thread exiting.")

    def _processing_thread(self) -> None:
        """
        Main AI loop: pull frames, dispatch to the correct mode pipeline.

        Metrics
        -------
        * Processing FPS logged at INFO every 30 s.
        * Process memory logged at INFO every 60 s.
        * ``dt`` (seconds since last processed frame) is computed locally and
          passed to the sensor-fusion Kalman filters.
        """
        logger.debug("Processing thread started.")

        _last_frame_time: float = time.monotonic()
        _frame_count: int = 0
        _fps_log_time: float = time.monotonic()
        _mem_log_time: float = time.monotonic()

        while self._running.is_set():

            # ── Dequeue with timeout so we can react to shutdown ───────────────
            try:
                frame = self._frame_queue.get(
                    timeout=config.THREAD_QUEUE_TIMEOUT_SEC
                )
            except queue.Empty:
                continue

            try:
                now = time.monotonic()
                dt = max(now - _last_frame_time, 1e-3)
                _last_frame_time = now
                _frame_count += 1

                # ── FPS log every 30 s ─────────────────────────────────────────
                if now - _fps_log_time >= 30.0:
                    fps = _frame_count / (now - _fps_log_time)
                    logger.info(
                        "Processing FPS: %.1f  (frames=%d)", fps, _frame_count
                    )
                    _frame_count = 0
                    _fps_log_time = now

                # ── Memory log every 60 s ──────────────────────────────────────
                if now - _mem_log_time >= 60.0:
                    self._log_memory()
                    _mem_log_time = now

                mode = self._mode_manager.get_current_mode()

                with self._sensor_lock:
                    sensor_data = dict(self._sensor_data)

                self._process_frame(frame, mode, sensor_data, dt)

            except Exception as exc:
                logger.error(
                    "Unhandled error in processing thread: %s", exc, exc_info=True
                )
            finally:
                del frame   # Release ndarray reference so GC can reclaim memory.

        logger.debug("Processing thread exiting.")

    # ── Frame processing dispatcher ───────────────────────────────────────────

    def _process_frame(
        self,
        frame: np.ndarray,
        mode: "config.Mode",  # type: ignore[name-defined]
        sensor_data: Dict,
        dt: float,
    ) -> None:
        """Route *frame* to the mode-specific processing pipeline."""
        if mode == config.Mode.TRAFFIC_LIGHT:
            self._process_traffic_light(frame)
        elif mode == config.Mode.NAVIGATION:
            self._process_navigation(frame, sensor_data, dt)
        elif mode == config.Mode.TEXT_READING:
            pass  # action-triggered only
        elif mode == config.Mode.CURRENCY:
            pass  # action-triggered only
        elif mode == config.Mode.SCENE:
            pass  # action-triggered only

    # ── Mode pipelines ────────────────────────────────────────────────────────

    def _process_traffic_light(self, frame: np.ndarray) -> None:
        """
        TRAFFIC_LIGHT mode pipeline.

        Steps:

        1. Auto-enhance if low-light.
        2. Detect traffic lights with the TFLite YOLOv8n model.
        3. Confirm each colour with the HSV pixel verifier; update class_name
           if the verifier disagrees with the neural network.
        4. Queue Bulgarian colour announcement via AudioManager.
           AudioManager's own duplicate-suppression provides the 2-second
           cooldown so the same colour is not announced every frame.
        """
        try:
            frame, _enhanced = self._low_light.auto_enhance(frame)
        except Exception as exc:
            logger.error("Low-light enhancement failed (TL mode): %s", exc)

        try:
            detections = self._tl_detector.detect(frame)
        except Exception as exc:
            logger.error("Traffic-light detection failed: %s", exc)
            return

        for det in detections:
            try:
                verified = self._color_verifier.verify_traffic_light_color(
                    frame, det.bbox
                )
                if verified is not None and verified != det.class_name:
                    logger.debug(
                        "HSV verifier corrected %s → %s  (conf=%.0f%%)",
                        det.class_name, verified, det.confidence * 100,
                    )
                    det.class_name = verified
            except Exception as exc:
                logger.error("Color verification error: %s", exc)

            tl_key = f"tl_{det.class_name}"
            if tl_key not in ("tl_red", "tl_yellow", "tl_green"):
                continue

            priority = (
                config.AUDIO_PRIORITY_HIGH
                if det.class_name in ("red", "yellow")
                else config.AUDIO_PRIORITY_NORMAL
            )
            self._audio.speak_key(tl_key, priority)
            logger.debug(
                "Traffic light → %s  conf=%.0f%%  priority=%d",
                det.class_name, det.confidence * 100, priority,
            )

    def _process_navigation(
        self,
        frame: np.ndarray,
        sensor_data: Dict,
        dt: float,
    ) -> None:
        """
        NAVIGATION mode pipeline.

        Steps:

        1. Auto-enhance if low-light.
        2. Run the shared navigation sub-pipeline (object detection → sensor
           fusion → priority engine).
        """
        try:
            frame, _enhanced = self._low_light.auto_enhance(frame)
        except Exception as exc:
            logger.error("Low-light enhancement failed (Nav mode): %s", exc)

        self._run_navigation_pipeline(frame, sensor_data, dt)

    def _on_text_reading_action(self, duration_ms: float) -> None:
        """
        Called by ActionDispatcher when ACTION is pressed in TEXT_READING mode.
        Runs in a daemon thread — blocks are fine here.
        """
        self._audio.speak_key("processing", config.AUDIO_PRIORITY_NORMAL)

        try:
            frame = self._camera.capture_frame()
        except Exception as exc:
            logger.error("Camera capture failed in text reading action: %s", exc)
            frame = None

        if frame is None:
            self._audio.speak_key("no_text_found", config.AUDIO_PRIORITY_NORMAL)
            return

        try:
            text_dets = self._text_reader.read_text(frame)
        except Exception as exc:
            logger.error("OCR failed in text reading action: %s", exc)
            self._audio.speak_key("no_text_found", config.AUDIO_PRIORITY_NORMAL)
            return

        if not text_dets:
            self._audio.speak_key("no_text_found", config.AUDIO_PRIORITY_NORMAL)
            return

        truncated = len(text_dets) > config.TEXT_READING_MAX_BLOCKS
        text_dets = text_dets[:config.TEXT_READING_MAX_BLOCKS]

        for td in text_dets:
            try:
                spoken = self._translator.translate_to_user_language(
                    td.text, td.language or "en"
                )
                self._audio.speak_text(spoken, config.AUDIO_PRIORITY_NORMAL)
                logger.debug(
                    "OCR spoke: %r (from %r, detected_lang=%s)",
                    spoken, td.text, td.language,
                )
            except Exception as exc:
                logger.error("Speak failed for %r: %s", td.text, exc)

        if truncated:
            self._audio.speak_key("more_text_truncated", config.AUDIO_PRIORITY_LOW)

    def _on_currency_action(self, duration_ms: float) -> None:
        """
        Called by ActionDispatcher when ACTION is pressed in CURRENCY mode.
        Runs in a daemon thread — captures a frame and runs ORB recognition.
        """
        self._audio.speak_key("processing", config.AUDIO_PRIORITY_NORMAL)

        if self._currency_recognizer is None:
            self._audio.speak_key("no_currency_found", config.AUDIO_PRIORITY_NORMAL)
            return

        try:
            frame = self._camera.capture_frame()
        except Exception as exc:
            logger.error("Camera capture failed in currency action: %s", exc)
            frame = None

        if frame is None:
            self._audio.speak_key("no_currency_found", config.AUDIO_PRIORITY_NORMAL)
            return

        try:
            result = self._currency_recognizer.recognize(frame)
        except Exception as exc:
            logger.error("Currency recognition failed: %s", exc)
            self._audio.speak_key("no_currency_found", config.AUDIO_PRIORITY_NORMAL)
            return

        if result is None:
            self._audio.speak_key("no_currency_found", config.AUDIO_PRIORITY_NORMAL)
            return

        currency_code, denomination, confidence = result
        try:
            audio_key = self._currency_recognizer.get_audio_key(currency_code, denomination)
        except ValueError as exc:
            logger.error("get_audio_key failed: %s", exc)
            self._audio.speak_key("no_currency_found", config.AUDIO_PRIORITY_NORMAL)
            return

        logger.info(
            "Currency recognised: %s %s (confidence=%.2f)", currency_code, denomination, confidence
        )
        self._audio.speak_key(audio_key, config.AUDIO_PRIORITY_NORMAL)

    def _on_scene_action(self, duration_ms: float) -> None:
        """
        Called by ActionDispatcher when ACTION is pressed in SCENE mode.
        Runs in a daemon thread — captures a frame, runs SmolVLM, speaks result.
        Latency on Pi 4: 5–15 s. The "processing" cue buys time before the result.
        """
        self._audio.speak_key("processing", config.AUDIO_PRIORITY_NORMAL)

        try:
            frame = self._camera.capture_frame()
        except Exception as exc:
            logger.error("Camera capture failed in scene action: %s", exc)
            frame = None

        if frame is None:
            # Rare edge case — hardcoded English via speak_text is acceptable here.
            # Future improvement: add "camera_error" audio key.
            self._audio.speak_text("Camera error", config.AUDIO_PRIORITY_NORMAL, "en")
            return

        if self._scene_describer is None:
            self._audio.speak_text("Scene description failed", config.AUDIO_PRIORITY_NORMAL, "en")
            return

        try:
            description_en = self._scene_describer.describe(frame, max_words=config.SCENE_MAX_WORDS)
        except Exception as exc:
            logger.error("Scene description raised unexpectedly: %s", exc)
            description_en = None

        if description_en is None:
            # Future improvement: add "scene_failed" audio key.
            self._audio.speak_text("Scene description failed", config.AUDIO_PRIORITY_NORMAL, "en")
            return

        logger.info("Scene description: %r", description_en)

        user_lang = self._settings.get_language()
        if user_lang == "bg":
            translated = self._translator.translate(description_en, "en", "bg")
            spoken = translated if translated else description_en
            self._audio.speak_text(spoken, config.AUDIO_PRIORITY_NORMAL, user_lang)
        else:
            self._audio.speak_text(description_en, config.AUDIO_PRIORITY_NORMAL, "en")

    # ── Shared sub-pipeline ───────────────────────────────────────────────────

    def _run_navigation_pipeline(
        self,
        frame: np.ndarray,
        sensor_data: Dict,
        dt: float,
    ) -> None:
        """
        Navigation sub-pipeline used by NAVIGATION mode.

        Runs object detection → Kalman sensor fusion → priority engine.
        Each stage is individually guarded; a single failure does not abort
        the remaining stages.
        """
        try:
            detections = self._obj_detector.detect(frame)
        except Exception as exc:
            logger.error("Object detection failed: %s", exc)
            return

        if not detections:
            return

        frame_h, frame_w = frame.shape[:2]

        try:
            detections = self._sensor_fusion.fuse(
                detections,
                sensor_data.get("left_cm"),
                sensor_data.get("right_cm"),
                sensor_data.get("tof_cm"),
                frame_w,
                frame_h,
                dt,
            )
        except Exception as exc:
            logger.error("Sensor fusion failed: %s", exc)
            # Continue with unfiltered camera-distance estimates.

        try:
            self._priority_engine.process_detections(detections)
        except Exception as exc:
            logger.error("Priority engine failed: %s", exc)

    # ── Static helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _setup_logging() -> None:
        """
        Configure the root logger with a console handler and a UTF-8
        rotating file handler.  All parameters come from ``config``.
        """
        root = logging.getLogger()
        root.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))

        formatter = logging.Formatter(config.LOG_FORMAT)

        # Console — always available.
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        root.addHandler(ch)

        # Rotating file — created only if the directory is accessible.
        try:
            os.makedirs(config.LOGS_DIR, exist_ok=True)
            fh = logging.handlers.RotatingFileHandler(
                config.LOG_FILE,
                maxBytes=config.LOG_MAX_BYTES,
                backupCount=config.LOG_BACKUP_COUNT,
                encoding="utf-8",
            )
            fh.setFormatter(formatter)
            root.addHandler(fh)
        except OSError as exc:
            logging.warning("Could not create rotating log file: %s", exc)

    @staticmethod
    def _log_memory() -> None:
        """
        Log current process RSS at INFO level.

        Uses ``resource.getrusage`` on Linux (Raspberry Pi target) with
        ``psutil`` as a fallback for Windows / macOS development machines.
        On Linux ``ru_maxrss`` is in kilobytes.
        """
        try:
            import resource  # Unix only
            mem_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            logger.info(
                "Memory usage: %d KB  (%.1f MB)", mem_kb, mem_kb / 1024.0
            )
            return
        except ImportError:
            pass

        try:
            import psutil  # type: ignore[import]
            rss = psutil.Process().memory_info().rss
            logger.info(
                "Memory usage: %d KB  (%.1f MB)", rss // 1024, rss / 1_048_576
            )
        except Exception:
            pass  # Memory logging is best-effort; never crash over it.


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    system = ClarifEyeSystem()
    try:
        system.start()
    except KeyboardInterrupt:
        pass
    finally:
        system.stop()
