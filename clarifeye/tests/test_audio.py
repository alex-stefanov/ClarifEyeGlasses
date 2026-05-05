"""
tests/test_audio.py
===================
Tests for ``hardware/audio.py`` (AudioManager).

``subprocess.Popen`` is patched throughout so no ``espeak-ng`` binary is
required.  A ``threading.Event`` inside the mock's ``wait()`` side-effect
synchronises assertions with the background worker thread, avoiding
arbitrary ``time.sleep()`` calls.
"""
import os
import sys
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from hardware.audio import AudioManager


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_mock_process(spoken_event: threading.Event) -> MagicMock:
    """
    Build a mock ``subprocess.Popen`` process whose ``wait()`` sets
    *spoken_event*, allowing tests to know when the worker has finished
    processing a message.

    ``poll()`` returns ``0`` (process finished) so the manager never tries
    to kill an "already-running" process unexpectedly.
    """
    mock_proc = MagicMock()
    mock_proc.poll.return_value = 0  # Already finished.

    def _wait():
        spoken_event.set()
        return 0

    mock_proc.wait = _wait
    return mock_proc


# ════════════════════════════════════════════════════════════════════════════════
# Lifecycle
# ════════════════════════════════════════════════════════════════════════════════

class TestAudioManagerLifecycle:
    """Tests for start / stop / idempotency."""

    def test_initialises_without_crash(self):
        """``AudioManager()`` must not raise."""
        audio = AudioManager()
        assert audio is not None

    def test_start_launches_worker_thread(self):
        """After ``start()`` the worker thread must be alive."""
        audio = AudioManager()
        audio.start()
        try:
            assert audio._worker_thread is not None
            assert audio._worker_thread.is_alive()
        finally:
            audio.stop()

    def test_start_is_idempotent(self):
        """Calling ``start()`` twice must not create a second thread."""
        audio = AudioManager()
        audio.start()
        first_thread = audio._worker_thread
        audio.start()   # Second call — must be a no-op.
        try:
            assert audio._worker_thread is first_thread
        finally:
            audio.stop()

    def test_stop_terminates_worker(self):
        """After ``stop()`` the worker thread must no longer be alive."""
        audio = AudioManager()
        audio.start()
        audio.stop()
        if audio._worker_thread is not None:
            audio._worker_thread.join(timeout=2.0)
            assert not audio._worker_thread.is_alive()

    def test_stop_without_start_does_not_crash(self):
        """``stop()`` on an unstarted manager must not raise."""
        audio = AudioManager()
        audio.stop()   # Must be safe.

    def test_stop_kills_active_process(self):
        """
        ``stop()`` must call ``kill()`` on any currently running espeak-ng
        process to prevent orphaned subprocesses.
        """
        audio = AudioManager()
        running_proc = MagicMock()
        running_proc.poll.return_value = None   # Still running.
        with audio._lock:
            audio._current_process = running_proc
        audio.stop()
        running_proc.kill.assert_called_once()


# ════════════════════════════════════════════════════════════════════════════════
# Queue and is_speaking
# ════════════════════════════════════════════════════════════════════════════════

class TestAudioManagerQueue:
    """Tests for speak(), priority ordering, and is_speaking()."""

    def test_speak_enqueues_one_item(self):
        """``speak()`` must put exactly one item in the priority queue."""
        audio = AudioManager()
        assert audio._queue.empty()
        audio.speak("тест", config.AUDIO_PRIORITY_NORMAL, "bg")
        assert audio._queue.qsize() == 1

    def test_speak_enqueues_correct_fields(self):
        """
        The queued tuple must be ``(priority, timestamp, payload)``
        where payload carries the text and language.
        """
        audio = AudioManager()
        before = time.time()
        audio.speak("hello", config.AUDIO_PRIORITY_HIGH, "en")
        after = time.time()

        priority, timestamp, payload = audio._queue.get_nowait()
        assert priority == config.AUDIO_PRIORITY_HIGH
        assert payload["text"] == "hello"
        assert payload["language"] == "en"
        assert before <= timestamp <= after

    def test_priority_queue_ordering(self):
        """
        The PriorityQueue must dequeue messages in ascending priority order
        (lower integer = higher urgency), regardless of insertion order.
        """
        audio = AudioManager()
        # Insert in reverse urgency order.
        audio.speak("low",      config.AUDIO_PRIORITY_LOW,      "bg")
        audio.speak("normal",   config.AUDIO_PRIORITY_NORMAL,   "bg")
        audio.speak("high",     config.AUDIO_PRIORITY_HIGH,     "bg")
        audio.speak("critical", config.AUDIO_PRIORITY_CRITICAL, "bg")

        priorities = []
        while not audio._queue.empty():
            priority, _, _ = audio._queue.get_nowait()
            priorities.append(priority)

        assert priorities == sorted(priorities), (
            f"Priority queue out of order: {priorities}"
        )

    def test_is_speaking_false_initially(self):
        """``is_speaking()`` must return ``False`` before any TTS starts."""
        audio = AudioManager()
        assert audio.is_speaking() is False

    def test_is_speaking_true_when_process_alive(self):
        """
        ``is_speaking()`` must return ``True`` when ``_current_process`` is a
        running mock process (``poll()`` returns ``None``).
        """
        audio = AudioManager()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None   # Simulate running process.
        with audio._lock:
            audio._current_process = mock_proc
        assert audio.is_speaking() is True

    def test_is_speaking_false_after_process_ends(self):
        """
        ``is_speaking()`` must return ``False`` when the process has exited
        (``poll()`` returns a non-``None`` exit code).
        """
        audio = AudioManager()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0   # Simulate finished process.
        with audio._lock:
            audio._current_process = mock_proc
        assert audio.is_speaking() is False


# ════════════════════════════════════════════════════════════════════════════════
# Worker thread behaviour (Popen mocked)
# ════════════════════════════════════════════════════════════════════════════════

class TestAudioManagerWorker:
    """Tests that exercise the background _speak_worker thread."""

    def test_worker_calls_espeak_with_correct_text(self):
        """
        The worker must invoke ``subprocess.Popen`` with the message text
        present in the command list.
        """
        spoken = threading.Event()
        mock_proc = _make_mock_process(spoken)

        with patch("hardware.audio.subprocess.Popen", return_value=mock_proc) as mock_popen:
            audio = AudioManager()
            audio.start()
            audio.speak("Автомобил напред", config.AUDIO_PRIORITY_NORMAL, "bg")
            spoken.wait(timeout=2.0)
            audio.stop()

        assert mock_popen.called, "subprocess.Popen was never called"
        cmd = mock_popen.call_args[0][0]   # First positional arg = command list.
        assert "Автомобил напред" in cmd

    def test_worker_uses_correct_voice_for_bulgarian(self):
        """
        Bulgarian messages must pass ``config.TTS_VOICE_BG`` to espeak-ng.
        """
        spoken = threading.Event()
        mock_proc = _make_mock_process(spoken)

        with patch("hardware.audio.subprocess.Popen", return_value=mock_proc) as mock_popen:
            audio = AudioManager()
            audio.start()
            audio.speak("тест", config.AUDIO_PRIORITY_NORMAL, "bg")
            spoken.wait(timeout=2.0)
            audio.stop()

        cmd = mock_popen.call_args[0][0]
        assert config.TTS_VOICE_BG in cmd

    def test_worker_uses_correct_voice_for_english(self):
        """
        English messages must pass ``config.TTS_VOICE_EN`` to espeak-ng.
        """
        spoken = threading.Event()
        mock_proc = _make_mock_process(spoken)

        with patch("hardware.audio.subprocess.Popen", return_value=mock_proc) as mock_popen:
            audio = AudioManager()
            audio.start()
            audio.speak("test", config.AUDIO_PRIORITY_NORMAL, "en")
            spoken.wait(timeout=2.0)
            audio.stop()

        cmd = mock_popen.call_args[0][0]
        assert config.TTS_VOICE_EN in cmd

    def test_worker_includes_tts_engine_in_command(self):
        """
        The espeak-ng command must start with ``config.TTS_ENGINE``.
        """
        spoken = threading.Event()
        mock_proc = _make_mock_process(spoken)

        with patch("hardware.audio.subprocess.Popen", return_value=mock_proc) as mock_popen:
            audio = AudioManager()
            audio.start()
            audio.speak("тест", config.AUDIO_PRIORITY_NORMAL, "bg")
            spoken.wait(timeout=2.0)
            audio.stop()

        cmd = mock_popen.call_args[0][0]
        assert cmd[0] == config.TTS_ENGINE

    def test_worker_handles_missing_espeak_gracefully(self):
        """
        If ``espeak-ng`` is not installed (``FileNotFoundError``) the worker
        must not crash; the thread must remain alive.
        """
        with patch(
            "hardware.audio.subprocess.Popen",
            side_effect=FileNotFoundError("espeak-ng not found"),
        ):
            audio = AudioManager()
            audio.start()
            audio.speak("тест", config.AUDIO_PRIORITY_NORMAL, "bg")
            time.sleep(0.3)   # Give worker time to catch and log the error.
            assert audio._worker_thread is not None
            assert audio._worker_thread.is_alive(), (
                "Worker thread must survive FileNotFoundError"
            )
            audio.stop()

    def test_cooldown_suppresses_duplicate_within_window(self):
        """
        The same text spoken twice within ``NOTIFICATION_COOLDOWN_SEC`` must
        result in only one espeak-ng invocation.
        """
        call_count = {"n": 0}
        spoken = threading.Event()

        def _wait_side():
            call_count["n"] += 1
            spoken.set()
            return 0

        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0
        mock_proc.wait = _wait_side

        with patch("hardware.audio.subprocess.Popen", return_value=mock_proc):
            audio = AudioManager()
            audio.start()
            audio.speak("Човек напред", config.AUDIO_PRIORITY_NORMAL, "bg")
            spoken.wait(timeout=2.0)
            # Second speak of the same text before cooldown expires.
            audio.speak("Човек напред", config.AUDIO_PRIORITY_NORMAL, "bg")
            time.sleep(0.3)   # Let worker process (or reject) the second message.
            audio.stop()

        assert call_count["n"] == 1, (
            f"Expected 1 espeak call (cooldown), got {call_count['n']}"
        )

    def test_cooldown_allows_same_text_after_expiry(self):
        """
        The same text spoken after the cooldown window has elapsed must be
        passed to espeak-ng a second time.
        """
        call_count = {"n": 0}
        first_spoken = threading.Event()
        second_spoken = threading.Event()

        def _wait_side():
            call_count["n"] += 1
            if call_count["n"] == 1:
                first_spoken.set()
            else:
                second_spoken.set()
            return 0

        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0
        mock_proc.wait = _wait_side

        with patch("hardware.audio.subprocess.Popen", return_value=mock_proc):
            audio = AudioManager()
            audio.start()
            audio.speak("Тест", config.AUDIO_PRIORITY_NORMAL, "bg")
            first_spoken.wait(timeout=2.0)

            # Artificially expire the cooldown entry.
            with audio._lock:
                audio._recent_messages["Тест"] = 0.0

            audio.speak("Тест", config.AUDIO_PRIORITY_NORMAL, "bg")
            second_spoken.wait(timeout=2.0)
            audio.stop()

        assert call_count["n"] == 2, (
            f"Expected 2 espeak calls after cooldown reset, got {call_count['n']}"
        )

    def test_different_texts_both_spoken(self):
        """
        Two different messages must each result in a separate espeak-ng call
        (cooldown is per-text, not global).
        """
        call_count = {"n": 0}
        second_spoken = threading.Event()

        def _wait_side():
            call_count["n"] += 1
            if call_count["n"] >= 2:
                second_spoken.set()
            return 0

        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0
        mock_proc.wait = _wait_side

        with patch("hardware.audio.subprocess.Popen", return_value=mock_proc):
            audio = AudioManager()
            audio.start()
            audio.speak("Човек вляво",   config.AUDIO_PRIORITY_NORMAL, "bg")
            audio.speak("Автомобил напред", config.AUDIO_PRIORITY_HIGH, "bg")
            second_spoken.wait(timeout=3.0)
            audio.stop()

        assert call_count["n"] == 2, (
            f"Expected 2 espeak calls for 2 different texts, got {call_count['n']}"
        )


# ════════════════════════════════════════════════════════════════════════════════
# Critical interrupt behaviour
# ════════════════════════════════════════════════════════════════════════════════

class TestAudioManagerCritical:
    """Tests for CRITICAL priority interrupt behaviour."""

    def test_critical_speak_kills_running_process(self):
        """
        Calling ``speak(CRITICAL)`` while a process is running must invoke
        ``process.kill()`` immediately (before the worker even dequeues it).
        """
        audio = AudioManager()
        running_proc = MagicMock()
        running_proc.poll.return_value = None   # Simulate running process.
        with audio._lock:
            audio._current_process = running_proc

        audio.speak("Внимание!", config.AUDIO_PRIORITY_CRITICAL, "bg")
        running_proc.kill.assert_called_once()

    def test_critical_speak_does_not_kill_if_no_process(self):
        """
        ``speak(CRITICAL)`` when no process is running must not raise and
        must enqueue the message correctly.
        """
        audio = AudioManager()
        audio.speak("Внимание!", config.AUDIO_PRIORITY_CRITICAL, "bg")
        assert not audio._queue.empty()
        priority, _, payload = audio._queue.get_nowait()
        assert priority == config.AUDIO_PRIORITY_CRITICAL
        assert payload["text"] == "Внимание!"

    def test_critical_speak_does_not_kill_finished_process(self):
        """
        ``speak(CRITICAL)`` must not call ``kill()`` on a process that has
        already finished (``poll()`` returns non-``None``).
        """
        audio = AudioManager()
        finished_proc = MagicMock()
        finished_proc.poll.return_value = 0   # Already done.
        with audio._lock:
            audio._current_process = finished_proc

        audio.speak("Внимание!", config.AUDIO_PRIORITY_CRITICAL, "bg")
        finished_proc.kill.assert_not_called()

    def test_critical_message_enqueued_at_priority_zero(self):
        """
        After ``speak(CRITICAL)`` the dequeued priority must be
        ``AUDIO_PRIORITY_CRITICAL`` (0).
        """
        audio = AudioManager()
        audio.speak("критично", config.AUDIO_PRIORITY_CRITICAL, "bg")
        priority, _, _ = audio._queue.get_nowait()
        assert priority == config.AUDIO_PRIORITY_CRITICAL == 0
