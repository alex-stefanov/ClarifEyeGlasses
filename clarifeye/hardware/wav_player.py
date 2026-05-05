"""
ClarifEye WAV File Player

Plays pre-recorded WAV files via aplay (ALSA, standard on Pi OS).
Files live at: {audio_dir}/{language}/{key}.wav
Expected format: 22050 Hz, mono, 16-bit PCM.
"""
import logging
import os
import subprocess
import threading
from typing import List, Optional

logger = logging.getLogger("clarifeye.hardware.wav_player")


class WavPlayer:
    def __init__(self, audio_dir: str) -> None:
        self._audio_dir = audio_dir

    def _path(self, key: str, language: str) -> str:
        return os.path.join(self._audio_dir, language, f"{key}.wav")

    def has(self, key: str, language: str) -> bool:
        return os.path.isfile(self._path(key, language))

    def play(self, key: str, language: str) -> Optional[subprocess.Popen]:
        """
        Spawn aplay for a single WAV file.
        Returns the Popen handle so the caller can kill it on interruption,
        or None if the file is missing.
        """
        path = self._path(key, language)
        if not os.path.isfile(path):
            return None
        try:
            proc = subprocess.Popen(
                ["aplay", "-q", path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return proc
        except FileNotFoundError:
            logger.error("aplay not found — install with: sudo apt install alsa-utils")
        except Exception as exc:
            logger.error("aplay error for %r: %s", path, exc)
        return None

    def play_sequence(
        self,
        keys: List[str],
        language: str,
        kill_event: threading.Event,
    ) -> bool:
        """
        Play a list of WAV files back-to-back with no gap.

        Checks kill_event between files so an interrupt stops the sequence
        cleanly. Returns True if the entire sequence completed, False if
        interrupted or any file was missing.
        """
        for key in keys:
            if kill_event.is_set():
                return False

            path = self._path(key, language)
            if not os.path.isfile(path):
                logger.warning("WAV missing for sequence key %r (%s)", key, language)
                return False

            try:
                proc = subprocess.Popen(
                    ["aplay", "-q", path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except FileNotFoundError:
                logger.error("aplay not found")
                return False
            except Exception as exc:
                logger.error("aplay error: %s", exc)
                return False

            while proc.poll() is None:
                if kill_event.is_set():
                    proc.kill()
                    return False
                kill_event.wait(timeout=0.02)

        return True
