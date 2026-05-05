"""
ClarifEye Piper TTS Wrapper

Uses the piper CLI (from pip install piper-tts, or a manually downloaded
binary) to synthesize natural-sounding speech to a temporary WAV file,
then plays it via aplay.

CLI approach chosen over the Python API because:
- The piper Python package has complex native dependencies (onnxruntime wheels
  vary per Pi OS version).
- The single-binary CLI works reliably on Pi OS bookworm via pip install or
  the official GitHub release binary.
- aplay gives us the same subprocess/kill_event interrupt model as WavPlayer.

Voice models live in voices/ (downloaded separately via
scripts/download_piper_voices.sh):
  voices/en_US-amy-medium.onnx
  voices/en_US-amy-medium.onnx.json
  voices/bg_BG-dimitar-medium.onnx
  voices/bg_BG-dimitar-medium.onnx.json
"""
import logging
import os
import subprocess
import tempfile
import threading
from typing import Dict, Optional

logger = logging.getLogger("clarifeye.hardware.piper_tts")

_VOICE_MODELS: Dict[str, str] = {
    "en": "en_US-amy-medium.onnx",
    "bg": "bg_BG-dimitar-medium.onnx",
}


class PiperTTS:
    def __init__(self, voices_dir: str) -> None:
        self._voices_dir = voices_dir
        self._available: Dict[str, bool] = {}

        for lang, filename in _VOICE_MODELS.items():
            model_path = os.path.join(voices_dir, filename)
            if os.path.isfile(model_path):
                self._available[lang] = True
                logger.info("Piper voice loaded: %s (%s)", filename, lang)
            else:
                self._available[lang] = False
                logger.warning(
                    "Piper voice not found: %s — language %r will use espeak-ng. "
                    "Run scripts/download_piper_voices.sh to download.",
                    model_path, lang,
                )

        if not self._piper_binary_available():
            logger.warning(
                "piper binary not found in PATH — install with: pip install piper-tts"
            )
            self._available = {lang: False for lang in self._available}

    def is_available(self, language: str) -> bool:
        return self._available.get(language, False)

    def synthesize_to_wav(self, text: str, language: str, output_path: str) -> bool:
        """
        Synthesize *text* in *language* to a WAV file at *output_path*.
        Returns True on success.
        """
        model_path = os.path.join(self._voices_dir, _VOICE_MODELS.get(language, ""))
        if not os.path.isfile(model_path):
            return False

        try:
            result = subprocess.run(
                ["piper", "--model", model_path, "--output_file", output_path],
                input=text.encode("utf-8"),
                capture_output=True,
                timeout=10,
            )
            if result.returncode != 0:
                logger.error(
                    "Piper failed (rc=%d): %s", result.returncode,
                    result.stderr.decode(errors="replace"),
                )
                return False
            return os.path.isfile(output_path)
        except FileNotFoundError:
            logger.error("piper binary not found")
        except subprocess.TimeoutExpired:
            logger.error("Piper synthesis timed out for %r", text)
        except Exception as exc:
            logger.error("Piper synthesis error: %s", exc)
        return False

    def speak(
        self,
        text: str,
        language: str,
        kill_event: threading.Event,
    ) -> bool:
        """
        Synthesize *text* to a temp WAV and play it via aplay.
        Returns True if completed, False if interrupted or failed.
        """
        if not self.is_available(language):
            return False

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            if not self.synthesize_to_wav(text, language, tmp_path):
                return False

            if kill_event.is_set():
                return False

            try:
                proc = subprocess.Popen(
                    ["aplay", "-q", tmp_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except FileNotFoundError:
                logger.error("aplay not found — install alsa-utils")
                return False

            while proc.poll() is None:
                if kill_event.is_set():
                    proc.kill()
                    return False
                kill_event.wait(timeout=0.02)

            return proc.returncode == 0

        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    @staticmethod
    def _piper_binary_available() -> bool:
        try:
            subprocess.run(
                ["piper", "--version"],
                capture_output=True,
                timeout=2,
            )
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            return False
