"""
ClarifEye Global Settings

Manages persistent app settings backed by data/settings.json.
The Settings instance is created once in main.py and passed to any module
that needs language awareness (AudioManager, ModeManager, etc.).
"""
import json
import logging
import os
import threading
from typing import Any, Dict

logger = logging.getLogger("clarifeye.core.settings")

_DEFAULTS: Dict[str, Any] = {
    "language": "en",
}


class Settings:
    def __init__(self, settings_path: str) -> None:
        self._path = settings_path
        self._lock = threading.Lock()
        self._data: Dict[str, Any] = dict(_DEFAULTS)
        self._load()

    def _load(self) -> None:
        try:
            with open(self._path, encoding="utf-8") as fh:
                loaded = json.load(fh)
            if isinstance(loaded, dict):
                self._data.update(loaded)
                logger.debug("Settings loaded from %s", self._path)
        except FileNotFoundError:
            pass  # Normal on first run — defaults will be used.
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not load settings from %s: %s", self._path, exc)

    def _save(self) -> None:
        tmp = self._path + ".tmp"
        try:
            os.makedirs(os.path.dirname(self._path), exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(self._data, fh, ensure_ascii=False, indent=2)
            os.replace(tmp, self._path)
        except OSError as exc:
            logger.error("Failed to save settings: %s", exc)
            try:
                os.unlink(tmp)
            except OSError:
                pass

    def get_language(self) -> str:
        with self._lock:
            return self._data.get("language", "en")

    def set_language(self, lang: str) -> None:
        if lang not in ("en", "bg"):
            raise ValueError(f"Unsupported language: {lang!r}")
        with self._lock:
            self._data["language"] = lang
            self._save()

    def toggle_language(self) -> str:
        with self._lock:
            current = self._data.get("language", "en")
            new_lang = "bg" if current == "en" else "en"
            self._data["language"] = new_lang
            self._save()
            return new_lang
