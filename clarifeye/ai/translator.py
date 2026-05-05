"""
ClarifEye Translator

Offline translation for TEXT_READING mode with a three-level cache.

Resolution order
~~~~~~~~~~~~~~~~
1. **Static phrase cache** — ``data/common_phrases.json`` loaded at startup.
   Read-only; covers ~50 common en↔bg signs and landmarks.

2. **Runtime translation cache** — ``data/translation_cache.json``.
   Populated on argostranslate successes and persisted atomically to disk.

3. **argostranslate** — fully offline neural translation, ~50–200ms on Pi 4.
   Requires the en↔bg packages to be installed once via
   ``python scripts/install_argos_packages.py``.

Cache key format
~~~~~~~~~~~~~~~~
``"{src}_{tgt}:{text_lower}"``  e.g. ``"en_bg:exit"`` → ``"Изход"``

Thread safety
~~~~~~~~~~~~~
All mutable state (``_runtime_cache``, ``_cache``, hit/miss counters) is
guarded by ``_lock``.  The argostranslate call runs *outside* the lock.
"""
import json
import logging
import os
import threading
from typing import Dict, Optional

try:
    from .. import config
except ImportError:
    import config  # type: ignore[no-redef]

logger = logging.getLogger("clarifeye.ai.translator")

_STATS_LOG_INTERVAL: int = 50


class Translator:
    """
    Offline-first translator backed by argostranslate and a persistent cache.

    Pass the shared Settings instance so translate_to_user_language() can
    read the current language without a separate argument on each call.
    """

    def __init__(self, settings) -> None:
        self._settings = settings
        self._lock = threading.Lock()

        self._phrases: Dict[str, str] = {}
        self._runtime_cache: Dict[str, str] = {}
        self._cache: Dict[str, str] = {}

        self._hits: int = 0
        self._misses: int = 0
        self._total: int = 0

        self._cache_path: str = config.TRANSLATION_CACHE_PATH

        self._load_common_phrases()
        self._load_runtime_cache()
        self._rebuild_lookup()

        # Pre-load argostranslate translation objects for en↔bg
        # (get_installed_languages is slow; do it once at startup)
        self._argos: Dict[str, object] = {}
        self._init_argos()

        logger.info(
            "Translator ready: %d static phrase(s), %d runtime cache entry(s), "
            "%d argos direction(s).",
            len(self._phrases),
            len(self._runtime_cache),
            len(self._argos),
        )

    # ── Initialisation helpers ────────────────────────────────────────────────

    def _load_common_phrases(self) -> None:
        try:
            with open(config.COMMON_PHRASES_PATH, encoding="utf-8") as fh:
                data = json.load(fh)
        except FileNotFoundError:
            logger.warning("common_phrases.json not found at '%s'.", config.COMMON_PHRASES_PATH)
            return
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not load common_phrases.json: %s", exc)
            return

        for src_text, tgt_text in data.get("en_to_bg", {}).items():
            self._phrases[f"en_bg:{src_text.lower()}"] = str(tgt_text)

        for src_text, tgt_text in data.get("bg_to_en", {}).items():
            self._phrases[f"bg_en:{src_text.lower()}"] = str(tgt_text)

        logger.debug("Loaded %d static phrase(s).", len(self._phrases))

    def _load_runtime_cache(self) -> None:
        try:
            with open(self._cache_path, encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                self._runtime_cache = {str(k): str(v) for k, v in data.items()}
                logger.debug("Loaded %d runtime cache entry(s).", len(self._runtime_cache))
            else:
                logger.warning("translation_cache.json has unexpected format; starting fresh.")
        except FileNotFoundError:
            pass
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Corrupted translation cache, starting fresh: %s", exc)

    def _rebuild_lookup(self) -> None:
        self._cache = {**self._phrases, **self._runtime_cache}

    def _init_argos(self) -> None:
        try:
            from argostranslate import translate as _at
            installed = _at.get_installed_languages()
            lang_map = {lang.code: lang for lang in installed}

            for src_code, tgt_code in [("en", "bg"), ("bg", "en")]:
                key = f"{src_code}_{tgt_code}"
                src_lang = lang_map.get(src_code)
                tgt_lang = lang_map.get(tgt_code)
                if src_lang and tgt_lang:
                    translation = src_lang.get_translation(tgt_lang)
                    if translation:
                        self._argos[key] = translation
                        logger.debug("argostranslate: %s → %s ready.", src_code, tgt_code)
                    else:
                        logger.warning(
                            "argostranslate: no package installed for %s → %s. "
                            "Run: python scripts/install_argos_packages.py",
                            src_code, tgt_code,
                        )
                else:
                    missing = [c for c in (src_code, tgt_code) if c not in lang_map]
                    logger.warning(
                        "argostranslate: language(s) not installed: %s. "
                        "Run: python scripts/install_argos_packages.py",
                        missing,
                    )
        except ImportError:
            logger.warning(
                "argostranslate is not installed; translation will use cache only. "
                "Install with: pip install argostranslate"
            )

    # ── Public API ────────────────────────────────────────────────────────────

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> Optional[str]:
        """
        Translate *text* from *source_lang* to *target_lang*.

        Resolution order:
          1. Local cache (static phrases + runtime cache) — instant.
          2. argostranslate — offline, ~50–200ms.
          3. Return None on failure (caller falls back to original text).

        Returns:
            Translated string, or None if translation is unavailable.
        """
        text = text.strip()
        if not text:
            return text

        if source_lang == target_lang:
            return text

        key = f"{source_lang}_{target_lang}:{text.lower()}"

        # ── 1. Cache lookup ────────────────────────────────────────────────────
        with self._lock:
            self._total += 1
            if key in self._cache:
                self._hits += 1
                result = self._cache[key]
                logger.debug("Cache HIT  [%s] %r → %r", key, text, result)
                self._maybe_log_stats()
                return result

        # ── 2. argostranslate (outside lock — blocking but fast) ───────────────
        translated = self._call_argos(text, source_lang, target_lang)

        with self._lock:
            self._misses += 1
            if translated is not None:
                self._runtime_cache[key] = translated
                self._rebuild_lookup()
                self._save_cache()
                logger.debug("Cache MISS [%s] %r → %r (argos).", key, text, translated)
                self._maybe_log_stats()
                return translated

            logger.warning(
                "Translation unavailable for %r (%s→%s).",
                text, source_lang, target_lang,
            )
            self._maybe_log_stats()
            return None

    def translate_to_user_language(self, text: str, source_lang: str) -> str:
        """
        Translate *text* to the user's current language (from Settings).

        If source already matches the user language, returns *text* unchanged.
        Returns the original *text* on translation failure.
        """
        user_lang = self._settings.get_language()
        if source_lang == user_lang:
            return text
        result = self.translate(text, source_lang, user_lang)
        return result if result is not None else text

    def get_stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "total": self._total,
                "cache_size": len(self._cache),
            }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _call_argos(self, text: str, source_lang: str, target_lang: str) -> Optional[str]:
        key = f"{source_lang}_{target_lang}"
        translation = self._argos.get(key)
        if translation is None:
            logger.debug("No argos translation object for %s → %s.", source_lang, target_lang)
            return None
        try:
            result = translation.translate(text)  # type: ignore[union-attr]
            return result if result else None
        except Exception as exc:
            logger.warning("argostranslate error for %r: %s", text, exc)
            return None

    def _save_cache(self) -> None:
        tmp_path = self._cache_path + ".tmp"
        try:
            os.makedirs(os.path.dirname(self._cache_path), exist_ok=True)
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(self._runtime_cache, fh, ensure_ascii=False, indent=2)
            os.replace(tmp_path, self._cache_path)
            logger.debug(
                "Saved translation cache (%d entry(s)) to '%s'.",
                len(self._runtime_cache), self._cache_path,
            )
        except OSError as exc:
            logger.error("Failed to save translation cache: %s", exc)
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _maybe_log_stats(self) -> None:
        if self._total == 0 or self._total % _STATS_LOG_INTERVAL != 0:
            return
        hit_rate = self._hits / self._total * 100
        logger.info(
            "Translation stats — total: %d  hits: %d (%.0f%%)  misses: %d  cache_size: %d",
            self._total, self._hits, hit_rate, self._misses, len(self._cache),
        )
