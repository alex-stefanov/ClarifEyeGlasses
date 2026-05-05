"""
ClarifEye Audio Key Registry

Single source of truth for every pre-recorded audio key.
Keys map to canonical English and Bulgarian text used for:
  - Generating RECORDING_SCRIPT.txt (recording guide)
  - WAV filename validation in WavPlayer
  - Piper/espeak-ng fallback text when a WAV file is missing

WAV file layout: data/audio/{language}/{key}.wav
"""
from typing import Dict

AUDIO_KEYS: Dict[str, Dict[str, str]] = {
    # ── System messages ───────────────────────────────────────────────────────
    "system_starting": {
        "en": "ClarifEye starting",
        "bg": "ClarifEye стартира",
        "category": "system",
    },
    "system_stopping": {
        "en": "System stopping",
        "bg": "Системата спира",
        "category": "system",
    },
    "system_recovering": {
        "en": "System recovering",
        "bg": "Системата се възстановява",
        "category": "system",
    },
    "battery_low": {
        "en": "Battery low",
        "bg": "Ниска батерия",
        "category": "system",
    },
    "language_set_en": {
        "en": "English",
        "bg": "English",
        "category": "system",
    },
    "language_set_bg": {
        "en": "Български",
        "bg": "Български",
        "category": "system",
    },
    "no_detection": {
        "en": "Nothing detected",
        "bg": "Нищо не е открито",
        "category": "system",
    },

    # ── Mode names ────────────────────────────────────────────────────────────
    "mode_traffic_light": {
        "en": "Traffic light mode",
        "bg": "Режим светофар",
        "category": "mode_name",
    },
    "mode_navigation": {
        "en": "Navigation mode",
        "bg": "Навигационен режим",
        "category": "mode_name",
    },
    "mode_text_reading": {
        "en": "Text reading mode",
        "bg": "Режим четене",
        "category": "mode_name",
    },
    "mode_currency": {
        "en": "Currency mode",
        "bg": "Режим валута",
        "category": "mode_name",
    },
    "mode_scene": {
        "en": "Scene description mode",
        "bg": "Режим описание",
        "category": "mode_name",
    },

    # ── Warning prefix ────────────────────────────────────────────────────────
    "warning": {
        "en": "Warning!",
        "bg": "Внимание!",
        "category": "warning",
    },

    # ── Connector ─────────────────────────────────────────────────────────────
    "and": {
        "en": "",   # Record 0.3 seconds of silence for English
        "bg": "и",
        "category": "connector",
    },

    # ── Positions ─────────────────────────────────────────────────────────────
    "pos_left": {
        "en": "on the left",
        "bg": "отляво",
        "category": "position",
    },
    "pos_center": {
        "en": "ahead",
        "bg": "отпред",
        "category": "position",
    },
    "pos_right": {
        "en": "on the right",
        "bg": "отдясно",
        "category": "position",
    },

    # ── Objects (mobility-relevant) ───────────────────────────────────────────
    "obj_person": {
        "en": "person",
        "bg": "човек",
        "category": "object",
    },
    "obj_bicycle": {
        "en": "bicycle",
        "bg": "велосипед",
        "category": "object",
    },
    "obj_car": {
        "en": "car",
        "bg": "кола",
        "category": "object",
    },
    "obj_motorcycle": {
        "en": "motorcycle",
        "bg": "мотор",
        "category": "object",
    },
    "obj_bus": {
        "en": "bus",
        "bg": "автобус",
        "category": "object",
    },
    "obj_truck": {
        "en": "truck",
        "bg": "камион",
        "category": "object",
    },
    "obj_traffic_light": {
        "en": "traffic light",
        "bg": "светофар",
        "category": "object",
    },
    "obj_stop_sign": {
        "en": "stop sign",
        "bg": "стоп знак",
        "category": "object",
    },
    "obj_fire_hydrant": {
        "en": "fire hydrant",
        "bg": "пожарен кран",
        "category": "object",
    },
    "obj_bench": {
        "en": "bench",
        "bg": "пейка",
        "category": "object",
    },

    # ── Traffic lights ────────────────────────────────────────────────────────
    "tl_red": {
        "en": "Red traffic light",
        "bg": "Червен светофар",
        "category": "traffic_light",
    },
    "tl_yellow": {
        "en": "Yellow traffic light",
        "bg": "Жълт светофар",
        "category": "traffic_light",
    },
    "tl_green": {
        "en": "Green traffic light",
        "bg": "Зелен светофар",
        "category": "traffic_light",
    },

    # ── Distances — centimetres (always plural; rounded to nearest 10 cm) ─────
    "dist_10_cm": {"en": "10 centimeters", "bg": "10 сантиметра", "category": "distance_cm"},
    "dist_20_cm": {"en": "20 centimeters", "bg": "20 сантиметра", "category": "distance_cm"},
    "dist_30_cm": {"en": "30 centimeters", "bg": "30 сантиметра", "category": "distance_cm"},
    "dist_40_cm": {"en": "40 centimeters", "bg": "40 сантиметра", "category": "distance_cm"},
    "dist_50_cm": {"en": "50 centimeters", "bg": "50 сантиметра", "category": "distance_cm"},
    "dist_60_cm": {"en": "60 centimeters", "bg": "60 сантиметра", "category": "distance_cm"},
    "dist_70_cm": {"en": "70 centimeters", "bg": "70 сантиметра", "category": "distance_cm"},
    "dist_80_cm": {"en": "80 centimeters", "bg": "80 сантиметра", "category": "distance_cm"},
    "dist_90_cm": {"en": "90 centimeters", "bg": "90 сантиметра", "category": "distance_cm"},

    # ── Distances — whole metres (Bulgarian singular vs plural) ───────────────
    "dist_1_m": {
        "en": "1 meter",
        "bg": "1 метър",
        "category": "distance_m",
        "notes": "Bulgarian singular form",
    },
    "dist_2_m": {"en": "2 meters", "bg": "2 метра", "category": "distance_m"},
    "dist_3_m": {"en": "3 meters", "bg": "3 метра", "category": "distance_m"},
    "dist_4_m": {"en": "4 meters", "bg": "4 метра", "category": "distance_m"},
    "dist_5_m": {"en": "5 meters", "bg": "5 метра", "category": "distance_m"},

    # ── Distance "far" ────────────────────────────────────────────────────────
    "dist_far": {
        "en": "far away",
        "bg": "далеч",
        "category": "distance_far",
    },

    # ── Currency — Bulgarian leva (singular vs plural) ────────────────────────
    "currency_5_lev":   {"en": "5 leva",   "bg": "5 лева",   "category": "currency"},
    "currency_10_lev":  {"en": "10 leva",  "bg": "10 лева",  "category": "currency"},
    "currency_20_lev":  {"en": "20 leva",  "bg": "20 лева",  "category": "currency"},
    "currency_50_lev":  {"en": "50 leva",  "bg": "50 лева",  "category": "currency"},
    "currency_100_lev": {"en": "100 leva", "bg": "100 лева", "category": "currency"},

    # ── Currency — Euros ──────────────────────────────────────────────────────
    "currency_5_eur":   {"en": "5 euros",   "bg": "5 евро",   "category": "currency"},
    "currency_10_eur":  {"en": "10 euros",  "bg": "10 евро",  "category": "currency"},
    "currency_20_eur":  {"en": "20 euros",  "bg": "20 евро",  "category": "currency"},
    "currency_50_eur":  {"en": "50 euros",  "bg": "50 евро",  "category": "currency"},
    "currency_100_eur": {"en": "100 euros", "bg": "100 евро", "category": "currency"},
    "currency_200_eur": {"en": "200 euros", "bg": "200 евро", "category": "currency"},

    # ── Action / feedback ─────────────────────────────────────────────────────
    # Could be replaced with a short beep WAV at data/audio/<lang>/action_received.wav.
    # The recording script will list it as "OK" but the user is free to record a beep
    # instead — both options work.
    "action_received": {
        "en": "OK",
        "bg": "OK",
        "category": "feedback",
    },
    "no_text_found": {
        "en": "No text found",
        "bg": "Не е намерен текст",
        "category": "feedback",
    },
    "no_currency_found": {
        "en": "No currency detected",
        "bg": "Не е разпозната валута",
        "category": "feedback",
    },
    "processing": {
        "en": "Processing",
        "bg": "Обработвам",
        "category": "feedback",
    },
    # language_switched announces the NEW language in the NEW language.
    # If the user toggles to Bulgarian, speak_key reads "bg" from Settings → plays "Избран български".
    # If the user toggles to English, speak_key reads "en" from Settings → plays "English selected".
    "language_switched": {
        "en": "English selected",
        "bg": "Избран български",
        "category": "feedback",
    },
    "more_text_truncated": {
        "en": "more text not read",
        "bg": "още текст не прочетен",
        "category": "system",
    },
}


def get_canonical_text(key: str, language: str) -> str:
    """
    Return the canonical text for an audio key in the requested language.

    Raises:
        KeyError: if *key* is not in the registry.
        ValueError: if *language* is not "en" or "bg".
    """
    if key not in AUDIO_KEYS:
        raise KeyError(f"Unknown audio key: {key!r}")
    if language not in ("en", "bg"):
        raise ValueError(f"Unsupported language: {language!r}")
    return AUDIO_KEYS[key][language]
