"""
ClarifEye Recording Processor

Walks data/audio/bg/ and data/audio/en/, then for each WAV file:
  1. Converts to 22050 Hz mono if needed.
  2. Trims leading/trailing silence (threshold: -40 dB, ~0.01 amplitude).
  3. Normalises peak amplitude to -3 dB (~0.708 linear).
  4. Writes back to the same path (overwrites the raw recording).

Also cross-references against the AUDIO_KEYS registry and prints a report
showing which keys are present, which are missing, and any unknown WAV files.

Usage:
    python tools/process_recordings.py

Dependencies:
    pip install soundfile numpy
"""
import os
import sys

import numpy as np

# Allow running from repo root or from tools/
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

try:
    import soundfile as sf
except ImportError:
    print("ERROR: soundfile not installed. Run: pip install soundfile")
    sys.exit(1)

from hardware.audio_keys import AUDIO_KEYS
import config

TARGET_SR = 22050
SILENCE_THRESHOLD = 0.01      # ~-40 dB amplitude
PEAK_TARGET = 10 ** (-3 / 20)  # -3 dB linear ≈ 0.708

LANGUAGES = ["bg", "en"]


def trim_silence(samples: np.ndarray, threshold: float = SILENCE_THRESHOLD) -> np.ndarray:
    """Remove leading and trailing silence from a 1-D float array."""
    abs_s = np.abs(samples)
    above = np.where(abs_s > threshold)[0]
    if len(above) == 0:
        return samples
    return samples[above[0]: above[-1] + 1]


def normalise(samples: np.ndarray, target_peak: float = PEAK_TARGET) -> np.ndarray:
    """Scale samples so the peak absolute value equals target_peak."""
    peak = np.max(np.abs(samples))
    if peak == 0:
        return samples
    return samples * (target_peak / peak)


def process_wav(path: str) -> bool:
    """
    Process a single WAV file in-place.
    Returns True on success.
    """
    try:
        data, sr = sf.read(path, dtype="float32", always_2d=True)
    except Exception as exc:
        print(f"  ERROR reading {path}: {exc}")
        return False

    # Convert stereo → mono by averaging channels.
    if data.shape[1] > 1:
        data = data.mean(axis=1)
    else:
        data = data[:, 0]

    # Resample if needed (simple linear — good enough for speech post-processing).
    if sr != TARGET_SR:
        ratio = TARGET_SR / sr
        new_len = int(len(data) * ratio)
        data = np.interp(
            np.linspace(0, len(data) - 1, new_len),
            np.arange(len(data)),
            data,
        )

    data = trim_silence(data)
    data = normalise(data)

    try:
        sf.write(path, data, TARGET_SR, subtype="PCM_16")
    except Exception as exc:
        print(f"  ERROR writing {path}: {exc}")
        return False

    duration_ms = len(data) / TARGET_SR * 1000
    print(f"  OK  {os.path.basename(path)}  ({duration_ms:.0f} ms)")
    return True


def main() -> None:
    known_keys = set(AUDIO_KEYS.keys())
    present: dict = {lang: set() for lang in LANGUAGES}
    unknown: dict = {lang: [] for lang in LANGUAGES}
    errors: dict = {lang: [] for lang in LANGUAGES}

    for lang in LANGUAGES:
        audio_dir = os.path.join(config.AUDIO_DIR, lang)
        if not os.path.isdir(audio_dir):
            print(f"[{lang}] Directory not found: {audio_dir}")
            continue

        wav_files = sorted(
            f for f in os.listdir(audio_dir) if f.lower().endswith(".wav")
        )

        if not wav_files:
            print(f"[{lang}] No WAV files found in {audio_dir}")
            continue

        print(f"\n[{lang}] Processing {len(wav_files)} file(s) in {audio_dir} ...")

        for fname in wav_files:
            key = fname[:-4]  # Strip .wav
            path = os.path.join(audio_dir, fname)

            if key not in known_keys:
                unknown[lang].append(fname)
                print(f"  UNKNOWN key: {fname} (not in AUDIO_KEYS registry)")
                continue

            ok = process_wav(path)
            if ok:
                present[lang].add(key)
            else:
                errors[lang].append(key)

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("REPORT")
    print("=" * 60)

    for lang in LANGUAGES:
        missing = sorted(known_keys - present[lang])
        print(f"\n  Language: {lang}")
        print(f"    Present : {len(present[lang])} / {len(known_keys)} keys")
        print(f"    Missing : {len(missing)}")
        if missing:
            for k in missing:
                print(f"      - {k}")
        if unknown[lang]:
            print(f"    Unknown WAV files (not in registry):")
            for f in unknown[lang]:
                print(f"      ! {f}")
        if errors[lang]:
            print(f"    Errors during processing:")
            for k in errors[lang]:
                print(f"      x {k}")

    total_missing = sum(len(known_keys - present[lang]) for lang in LANGUAGES)
    if total_missing == 0:
        print(f"\nAll {len(known_keys)} keys present in both languages. Ready to deploy.")
    else:
        print(f"\n{total_missing} recording(s) still missing. See above.")


if __name__ == "__main__":
    main()
