"""
Distance-to-audio-key converter for the ClarifEye priority engine.

Converts a fused distance in centimetres to a list of audio key strings
ready to pass to AudioManager.speak_sequence().
"""
from typing import List, Optional


def distance_to_keys(distance_cm: float, language: str) -> List[str]:
    """
    Convert a fused distance in cm to a list of audio keys for speak_sequence().

    Bucket rules:
      distance < 10 cm        → ["dist_10_cm"]  (clamped; don't say "0 cm")
      10 <= distance < 100    → rounded to nearest 10 cm, clamped [10, 90]
      100 <= distance < 500   → whole metres + remainder rounded to 10 cm;
                                if remainder rounds to 100, carry to next metre;
                                if remainder is 0 after rounding, omit "and" phrase
      distance >= 500         → ["dist_far"]

    Args:
        distance_cm: Fused distance in centimetres. Must not be None.
        language:    Language code (unused; accepted for future per-language formatting).

    Returns:
        List of audio key strings.

    Raises:
        TypeError: if distance_cm is None.
    """
    if distance_cm is None:
        raise TypeError("distance_cm must not be None")

    # Clamp negative values and sub-10 cm to the minimum bucket.
    if distance_cm < 10.0:
        return ["dist_10_cm"]

    # Far: >= 500 cm.
    if distance_cm >= 500.0:
        return ["dist_far"]

    # Sub-metre: 10 – 99.999 cm.
    if distance_cm < 100.0:
        rounded = round(distance_cm / 10.0) * 10
        rounded = max(10, min(90, rounded))
        return [f"dist_{int(rounded)}_cm"]

    # Metre range: 100 – 499.999 cm.
    # Python's built-in round() uses banker's rounding (round-half-to-even),
    # which matches the example: 305 cm → 3 m + 5 cm → rounds to 0 cm → "dist_3_m".
    whole_m = int(distance_cm // 100)
    remainder_cm = distance_cm % 100
    rounded_cm = round(remainder_cm / 10.0) * 10

    # Carry if rounding pushed remainder up to 100 cm.
    if rounded_cm >= 100:
        whole_m += 1
        rounded_cm = 0

    if rounded_cm == 0:
        return [f"dist_{whole_m}_m"]
    return [f"dist_{whole_m}_m", "and", f"dist_{int(rounded_cm)}_cm"]
