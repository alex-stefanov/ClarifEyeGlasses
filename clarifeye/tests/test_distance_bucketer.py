"""Unit tests for core/distance_bucketer.py"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from core.distance_bucketer import distance_to_keys


@pytest.mark.parametrize("dist_cm, expected", [
    # Edge / clamp
    (-10,  ["dist_10_cm"]),
    (0,    ["dist_10_cm"]),
    (5,    ["dist_10_cm"]),
    # Sub-metre exact and rounded
    (10,   ["dist_10_cm"]),
    (23,   ["dist_20_cm"]),   # round(2.3)=2 → 20 cm
    (50,   ["dist_50_cm"]),
    (95,   ["dist_90_cm"]),   # round(9.5)=10 (banker's) → 100, clamped to 90
    (99,   ["dist_90_cm"]),   # round(9.9)=10 → 100, clamped to 90
    # Metre-range boundaries
    (100,  ["dist_1_m"]),
    (105,  ["dist_1_m"]),     # 5 cm → round(0.5)=0 (banker's) → no cm phrase
    (127,  ["dist_1_m", "and", "dist_30_cm"]),
    (200,  ["dist_2_m"]),
    (250,  ["dist_2_m", "and", "dist_50_cm"]),
    (305,  ["dist_3_m"]),     # 5 cm → round(0.5)=0 → no cm phrase
    (495,  ["dist_5_m"]),     # 95 cm → rounds to 100, carry → 5 m
    # Far boundary (inclusive)
    (500,  ["dist_far"]),
    (600,  ["dist_far"]),
])
def test_distance_to_keys(dist_cm, expected):
    assert distance_to_keys(dist_cm, "bg") == expected


def test_language_param_unused():
    """Language parameter is accepted but currently has no effect on output."""
    assert distance_to_keys(127, "en") == distance_to_keys(127, "bg")


def test_none_raises():
    with pytest.raises(TypeError):
        distance_to_keys(None, "bg")
