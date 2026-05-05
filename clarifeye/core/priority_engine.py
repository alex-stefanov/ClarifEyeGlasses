"""
ClarifEye Priority Engine — audio-key-based announcement pipeline.

Converts fused detections into speak_sequence() calls using the pre-recorded
audio key registry.  All announcements are language-independent at this layer;
the audio system handles language lookup.

Sequence template per detection (in order):
  1. "warning"              — prepended if threat_score >= 7.0
  2. "obj_{class_name}"     — e.g. "obj_car", "obj_person"
  3. distance keys          — from distance_bucketer.distance_to_keys()
  4. "pos_{position}"       — "pos_left" | "pos_center" | "pos_right"

Example: car at 127 cm, centre, threat 8.0 →
  ["warning", "obj_car", "dist_1_m", "and", "dist_30_cm", "pos_center"]
"""
import logging
import threading
import time
from typing import Dict, List, Optional, Tuple

try:
    from .. import config
except ImportError:
    import config  # type: ignore[no-redef]

try:
    from .detection import Detection
except ImportError:
    try:
        from ai.detection import Detection  # type: ignore[no-redef]
    except ImportError:
        from detection import Detection  # type: ignore[no-redef]

try:
    from .distance_bucketer import distance_to_keys
except ImportError:
    try:
        from core.distance_bucketer import distance_to_keys  # type: ignore[no-redef]
    except ImportError:
        from distance_bucketer import distance_to_keys  # type: ignore[no-redef]

try:
    from ..hardware.audio_keys import AUDIO_KEYS
except ImportError:
    try:
        from hardware.audio_keys import AUDIO_KEYS  # type: ignore[no-redef]
    except ImportError:
        AUDIO_KEYS = {}  # type: ignore[assignment]

logger = logging.getLogger("clarifeye.core.priority_engine")

_STALE_CLEANUP_SEC = 15.0
_APPROACH_THRESHOLD_CM = 2.0      # min delta to count as "approaching"
_APPROACH_BOOST = 1.5             # threat multiplier when object is closing
_COOLDOWN_MAX_SEC = 15.0          # cap for scaled cooldown
_APPROACH_BYPASS_RATIO = 0.25     # bypass cooldown if distance shrank >= 25%


class PriorityEngine:
    """
    Converts fused Detection lists into prioritised audio-key sequences.

    Cooldown key   — det.tracker_id when available, else det.class_name.
    Cooldown scale — longer cooldowns for lower-threat objects (less urgent
                     objects don't need frequent re-announcement).
    Distance bypass— if the same object is now >= 25% closer than when last
                     announced, the cooldown is ignored (approaching fast).
    """

    def __init__(self, audio_manager) -> None:
        self._audio = audio_manager
        # cooldown_key → (last_announced_time, last_announced_dist_cm)
        self._recently_notified: Dict[str, Tuple[float, Optional[float]]] = {}
        self._prev_distances: Dict[str, Optional[float]] = {}
        self._lock = threading.Lock()
        self._last_cleanup = time.time()

    # ── Public API ─────────────────────────────────────────────────────────────

    def process_detections(self, detections: List[Detection]) -> List[Detection]:
        """
        Select the highest-threat detections, build audio key sequences,
        and enqueue them via AudioManager.speak_sequence().

        Args:
            detections: Fused detections from the current frame.

        Returns:
            Subset of detections that triggered an announcement this cycle.
        """
        now = time.time()
        selected: List[Detection] = []

        with self._lock:
            self._cleanup_stale_notifications(now)

            # Score all detections and cache prev distances for next cycle.
            for det in detections:
                object_id = self._cooldown_key(det)
                det.threat_score = self._compute_threat(det, object_id)
                if det.fused_distance_cm is not None:
                    self._prev_distances[object_id] = det.fused_distance_cm

            detections.sort(key=lambda d: d.threat_score, reverse=True)

            for det in detections:
                if det.threat_score <= 0.0:
                    continue
                if len(selected) >= config.MAX_NOTIFICATION_OBJECTS:
                    break

                # Second slot requires a minimum threat score.
                if len(selected) >= 1 and det.threat_score < config.SECOND_ANNOUNCEMENT_MIN_SCORE:
                    continue

                obj_key = f"obj_{det.class_name}"
                if obj_key not in AUDIO_KEYS:
                    logger.warning(
                        "No audio key for class '%s' ('%s' not in registry) — "
                        "skipping announcement.",
                        det.class_name,
                        obj_key,
                    )
                    continue

                cooldown_key = self._cooldown_key(det)
                record = self._recently_notified.get(cooldown_key)
                last_t = record[0] if record else 0.0
                last_dist = record[1] if record else None
                cooldown_sec = self._compute_cooldown(det.threat_score)

                in_cooldown = (now - last_t) < cooldown_sec
                if in_cooldown:
                    if self._approaching_fast(det.fused_distance_cm, last_dist):
                        logger.debug(
                            "Cooldown bypass (approaching ≥25%%): %s  "
                            "last=%.0f cm  now=%.0f cm",
                            cooldown_key,
                            last_dist or 0.0,
                            det.fused_distance_cm or 0.0,
                        )
                    else:
                        continue

                keys = self._build_sequence(det)
                priority = self._audio_priority(det.threat_score)

                if self._audio is not None:
                    self._audio.speak_sequence(keys, priority, cooldown=None)

                self._recently_notified[cooldown_key] = (now, det.fused_distance_cm)
                selected.append(det)

                logger.info(
                    "Announced: %-20s  dist=%s  score=%5.2f  pri=%d  keys=%s",
                    cooldown_key,
                    f"{det.fused_distance_cm:.0f} cm"
                    if det.fused_distance_cm is not None
                    else "unknown",
                    det.threat_score,
                    priority,
                    keys,
                )

        return selected

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _cooldown_key(self, det: Detection) -> str:
        """Return the stable per-object key used for cooldown tracking."""
        return det.tracker_id if det.tracker_id is not None else det.class_name

    def _build_sequence(self, det: Detection) -> List[str]:
        """Build the ordered list of audio keys for this detection."""
        keys: List[str] = []

        if det.threat_score >= 7.0:
            keys.append("warning")

        keys.append(f"obj_{det.class_name}")

        if det.fused_distance_cm is not None:
            try:
                keys.extend(distance_to_keys(det.fused_distance_cm, "bg"))
            except (TypeError, ValueError) as exc:
                logger.warning("distance_to_keys failed for %.1f cm: %s",
                               det.fused_distance_cm, exc)

        pos_key = f"pos_{det.position}"
        if pos_key in AUDIO_KEYS:
            keys.append(pos_key)

        return keys

    def _compute_cooldown(self, threat_score: float) -> float:
        """
        Scale cooldown inversely with threat: high-threat objects are
        re-announced sooner.

        Formula: max(2.0, min(15.0, base * 5 / max(score, 1)))
          score=10 → 3.0 s   score=5 → 6.0 s   score=1 → 30 s (capped at 15 s)
        """
        base = config.NOTIFICATION_COOLDOWN_SEC
        cooldown = base * 5.0 / max(threat_score, 1.0)
        return max(2.0, min(_COOLDOWN_MAX_SEC, cooldown))

    def _compute_threat(self, det: Detection, object_id: str) -> float:
        base = float(
            config.THREAT_BASE_SCORES.get(det.class_name, int(config.ThreatLevel.LOW))
        )
        dist = det.fused_distance_cm
        dist_factor = 0.3 if dist is None else max(0.0, 1.0 - dist / 500.0)

        prev_dist = self._prev_distances.get(object_id)
        if (
            prev_dist is not None
            and dist is not None
            and (prev_dist - dist) > _APPROACH_THRESHOLD_CM
        ):
            vel_boost = _APPROACH_BOOST
        else:
            vel_boost = 1.0

        return base * dist_factor * vel_boost

    @staticmethod
    def _approaching_fast(
        current: Optional[float],
        last: Optional[float],
    ) -> bool:
        """Return True if the object has closed >= 25% of its last distance."""
        if current is None or last is None or last <= 0.0:
            return False
        return (last - current) / last >= _APPROACH_BYPASS_RATIO

    def _cleanup_stale_notifications(self, now: float) -> None:
        if now - self._last_cleanup < _STALE_CLEANUP_SEC / 2:
            return
        stale = [
            k for k, (last_t, _) in self._recently_notified.items()
            if now - last_t > _STALE_CLEANUP_SEC
        ]
        for k in stale:
            self._recently_notified.pop(k, None)
            self._prev_distances.pop(k, None)
        self._last_cleanup = now

    def _audio_priority(self, threat_score: float) -> int:
        if threat_score >= 8.0:
            return config.AUDIO_PRIORITY_CRITICAL
        if threat_score >= 5.0:
            return config.AUDIO_PRIORITY_HIGH
        if threat_score >= 3.0:
            return config.AUDIO_PRIORITY_NORMAL
        return config.AUDIO_PRIORITY_LOW
