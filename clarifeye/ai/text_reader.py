"""
Text Reader Module — uses EasyOCR for natural-scene text recognition.
Supports Bulgarian and English.

EasyOCR downloads ~100MB of model weights on first run; ensure the Pi has
internet for the initial run, then it works fully offline afterward.
"""
import logging
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
from easyocr import Reader

import config

logger = logging.getLogger("clarifeye.ai.text_reader")


@dataclass
class TextDetection:
    bbox: Tuple[int, int, int, int]
    text: str
    confidence: float
    language: Optional[str] = None


class TextReader:
    def __init__(self) -> None:
        self._reader = Reader(['en', 'bg'], gpu=False)
        logger.info("TextReader initialized with EasyOCR (en, bg)")

    def read_text(self, frame: np.ndarray) -> List[TextDetection]:
        try:
            start = time.time()

            # EasyOCR expects BGR; the camera pipeline yields RGB
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame

            # paragraph=True groups nearby text into blocks — much better for
            # natural scenes (a multi-line sign becomes one block, not N lines)
            results = self._reader.readtext(frame_bgr, paragraph=True)

            detections = []
            for bbox_polygon, text, confidence in results:
                text = text.strip()
                if not text:
                    continue
                if confidence < config.OCR_CONFIDENCE_THRESHOLD:
                    continue

                # Convert EasyOCR's polygon to an axis-aligned bounding box
                xs = [pt[0] for pt in bbox_polygon]
                ys = [pt[1] for pt in bbox_polygon]
                bbox = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))

                lang = self._detect_language(text)
                detections.append(TextDetection(
                    bbox=bbox,
                    text=text,
                    confidence=confidence,
                    language=lang,
                ))

            elapsed = time.time() - start
            logger.debug("OCR completed in %.1fms, found %d text region(s)",
                         elapsed * 1000, len(detections))
            return detections

        except Exception as exc:
            logger.error("OCR failed: %s", exc)
            return []

    @staticmethod
    def _detect_language(text: str) -> str:
        cyrillic_chars = len([c for c in text if 'а' <= c.lower() <= 'я' or c in 'ѝ'])
        total_alpha = len([c for c in text if c.isalpha()])
        return "bg" if total_alpha > 0 and cyrillic_chars / total_alpha > 0.5 else "en"
