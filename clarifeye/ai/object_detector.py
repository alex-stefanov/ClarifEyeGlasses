"""
ClarifEye Object Detector — YOLOv8n COCO TFLite Inference

Replaces the SSD MobileNet V2 detector. Uses YOLOv8n exported to TFLite INT8
at 320x320 input resolution.

YOLOv8 TFLite output shape (typical for 320x320)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  [1, 84, 2100]
  84   = 4 bbox coords (cx, cy, w, h in model-pixel space 0-320)
        + 80 COCO class scores (softmax-activated by the export)
  2100 = 40x40 + 20x20 + 10x10 anchor grid cells across 3 strides

Both orientations are handled automatically:
  [1, 84, N]  → after batch-dim drop: [84, N]  → transposed to [N, 84]
  [1, N, 84]  → after batch-dim drop: [N, 84]  → used directly

If neither axis equals 84 an error is logged and an empty list is returned
(graceful degradation — never crashes).

Camera is 640x640; model input is 320x320, so resize ratio is 2.0 and there
is no letterboxing needed (both are square). Bboxes are scaled back to 640x640.

Thread safety
~~~~~~~~~~~~~
ai_edge_litert.Interpreter is not concurrent-safe. Call detect() from a
single thread (the pipeline processing thread).

Export
~~~~~~
Run scripts/export_yolov8n_coco.py on a desktop machine (not the Pi),
then copy models/yolov8n_coco_int8.tflite to the Pi.
"""
import logging
import os
import time
from typing import List, Optional, Tuple

import numpy as np

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
    import cv2  # type: ignore[import-untyped]
    _OPENCV_AVAILABLE = True
except ImportError:
    _OPENCV_AVAILABLE = False

logger = logging.getLogger("clarifeye.ai.object_detector")

_COCO_NUM_CLASSES = 80  # YOLOv8n trained on COCO 80-class dataset


# ─── Module-level helpers ──────────────────────────────────────────────────────

def estimate_distance_cm(
    detection: "Detection",
    frame_height: int,
) -> Optional[float]:
    """
    Estimate object distance via the pinhole camera model.

    distance_m = (known_real_height_m * focal_length_px) / bbox_height_px

    Args:
        detection:    Detection with bbox in original-frame pixels.
        frame_height: Original frame height (kept for interface symmetry).

    Returns:
        Estimated distance in centimetres (1 dp), or None if the class has
        no reference height or the bbox has zero height.
    """
    known_height_m: Optional[float] = config.KNOWN_HEIGHTS.get(detection.class_name)
    if known_height_m is None:
        return None
    pixel_height: int = detection.bbox[3] - detection.bbox[1]
    if pixel_height <= 0:
        return None
    distance_m = (known_height_m * config.CAMERA_FOCAL_LENGTH_PX) / pixel_height
    return round(distance_m * 100.0, 1)


def _compute_iou(
    box_a: Tuple[int, int, int, int],
    box_b: Tuple[int, int, int, int],
) -> float:
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])
    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    intersection = inter_w * inter_h
    if intersection == 0:
        return 0.0
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _apply_nms(
    detections: List["Detection"],
    iou_threshold: float,
) -> List["Detection"]:
    """Greedy NMS, sorted by descending confidence."""
    if not detections:
        return []
    ordered = sorted(detections, key=lambda d: d.confidence, reverse=True)
    kept: List["Detection"] = []
    while ordered:
        best = ordered[0]
        kept.append(best)
        ordered = [
            d for d in ordered[1:]
            if _compute_iou(best.bbox, d.bbox) <= iou_threshold
        ]
    return kept


# ─── Detector ─────────────────────────────────────────────────────────────────

class ObjectDetector:
    """
    YOLOv8n COCO obstacle detector for ClarifEye navigation mode.

    Class interface is identical to the previous SSD detector so
    main.py and sensor_fusion.py require no changes.
    """

    def __init__(
        self,
        model_path: str = config.OBJECT_MODEL_PATH,
    ) -> None:
        self._model_path = model_path
        self._interpreter: Optional[object] = None
        self._input_details: Optional[list] = None
        self._output_details: Optional[list] = None
        self._input_h: int = 320
        self._input_w: int = 320
        self._input_dtype: type = np.int8
        self._available: bool = False

        if not os.path.exists(model_path):
            logger.error(
                "TFLite model not found: %s\n"
                "Run scripts/export_yolov8n_coco.py on a desktop machine, "
                "then copy models/yolov8n_coco_int8.tflite to the Pi.",
                model_path,
            )
            return

        try:
            from ai_edge_litert.interpreter import (  # type: ignore[import-untyped]
                Interpreter,
            )
        except ImportError:
            logger.error(
                "ai_edge_litert is not installed. "
                "Install with: pip install ai-edge-litert"
            )
            return

        try:
            self._interpreter = Interpreter(model_path=model_path)
            self._interpreter.allocate_tensors()  # type: ignore[union-attr]
            self._input_details = (
                self._interpreter.get_input_details()  # type: ignore[union-attr]
            )
            self._output_details = (
                self._interpreter.get_output_details()  # type: ignore[union-attr]
            )

            in_shape = self._input_details[0]["shape"]   # [1, H, W, C]
            out_shape = tuple(self._output_details[0]["shape"])
            self._input_h = int(in_shape[1])
            self._input_w = int(in_shape[2])
            self._input_dtype = self._input_details[0]["dtype"]

            logger.info(
                "ObjectDetector ready  model=%s  input=%s (%s)  output=%s  "
                "whitelist_classes=%d",
                os.path.basename(model_path),
                tuple(in_shape),
                self._input_dtype.__name__,
                out_shape,
                len(config.OBJECT_CLASSES_OF_INTEREST),
            )
            self._available = True

        except Exception as exc:
            logger.error("ObjectDetector initialisation failed: %s", exc)

    # ── Public API ─────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run inference on *frame* and return NMS-filtered obstacle detections.

        Args:
            frame: Full-resolution RGB numpy.ndarray (any spatial size).

        Returns:
            Detection list sorted by descending confidence.
            Empty list on any error or when no detections pass all filters.
        """
        if not self._available or self._interpreter is None:
            return []

        if not _OPENCV_AVAILABLE:
            logger.error("OpenCV unavailable — cannot preprocess frame for inference.")
            return []

        try:
            orig_h, orig_w = frame.shape[:2]

            input_tensor = self._preprocess(frame)
            self._interpreter.set_tensor(  # type: ignore[union-attr]
                self._input_details[0]["index"], input_tensor
            )

            t_start = time.perf_counter()
            self._interpreter.invoke()  # type: ignore[union-attr]
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            logger.info("Inference: %.1f ms", elapsed_ms)

            raw_output = self._interpreter.get_tensor(  # type: ignore[union-attr]
                self._output_details[0]["index"]
            )
            output_f32 = self._dequantize_output(raw_output)

            return self._parse_detections(output_f32, orig_h, orig_w)

        except Exception as exc:
            logger.error("detect() error: %s", exc)
            return []

    # ── Preprocessing ──────────────────────────────────────────────────────────

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Resize frame to model input size and apply quantisation-appropriate
        normalisation. Returns [1, H, W, 3] ready for the interpreter.
        """
        resized = cv2.resize(frame, (self._input_w, self._input_h))

        if self._input_dtype == np.float32:
            tensor = resized.astype(np.float32) / 255.0

        elif self._input_dtype == np.uint8:
            tensor = resized.astype(np.uint8)

        else:  # int8 — most common for exported INT8 models
            quant = self._input_details[0]["quantization"]
            scale: float = float(quant[0]) if float(quant[0]) != 0.0 else 1.0 / 255.0
            zero_point: int = int(quant[1])
            normalized = resized.astype(np.float32) / 255.0
            tensor = np.clip(
                np.round(normalized / scale) + zero_point,
                -128,
                127,
            ).astype(np.int8)

        return np.expand_dims(tensor, axis=0)

    # ── Output dequantisation ──────────────────────────────────────────────────

    def _dequantize_output(self, raw: np.ndarray) -> np.ndarray:
        """
        Convert INT8 output tensor to float32 using stored quantisation params.
        Float32 tensors pass through with only a dtype cast.
        """
        if raw.dtype in (np.int8, np.uint8):
            quant = self._output_details[0]["quantization"]
            scale: float = float(quant[0]) if float(quant[0]) != 0.0 else 1.0
            zero_point: int = int(quant[1])
            return (raw.astype(np.float32) - zero_point) * scale
        return raw.astype(np.float32)

    # ── Output parsing ─────────────────────────────────────────────────────────

    def _parse_detections(
        self,
        output_f32: np.ndarray,
        orig_h: int,
        orig_w: int,
    ) -> List[Detection]:
        """
        Parse YOLOv8 output tensor into Detection objects.

        Normalises to [N, 84] (anchors x features), where:
          row[:4]  = cx, cy, w, h  in model-input pixel space
          row[4:]  = 80 COCO class scores (softmax-activated)

        Applies: class whitelist → per-class confidence → NMS.
        Scales bboxes back to original frame dimensions.

        Logs counts at DEBUG: raw, after class filter, after confidence, after NMS.
        """
        output = output_f32[0]  # Drop batch dim → [84, N] or [N, 84]

        if output.ndim != 2:
            logger.error(
                "Unexpected output rank %d, shape=%s — skipping frame.",
                output.ndim,
                output.shape,
            )
            return []

        expected_features = 4 + _COCO_NUM_CLASSES  # 84
        rows, cols = output.shape

        if cols == expected_features and rows != expected_features:
            anchors = output                    # already [N, 84]
        elif rows == expected_features and cols != expected_features:
            anchors = output.T                  # [84, N] → [N, 84]
        elif rows == expected_features and cols == expected_features:
            anchors = output.T                  # square: assume column-major (YOLOv8 default)
        else:
            logger.error(
                "Output shape %s has no axis equal to %d — "
                "model may not be YOLOv8n COCO. Skipping frame.",
                output.shape,
                expected_features,
            )
            return []

        scale_x = orig_w / self._input_w
        scale_y = orig_h / self._input_h
        frame_third = orig_w // 3

        n_raw = anchors.shape[0]
        n_after_class = 0
        n_after_conf = 0
        candidates: List[Detection] = []

        for i in range(n_raw):
            row = anchors[i]
            class_scores = row[4:]
            class_id = int(np.argmax(class_scores))

            # Whitelist filter first — cheapest rejection, done before confidence math.
            if class_id not in config.OBJECT_CLASSES_OF_INTEREST:
                continue
            n_after_class += 1

            class_name = config.OBJECT_CLASSES_OF_INTEREST[class_id]
            score = float(class_scores[class_id])

            threshold = config.OBJECT_CONFIDENCE_BY_CLASS.get(
                class_name, config.OBJECT_CONFIDENCE_DEFAULT
            )
            if score < threshold:
                continue
            n_after_conf += 1

            cx = float(row[0])
            cy = float(row[1])
            w = float(row[2])
            h = float(row[3])

            x1 = int((cx - w * 0.5) * scale_x)
            y1 = int((cy - h * 0.5) * scale_y)
            x2 = int((cx + w * 0.5) * scale_x)
            y2 = int((cy + h * 0.5) * scale_y)

            x1 = max(0, min(x1, orig_w - 1))
            y1 = max(0, min(y1, orig_h - 1))
            x2 = max(0, min(x2, orig_w - 1))
            y2 = max(0, min(y2, orig_h - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            if center_x < frame_third:
                position = "left"
            elif center_x < 2 * frame_third:
                position = "center"
            else:
                position = "right"

            det = Detection(
                bbox=(x1, y1, x2, y2),
                class_id=class_id,
                class_name=class_name,
                confidence=round(score, 4),
                center_x=center_x,
                center_y=center_y,
                position=position,
            )
            det.estimated_distance_cm = estimate_distance_cm(det, orig_h)

            logger.debug(
                "  [%04d] %-14s conf=%.2f  pos=%-6s  dist=%s cm  bbox=(%d,%d,%d,%d)",
                i,
                class_name,
                score,
                position,
                f"{det.estimated_distance_cm:.0f}"
                if det.estimated_distance_cm is not None
                else "n/a",
                x1, y1, x2, y2,
            )
            candidates.append(det)

        result = _apply_nms(candidates, config.OBJECT_IOU_THRESHOLD)

        logger.debug(
            "Detections: raw=%d  after_class=%d  after_conf=%d  pre_nms=%d  post_nms=%d",
            n_raw,
            n_after_class,
            n_after_conf,
            len(candidates),
            len(result),
        )

        return result
