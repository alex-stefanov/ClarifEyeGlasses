"""
ClarifEye Traffic Light Detector — TFLite Inference Module

Loads a custom-trained YOLOv8n TFLite model (float32 or INT8-quantized) and
runs inference on RGB frames from the camera module.  The full pipeline is:

    resize → quantise-or-normalise → invoke → dequantise → parse → NMS

YOLOv8 TFLite output can arrive in two orientations:
  * ``[1, 4+nc, 8400]``  columns are anchors  (needs transpose)
  * ``[1, 8400, 4+nc]``  rows are anchors     (used directly)

Both are normalised internally to ``[anchors, 4+nc]`` where the first four
columns are ``[cx, cy, w, h]`` in model-input pixel space and the remaining
*nc* columns are sigmoid-activated class probabilities.

NMS is implemented from scratch with pure NumPy/Python so the module has no
dependency on cv2.dnn, torchvision, or any library not available on the Pi.
"""
import logging
import os
from typing import List, Optional, Tuple

import numpy as np

try:
    from .. import config
except ImportError:
    import config  # type: ignore[no-redef]

try:
    import cv2  # type: ignore[import-untyped]
    _OPENCV_AVAILABLE = True
except ImportError:
    _OPENCV_AVAILABLE = False

logger = logging.getLogger("clarifeye.ai.traffic_light")

try:
    from .detection import Detection
except ImportError:
    try:
        from ai.detection import Detection  # type: ignore[no-redef]
    except ImportError:
        from detection import Detection  # type: ignore[no-redef]


# ─── Pure-NumPy NMS ───────────────────────────────────────────────────────────

def _compute_iou(
    box_a: Tuple[int, int, int, int],
    box_b: Tuple[int, int, int, int],
) -> float:
    """
    Compute Intersection-over-Union for two axis-aligned bounding boxes.

    Args:
        box_a: ``(x1, y1, x2, y2)`` for the first box.
        box_b: ``(x1, y1, x2, y2)`` for the second box.

    Returns:
        IoU in the range ``[0.0, 1.0]``.
    """
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
    detections: List[Detection],
    iou_threshold: float,
) -> List[Detection]:
    """
    Greedy Non-Maximum Suppression.

    Iterates detections sorted by descending confidence.  For each kept
    detection every remaining candidate whose IoU with it exceeds
    *iou_threshold* is discarded.

    Args:
        detections:    Unsorted list of candidate detections.
        iou_threshold: Suppress boxes with IoU above this value.

    Returns:
        Filtered list, still sorted by descending confidence.
    """
    if not detections:
        return []

    ordered = sorted(detections, key=lambda d: d.confidence, reverse=True)
    kept: List[Detection] = []

    while ordered:
        best = ordered[0]
        kept.append(best)
        ordered = [
            d for d in ordered[1:]
            if _compute_iou(best.bbox, d.bbox) <= iou_threshold
        ]

    return kept


# ─── Detector ─────────────────────────────────────────────────────────────────

class TrafficLightDetector:
    """
    TFLite inference wrapper for the YOLOv8n traffic light model.

    The model is expected at ``config.TRAFFIC_LIGHT_MODEL_PATH``.  If the file
    is absent (e.g. before training) the detector stays inert and
    :meth:`detect` returns an empty list — it never raises.

    Quantisation handling
    ~~~~~~~~~~~~~~~~~~~~~
    * ``float32`` input  → frame divided by 255.0.
    * ``uint8`` input    → frame kept as-is in ``[0, 255]``.
    * ``int8`` input     → frame normalised then shifted using the tensor's
      stored ``(scale, zero_point)`` quantisation parameters.
    * ``int8`` output    → dequantised to float32 before parsing.
    """

    def __init__(
        self,
        model_path: str = config.TRAFFIC_LIGHT_MODEL_PATH,
    ) -> None:
        """
        Load the TFLite model and allocate interpreter tensors.

        Args:
            model_path: Absolute path to the ``.tflite`` model file.
        """
        self._model_path: str = model_path
        self._interpreter: Optional[object] = None
        self._input_details: Optional[list] = None
        self._output_details: Optional[list] = None
        self._input_h: int = 640
        self._input_w: int = 640
        self._input_dtype: type = np.float32
        self._num_classes: int = len(config.TRAFFIC_LIGHT_CLASSES)
        self._available: bool = False

        if not os.path.exists(model_path):
            logger.error(
                "TFLite model not found: %s — run scripts/train_traffic_light.py "
                "first, then copy the .tflite file to clarifeye/models/.",
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
            out_shape = self._output_details[0]["shape"]
            self._input_h = int(in_shape[1])
            self._input_w = int(in_shape[2])
            self._input_dtype = self._input_details[0]["dtype"]

            logger.info(
                "TrafficLightDetector ready  model=%s  input=%s (%s)  output=%s",
                os.path.basename(model_path),
                tuple(in_shape),
                self._input_dtype.__name__,
                tuple(out_shape),
            )
            self._available = True

        except Exception as exc:
            logger.error("TFLite interpreter initialisation failed: %s", exc)

    # ── Public API ─────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run inference on *frame* and return NMS-filtered detections.

        The frame is resized to the model's native input size internally;
        all returned bounding boxes are in the *original* frame's coordinate
        space.

        Args:
            frame: Full-resolution RGB ``numpy.ndarray`` (any size).

        Returns:
            List of :class:`Detection` objects sorted by descending
            confidence.  Returns ``[]`` on any error.
        """
        if not self._available or self._interpreter is None:
            return []

        if not _OPENCV_AVAILABLE:
            logger.error("OpenCV unavailable — cannot resize frame for inference.")
            return []

        try:
            orig_h, orig_w = frame.shape[:2]

            input_tensor = self._preprocess(frame)
            self._interpreter.set_tensor(  # type: ignore[union-attr]
                self._input_details[0]["index"], input_tensor
            )
            self._interpreter.invoke()  # type: ignore[union-attr]

            raw_output = self._interpreter.get_tensor(  # type: ignore[union-attr]
                self._output_details[0]["index"]
            )
            output_f32 = self._dequantize_output(raw_output)

            detections = self._parse_output(output_f32, orig_h, orig_w)
            return _apply_nms(detections, config.TRAFFIC_LIGHT_IOU_THRESHOLD)

        except Exception as exc:
            logger.error("detect() error: %s", exc)
            return []

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Resize *frame* to the model input size and apply the correct
        normalisation / quantisation for the model's input ``dtype``.

        Returns:
            4-D array ``[1, H, W, 3]`` ready to feed into the interpreter.
        """
        resized = cv2.resize(frame, (self._input_w, self._input_h))

        if self._input_dtype == np.uint8:
            tensor = resized.astype(np.uint8)

        elif self._input_dtype == np.int8:
            quant = self._input_details[0]["quantization"]
            scale: float = float(quant[0]) if float(quant[0]) != 0.0 else 1.0 / 255.0
            zero_point: int = int(quant[1])
            normalized = resized.astype(np.float32) / 255.0
            tensor = np.clip(
                np.round(normalized / scale) + zero_point,
                -128,
                127,
            ).astype(np.int8)

        else:  # np.float32 — default for non-quantized and hybrid models
            tensor = resized.astype(np.float32) / 255.0

        return np.expand_dims(tensor, axis=0)

    def _dequantize_output(self, output_data: np.ndarray) -> np.ndarray:
        """
        Convert an INT8 output tensor to float32 using the stored quantisation
        parameters ``(scale, zero_point)``.  Float32 tensors pass through with
        only a dtype cast.

        Returns:
            float32 array with the same shape as *output_data*.
        """
        if output_data.dtype in (np.int8, np.uint8):
            quant = self._output_details[0]["quantization"]
            scale: float = float(quant[0]) if float(quant[0]) != 0.0 else 1.0
            zero_point: int = int(quant[1])
            return (output_data.astype(np.float32) - zero_point) * scale

        return output_data.astype(np.float32)

    def _parse_output(
        self,
        output_data: np.ndarray,
        orig_h: int,
        orig_w: int,
    ) -> List[Detection]:
        """
        Convert the raw YOLOv8 output tensor to :class:`Detection` objects
        scaled to the original frame dimensions.

        Handles both output orientations after stripping the batch dimension:

        * ``[4+nc, 8400]``  → transposed to ``[8400, 4+nc]``
        * ``[8400, 4+nc]``  → used directly

        Each row: ``[cx, cy, w, h, p_cls0, p_cls1, …, p_clsN-1]``
        where coordinates are in **model-input pixel space** (0–640).

        Args:
            output_data: float32 array of shape ``[1, ?, ?]``.
            orig_h:      Height of the original frame in pixels.
            orig_w:      Width of the original frame in pixels.

        Returns:
            Unfiltered list of detections passing the confidence threshold.
        """
        nc = self._num_classes
        output = output_data[0]  # Drop batch dimension.

        # Normalise to [anchors, 4+nc] regardless of export orientation.
        if output.shape[0] == 4 + nc:
            # Shape [4+nc, 8400] → transpose to [8400, 4+nc].
            output = output.T

        scale_x = orig_w / self._input_w
        scale_y = orig_h / self._input_h

        detections: List[Detection] = []

        for row in output:
            cx, cy, w, h = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            class_scores = row[4: 4 + nc]

            class_id = int(np.argmax(class_scores))
            confidence = float(class_scores[class_id])

            if confidence < config.TRAFFIC_LIGHT_CONFIDENCE_THRESHOLD:
                continue

            # Convert model-space cx/cy/w/h → original-frame x1/y1/x2/y2.
            x1 = int((cx - w * 0.5) * scale_x)
            y1 = int((cy - h * 0.5) * scale_y)
            x2 = int((cx + w * 0.5) * scale_x)
            y2 = int((cy + h * 0.5) * scale_y)

            # Clamp to valid frame bounds.
            x1 = max(0, min(x1, orig_w - 1))
            y1 = max(0, min(y1, orig_h - 1))
            x2 = max(0, min(x2, orig_w - 1))
            y2 = max(0, min(y2, orig_h - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            class_name = config.TRAFFIC_LIGHT_CLASSES.get(class_id, "unknown")
            detections.append(
                Detection(
                    bbox=(x1, y1, x2, y2),
                    class_id=class_id,
                    class_name=class_name,
                    confidence=round(confidence, 4),
                    center_x=(x1 + x2) // 2,
                    center_y=(y1 + y2) // 2,
                )
            )

        return detections
