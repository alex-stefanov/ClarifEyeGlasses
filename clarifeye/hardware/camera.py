"""
ClarifEye Camera Module

Wraps picamera2 (Pi Camera Module 3) with an OpenCV VideoCapture fallback
for development on non-Pi hardware.  Always returns frames as RGB numpy arrays.
"""
import logging
from typing import Optional

import numpy as np

try:
    from .. import config
except ImportError:
    import config  # type: ignore[no-redef]

logger = logging.getLogger("clarifeye.hardware.camera")

_PICAMERA2_AVAILABLE = False
try:
    from picamera2 import Picamera2
    _PICAMERA2_AVAILABLE = True
except ImportError:
    logger.warning("picamera2 not available — will attempt OpenCV fallback.")

_OPENCV_AVAILABLE = False
try:
    import cv2
    _OPENCV_AVAILABLE = True
except ImportError:
    logger.warning("opencv-python not available.")


class CameraModule:
    def __init__(self):
        self._camera = None
        self._cap = None
        self._use_picamera2 = False
        self._running = False

        if _PICAMERA2_AVAILABLE:
            self._init_picamera2()
        if not self._use_picamera2:
            self._init_opencv()

    def _init_picamera2(self):
        try:
            self._camera = Picamera2()
            cam_config = self._camera.create_preview_configuration(
                main={"size": config.CAMERA_RESOLUTION, "format": config.CAMERA_FORMAT},
                buffer_count=config.CAMERA_BUFFER_COUNT,
            )
            self._camera.configure(cam_config)
            self._use_picamera2 = True
            logger.info("Camera backend: picamera2  resolution=%s  format=%s  buffers=%d.",
                config.CAMERA_RESOLUTION, config.CAMERA_FORMAT, config.CAMERA_BUFFER_COUNT)
        except Exception as exc:
            logger.warning("picamera2 initialisation failed (%s) — falling back to OpenCV.", exc)
            self._camera = None
            self._use_picamera2 = False

    def _init_opencv(self):
        if not _OPENCV_AVAILABLE:
            logger.error("Neither picamera2 nor OpenCV is available.")
            return
        try:
            self._cap = cv2.VideoCapture(0)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_RESOLUTION[0])
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_RESOLUTION[1])
            self._cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FRAMERATE)
            if not self._cap.isOpened():
                logger.error("OpenCV VideoCapture(0) failed to open.")
                self._cap = None
                return
            logger.info("Camera backend: OpenCV VideoCapture(0)  resolution=%s.", config.CAMERA_RESOLUTION)
        except Exception as exc:
            logger.error("OpenCV VideoCapture initialisation failed: %s", exc)
            self._cap = None

    def start(self):
        if self._running:
            return
        try:
            if self._use_picamera2 and self._camera is not None:
                self._camera.start()
                self._running = True
                logger.info("picamera2 capture started.")
            elif self._cap is not None:
                if self._cap.isOpened():
                    self._running = True
                    logger.info("OpenCV VideoCapture started.")
                else:
                    logger.error("OpenCV VideoCapture is not open — cannot start.")
        except Exception as exc:
            logger.error("Camera start failed: %s", exc)

    def capture_frame(self):
        if not self._running:
            logger.warning("capture_frame() called before start() — returning None.")
            return None
        try:
            if self._use_picamera2 and self._camera is not None:
                return self._camera.capture_array()

            if self._cap is not None:
                ret, frame = self._cap.read()
                if not ret or frame is None:
                    logger.warning("OpenCV frame read failed.")
                    return None
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception as exc:
            logger.error("capture_frame error: %s", exc)
        return None

    def stop(self):
        self._running = False
        try:
            if self._use_picamera2 and self._camera is not None:
                self._camera.stop()
                self._camera.close()
                logger.info("picamera2 stopped and closed.")
        except Exception as exc:
            logger.error("picamera2 stop error: %s", exc)
        try:
            if self._cap is not None:
                self._cap.release()
                logger.info("OpenCV VideoCapture released.")
        except Exception as exc:
            logger.error("OpenCV VideoCapture release error: %s", exc)
