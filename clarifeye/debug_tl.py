import sys
sys.path.insert(0, '/home/alex/clarifeye')
import cv2
import numpy as np
from picamera2 import Picamera2
from ai.traffic_light_detector import TrafficLightDetector
import time

cam = Picamera2()
cam.configure(cam.create_preview_configuration(main={"size": (640, 640), "format": "RGB888"}))
cam.start()
time.sleep(1)

detector = TrafficLightDetector()

print("Capturing 10 frames, pointing camera at a traffic light image...")
print()

for i in range(10):
    frame = cam.capture_array()
    mean_brightness = np.mean(frame)
    print(f"Frame {i+1}: shape={frame.shape} dtype={frame.dtype} brightness={mean_brightness:.0f}")

    detections = detector.detect(frame)
    if detections:
        for d in detections:
            print(f"  DETECTED: {d.class_name} confidence={d.confidence:.2f} bbox={d.bbox}")
    else:
        print("  No detections")

    time.sleep(1)

cam.stop()
cam.close()
