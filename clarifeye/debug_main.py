import sys
sys.path.insert(0, '/home/alex/clarifeye')
import numpy as np
import time
from hardware.camera import CameraModule
from ai.traffic_light_detector import TrafficLightDetector
from ai.color_verifier import ColorVerifier
import config

cam = CameraModule()
cam.start()
time.sleep(2)

detector = TrafficLightDetector()
verifier = ColorVerifier()

print("Point at traffic light image. Testing 20 frames...")
print()

for i in range(20):
    frame = cam.capture_frame()
    if frame is None:
        print(f"Frame {i}: None")
        continue

    print(f"Frame {i}: shape={frame.shape} R={frame[:,:,0].mean():.0f} G={frame[:,:,1].mean():.0f} B={frame[:,:,2].mean():.0f}")

    dets = detector.detect(frame)
    if dets:
        for d in dets:
            print(f"  DETECTED: {d.class_name} conf={d.confidence:.3f} bbox={d.bbox}")
            verified = verifier.verify_traffic_light_color(frame, d.bbox)
            print(f"  HSV verified: {verified}")
    else:
        print(f"  No detections")

    time.sleep(1)

cam.stop()
