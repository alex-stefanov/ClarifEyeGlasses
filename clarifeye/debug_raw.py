import sys
sys.path.insert(0, '/home/alex/clarifeye')
import cv2
import numpy as np
from picamera2 import Picamera2
from ai_edge_litert.interpreter import Interpreter
import config
import time

cam = Picamera2()
cam.configure(cam.create_preview_configuration(main={"size": (640, 640), "format": "RGB888"}))
cam.start()
time.sleep(1)

interp = Interpreter(model_path=config.TRAFFIC_LIGHT_MODEL_PATH)
interp.allocate_tensors()
inp = interp.get_input_details()[0]
out = interp.get_output_details()[0]

print(f"Input: shape={inp['shape']} dtype={inp['dtype']}")
print(f"Output: shape={out['shape']} dtype={out['dtype']}")
print()

print("Point camera at a traffic light image on your phone...")
time.sleep(3)

frame = cam.capture_array()
resized = cv2.resize(frame, (640, 640))
tensor = resized.astype(np.float32) / 255.0
tensor = np.expand_dims(tensor, axis=0)

interp.set_tensor(inp['index'], tensor)
interp.invoke()
raw = interp.get_tensor(out['index'])

print(f"Raw output shape: {raw.shape}")
print(f"Raw output min={raw.min():.4f} max={raw.max():.4f} mean={raw.mean():.4f}")

data = raw[0]
nc = 3
if data.shape[0] == 4 + nc:
    data = data.T

print(f"Parsed shape: {data.shape}")
print()

all_max_scores = np.max(data[:, 4:4+nc], axis=1)
top_indices = np.argsort(all_max_scores)[-10:][::-1]

print("Top 10 highest confidence anchors:")
for idx in top_indices:
    scores = data[idx, 4:4+nc]
    best_cls = np.argmax(scores)
    cls_name = config.TRAFFIC_LIGHT_CLASSES.get(best_cls, "?")
    print(f"  anchor {idx}: {cls_name} score={scores[best_cls]:.4f}  all_scores={scores}")

cam.stop()
cam.close()
