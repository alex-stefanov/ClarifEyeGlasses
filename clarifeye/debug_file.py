import sys
sys.path.insert(0, '/home/alex/clarifeye')
import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import config

# Download a test image
import urllib.request
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/7/71/Aamp-Gr%C3%BCn.svg/220px-Aamp-Gr%C3%BCn.svg.png"
try:
    urllib.request.urlretrieve(url, "/tmp/tl_test.png")
    img = cv2.imread("/tmp/tl_test.png")
except:
    # If no internet, create a fake green traffic light
    img = np.zeros((300, 150, 3), dtype=np.uint8)
    cv2.circle(img, (75, 220), 30, (0, 255, 0), -1)
    print("No internet - using fake green circle")

if img is None:
    img = np.zeros((300, 150, 3), dtype=np.uint8)
    cv2.circle(img, (75, 220), 30, (0, 255, 0), -1)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(f"Test image shape: {img_rgb.shape}")

interp = Interpreter(model_path=config.TRAFFIC_LIGHT_MODEL_PATH)
interp.allocate_tensors()
inp = interp.get_input_details()[0]
out = interp.get_output_details()[0]

resized = cv2.resize(img_rgb, (640, 640))
tensor = np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)

interp.set_tensor(inp['index'], tensor)
interp.invoke()
raw = interp.get_tensor(out['index'])[0].T

scores = np.max(raw[:, 4:7], axis=1)
top = np.argsort(scores)[-5:][::-1]

print("\nTop 5 scores on test image:")
for i in top:
    cls = np.argmax(raw[i, 4:7])
    name = config.TRAFFIC_LIGHT_CLASSES.get(cls, "?")
    print(f"  {name}: {scores[i]:.4f}")

# Now test with camera frame
from picamera2 import Picamera2
import time
cam = Picamera2()
cam.configure(cam.create_preview_configuration(main={"size": (640, 640), "format": "RGB888"}))
cam.start()
time.sleep(2)
frame = cam.capture_array()

# Save the frame so we can look at it
cv2.imwrite("/tmp/camera_frame.jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
print(f"\nCamera frame saved to /tmp/camera_frame.jpg")
print(f"Camera R={frame[:,:,0].mean():.0f} G={frame[:,:,1].mean():.0f} B={frame[:,:,2].mean():.0f}")

tensor2 = np.expand_dims(frame.astype(np.float32) / 255.0, axis=0)
interp.set_tensor(inp['index'], tensor2)
interp.invoke()
raw2 = interp.get_tensor(out['index'])[0].T

scores2 = np.max(raw2[:, 4:7], axis=1)
top2 = np.argsort(scores2)[-5:][::-1]

print("\nTop 5 scores on camera frame:")
for i in top2:
    cls = np.argmax(raw2[i, 4:7])
    name = config.TRAFFIC_LIGHT_CLASSES.get(cls, "?")
    print(f"  {name}: {scores2[i]:.4f}")

cam.stop()
