import sys
sys.path.insert(0, '/home/alex/clarifeye')
import cv2
import numpy as np
from picamera2 import Picamera2
from ai_edge_litert.interpreter import Interpreter
import config
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

cam = Picamera2()
cam.configure(cam.create_preview_configuration(main={"size": (640, 640), "format": "RGB888"}))
cam.start()
time.sleep(2)

interp = Interpreter(model_path=config.TRAFFIC_LIGHT_MODEL_PATH)
interp.allocate_tensors()
inp = interp.get_input_details()[0]
out = interp.get_output_details()[0]

def get_annotated_frame():
    frame = cam.capture_array()
    tensor = np.expand_dims(frame.astype(np.float32) / 255.0, axis=0)
    interp.set_tensor(inp['index'], tensor)
    interp.invoke()
    raw = interp.get_tensor(out['index'])[0]
    if raw.shape[0] == 7:
        raw = raw.T
    scores = np.max(raw[:, 4:7], axis=1)
    top5 = np.argsort(scores)[-5:][::-1]

    display = frame.copy()
    for idx in top5:
        row = raw[idx]
        cx, cy, w, h = row[0], row[1], row[2], row[3]
        cls = int(np.argmax(row[4:7]))
        score = float(row[4 + cls])
        name = config.TRAFFIC_LIGHT_CLASSES.get(cls, "?")
        x1 = int(cx - w/2)
        y1 = int(cy - h/2)
        x2 = int(cx + w/2)
        y2 = int(cy + h/2)
        color = {"red": (255,0,0), "yellow": (255,255,0), "green": (0,255,0)}.get(name, (255,255,255))
        if score > 0.1:
            cv2.rectangle(display, (x1,y1), (x2,y2), color, 2)
            cv2.putText(display, f"{name} {score:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    info = f"Top score: {scores[top5[0]]:.4f}"
    cv2.putText(display, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return display

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            html = '<html><body style="margin:0;background:#000;text-align:center">'
            html += '<img src="/frame" id="f" style="max-height:100vh">'
            html += '<script>setInterval(function(){document.getElementById("f").src="/frame?t="+Date.now()},1000)</script>'
            html += '</body></html>'
            self.wfile.write(html.encode())
        elif self.path.startswith('/frame'):
            try:
                display = get_annotated_frame()
                result = cv2.imencode('.jpg', cv2.cvtColor(display, cv2.COLOR_RGB2BGR))
                jpg = result[1]
                self.send_response(200)
                self.send_header('Content-type', 'image/jpeg')
                self.send_header('Content-length', str(len(jpg)))
                self.end_headers()
                self.wfile.write(jpg.tobytes())
            except Exception as e:
                print(f"Error: {e}")
                self.send_response(500)
                self.end_headers()
    def log_message(self, format, *args):
        pass

import subprocess
ip = subprocess.check_output(['hostname', '-I']).decode().strip().split()[0]
print(f"Open: http://{ip}:8080")
print("Shows live camera with detection boxes and scores")
HTTPServer(('0.0.0.0', 8080), Handler).serve_forever()
