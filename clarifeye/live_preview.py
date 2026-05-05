from picamera2 import Picamera2
from http.server import HTTPServer, BaseHTTPRequestHandler
import cv2
import time

cam = Picamera2()
cam.configure(cam.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"}))
cam.start()
time.sleep(1)
print("Camera started")

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            html = '<html><body style="margin:0;background:#000;text-align:center">'
            html += '<img src="/frame" id="f" style="max-height:100vh">'
            html += '<script>setInterval(function(){document.getElementById("f").src="/frame?t="+Date.now()},500)</script>'
            html += '</body></html>'
            self.wfile.write(html.encode())
        elif self.path.startswith('/frame'):
            try:
                frame = cam.capture_array()
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                result = cv2.imencode('.jpg', frame_bgr)
                jpg = result[1]
                self.send_response(200)
                self.send_header('Content-type', 'image/jpeg')
                self.send_header('Content-length', str(len(jpg)))
                self.end_headers()
                self.wfile.write(jpg.tobytes())
            except Exception as e:
                print(f"Frame error: {e}")
                self.send_response(500)
                self.end_headers()
    def log_message(self, format, *args):
        pass

import subprocess; ip = subprocess.check_output(['hostname', '-I']).decode().strip().split()[0]
print(f"Open in browser: http://{ip}:8080")
HTTPServer(('0.0.0.0', 8080), Handler).serve_forever()
