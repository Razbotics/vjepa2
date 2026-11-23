"""Live camera preview in a browser, for aiming the camera on the headless desktop.

Serves an MJPEG stream over HTTP. It binds to localhost only; view it from the laptop through an SSH
tunnel, which `local/remote/camera_preview.sh` sets up:

    local/remote/camera_preview.sh            # then open http://localhost:8765
    python -m local.camera_preview --port 8765  # on the desktop itself

Overlay:
  yellow square   the centre crop the V-JEPA encoder actually sees (`preprocess` in live_camera.py)
  cross           frame centre
  text            mean brightness (0-255) and the share of blown-out pixels

The camera can only be opened by one program. Stop the preview (Ctrl-C) before running arm_collect.py
or the live_* scripts.
"""

import argparse
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

from local.camera import Camera, set_v4l2_controls

PAGE = b"""<!doctype html><meta charset="utf-8"><title>Camera preview</title>
<style>body{margin:0;background:#111;color:#ddd;font:14px system-ui;text-align:center}
img{max-width:100vw;max-height:92vh;display:block;margin:0 auto}p{margin:6px}</style>
<img src="/stream"><p>Yellow square: what the encoder sees. Stop this preview before collecting data.</p>
"""


def annotate(frame):
    h, w = frame.shape[:2]
    s = min(h, w)
    x0, y0 = (w - s) // 2, (h - s) // 2
    mean = float(frame.mean())
    blown = 100.0 * float((frame.max(axis=2) > 250).mean())
    out = frame.copy()
    cv2.rectangle(out, (x0, y0), (x0 + s - 1, y0 + s - 1), (0, 255, 255), 2)
    cv2.drawMarker(out, (w // 2, h // 2), (0, 255, 255), cv2.MARKER_CROSS, 24, 1)
    label = f"brightness {mean:.0f}/255  blown-out {blown:.1f}%"
    cv2.putText(out, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(out, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", default="/dev/video0")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--stream-fps", type=float, default=15.0)
    ap.add_argument("--stream-width", type=int, default=960, help="downscale the stream to this width")
    ap.add_argument("--quality", type=int, default=75)
    ap.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to serve the whole network")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--mains-hz", type=int, default=50, choices=[50, 60])
    ap.add_argument("--no-overlay", action="store_true")
    args = ap.parse_args()

    set_v4l2_controls(
        str(args.camera),
        {
            "exposure_dynamic_framerate": 0,
            "power_line_frequency": 1 if args.mains_hz == 50 else 2,
            "auto_exposure": 3,
        },
    )
    cam = Camera(args.camera, args.width, args.height, args.fps, maxlen=2)
    state = {"jpg": None, "n": 0}
    cond = threading.Condition()

    def encode_loop():
        last_t = None
        while not cam.stop:
            item = cam.latest(1, 1)
            if item is None or item[0][-1] == last_t:
                time.sleep(0.005)
                continue
            last_t = item[0][-1]
            frame = item[1][0]
            if not args.no_overlay:
                frame = annotate(frame)
            if frame.shape[1] > args.stream_width:
                h = round(frame.shape[0] * args.stream_width / frame.shape[1])
                frame = cv2.resize(frame, (args.stream_width, h), interpolation=cv2.INTER_AREA)
            ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
            if ok:
                with cond:
                    state["jpg"], state["n"] = jpg.tobytes(), state["n"] + 1
                    cond.notify_all()
            time.sleep(max(0.0, 1.0 / args.stream_fps - 0.005))

    threading.Thread(target=encode_loop, daemon=True).start()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path == "/":
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(PAGE)))
                self.end_headers()
                self.wfile.write(PAGE)
            elif self.path == "/snapshot.jpg":
                with cond:
                    cond.wait_for(lambda: state["jpg"] is not None, timeout=5)
                    jpg = state["jpg"]
                self.send_response(200 if jpg else 503)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(jpg or b"")
            elif self.path == "/stream":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                seen = -1
                try:
                    while True:
                        with cond:
                            if not cond.wait_for(lambda: state["n"] != seen, timeout=5):
                                return  # camera stopped delivering
                            jpg, seen = state["jpg"], state["n"]
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode() + jpg + b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send_error(404)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    print(f"camera preview on http://{args.host}:{args.port}  (Ctrl-C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        cam.close()
        print("preview stopped, camera released")


if __name__ == "__main__":
    main()
