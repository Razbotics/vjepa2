"""Webcam grabber and V4L2 control helper. No torch import, so it also loads in the LeRobot environment."""

import collections
import shutil
import subprocess
import threading
import time

import cv2
import numpy as np


def set_v4l2_controls(dev: str, controls: dict):
    """Apply V4L2 controls with v4l2-ctl. They are volatile (reset on replug/reboot), so do it every run."""
    if not dev.startswith("/dev/") or shutil.which("v4l2-ctl") is None:
        print("[camera] v4l2-ctl not available for this device; leaving camera controls at their defaults")
        return
    for k, v in controls.items():
        r = subprocess.run(["v4l2-ctl", "-d", dev, "-c", f"{k}={v}"], capture_output=True, text=True)
        status = "ok" if r.returncode == 0 else f"FAILED ({r.stderr.strip()})"
        print(f"[camera] {k}={v}: {status}")


class Camera:
    """Background grabber. Keeps the newest ``maxlen`` frames with their capture timestamps."""

    def __init__(self, dev, width, height, fps, maxlen):
        if isinstance(dev, str) and dev.startswith("/dev/"):
            self.cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        else:
            self.cap = cv2.VideoCapture(int(dev))
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open camera {dev}")
        # MJPG first: over USB 2 the C270 only reaches 30 fps at 720p with compression.
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        # Do not set CAP_PROP_BUFFERSIZE=1: with a single kernel buffer the C270 drops every second
        # frame (15 fps instead of 30). The grab thread below drains the queue, so frames stay fresh.
        self.buf = collections.deque(maxlen=maxlen)
        self.lock = threading.Lock()
        self.count = 0
        self.stop = False
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stop:
            ok, frame = self.cap.read()
            t = time.perf_counter()
            if not ok:
                self.error = "camera read failed"
                return
            with self.lock:
                self.buf.append((t, frame))
                self.count += 1

    def latest(self, n, stride):
        """Newest n frames spaced ``stride`` apart, oldest first, or None if not enough yet."""
        need = (n - 1) * stride + 1
        with self.lock:
            if len(self.buf) < need:
                return None
            items = list(self.buf)[-need::stride]
        return [t for t, _ in items], np.stack([f for _, f in items])

    def close(self):
        self.stop = True
        self.thread.join(timeout=2)
        self.cap.release()
