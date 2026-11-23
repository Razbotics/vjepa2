"""Live camera -> V-JEPA encoder loop with latency accounting.

Grabs frames from a webcam in a background thread, keeps a rolling window, and encodes
the most recent clip as fast as the device allows. Headless: prints one line per second
and a summary, no window. Meant for the GPU desktop with the Logitech C270.

    python -m local.live_camera                                   # ViT-L, 16 frames @256, 10 s
    python -m local.live_camera --model vjepa2_1_vit_base_384 --frames 8 --stride 4
    python -m local.live_camera --exposure 150 --seconds 30       # lock exposure (C270 units of 100 us)
    python -m local.live_camera --save local/runs/live_last.jpg

Reported per window:
  encode   encoder forward time
  prep     uint8 frames -> normalized [1, C, T, H, W] on the device
  age      now - capture time of the newest frame in the clip, measured after the encoder
           returns. This is the staleness a planner would act on.
  d_ref    1 - cosine similarity between this clip's mean-pooled features and the first
           clip's. Hold the camera still and it stays near 0; point it elsewhere and it rises.
  d_prev   same, against the previous clip.
"""

import argparse
import collections
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from local.camera import Camera, set_v4l2_controls  # noqa: F401 (re-exported)
from local.vjepa_local import IMAGENET_MEAN, IMAGENET_STD, SPECS, load_model, n_params, peak_mem_gb, pick_device, sync


def preprocess(frames_bgr: np.ndarray, size: int, device, mean, std):
    """[T, H, W, 3] uint8 BGR -> [1, 3, T, size, size] float, center-cropped square, ImageNet-normalized."""
    x = torch.from_numpy(frames_bgr).to(device)  # T H W C
    x = x.flip(-1).permute(0, 3, 1, 2).float().div_(255.0)  # BGR->RGB, T C H W
    h, w = x.shape[-2:]
    s = min(h, w)
    top, left = (h - s) // 2, (w - s) // 2
    x = x[..., top : top + s, left : left + s]
    x = F.interpolate(x, size=(size, size), mode="bilinear", antialias=True, align_corners=False)
    x = (x - mean) / std
    return x.permute(1, 0, 2, 3).unsqueeze(0)  # 1 C T H W


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vjepa2_1_vit_large_384", choices=sorted(SPECS))
    ap.add_argument("--camera", default="/dev/video0", help="V4L2 device path, or an integer index (e.g. 0 on macOS)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--frames", type=int, default=16, help="frames per clip (even: the tubelet is 2 frames)")
    ap.add_argument("--stride", type=int, default=2, help="take every Nth camera frame (2 at 30 fps = 15 fps clip)")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    ap.add_argument(
        "--exposure",
        type=int,
        default=None,
        help="lock manual exposure, C270 units of 100 us; above ~330 the camera can no longer hold 30 fps",
    )
    ap.add_argument("--gain", type=int, default=None, help="manual gain 0-255 (only meaningful with --exposure)")
    ap.add_argument("--mains-hz", type=int, default=50, choices=[50, 60], help="anti-flicker filter")
    ap.add_argument("--save", default=None, help="write the last captured frame to this jpg")
    args = ap.parse_args()
    if args.frames % 2:
        sys.exit("--frames must be even")

    device = pick_device(args.device)
    dtype = getattr(torch, args.dtype)
    span = (args.frames - 1) * args.stride / args.fps
    print(f"device={device} dtype={dtype} clip={args.frames} frames, stride {args.stride} ({span:.2f} s of video)")

    # Fixed frame rate matters: the model assumes a constant time step between frames.
    controls = {"exposure_dynamic_framerate": 0, "power_line_frequency": 1 if args.mains_hz == 50 else 2}
    if args.exposure is not None:
        controls.update({"auto_exposure": 1, "exposure_time_absolute": args.exposure})
        if args.gain is not None:
            controls["gain"] = args.gain
    else:
        controls["auto_exposure"] = 3
    set_v4l2_controls(str(args.camera), controls)

    encoder, _ = load_model(args.model, device=device, num_frames=args.frames)
    print(f"encoder {n_params(encoder):.0f}M params")
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    cam = Camera(args.camera, args.width, args.height, args.fps, maxlen=(args.frames - 1) * args.stride + 8)
    need = (args.frames - 1) * args.stride + 1
    t_wait = time.perf_counter()
    while cam.latest(args.frames, args.stride) is None:
        if cam.error or time.perf_counter() - t_wait > 10:
            cam.close()
            sys.exit(f"camera did not deliver {need} frames: {cam.error or 'timeout'}")
        time.sleep(0.01)
    ts, frames = cam.latest(args.frames, args.stride)
    print(f"camera {frames.shape[2]}x{frames.shape[1]}, first clip ready after {time.perf_counter() - t_wait:.2f} s")

    autocast = torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype != torch.float32)
    stats = collections.defaultdict(list)
    ref = prev = None
    last_newest = None
    t_start = t_report = time.perf_counter()
    frames_at_start = cam.count
    n_dup = 0
    print(
        f"{'t':>5} {'loop Hz':>8} {'cam fps':>8} {'prep ms':>8} {'enc ms':>7} {'age ms':>7} {'d_ref':>7} {'d_prev':>7}"
    )

    try:
        with torch.inference_mode(), autocast:
            # warm-up (cuDNN / kernel selection) is not counted
            for _ in range(2):
                encoder(preprocess(frames, args.size, device, mean, std))
            sync(device)
            t_start = t_report = time.perf_counter()
            frames_at_start = cam.count
            win = collections.defaultdict(list)
            while time.perf_counter() - t_start < args.seconds:
                if cam.error:
                    raise RuntimeError(cam.error)
                ts, frames = cam.latest(args.frames, args.stride)
                if ts[-1] == last_newest:  # encoder outran the camera; wait for a new frame
                    n_dup += 1
                    time.sleep(0.001)
                    continue
                last_newest = ts[-1]

                t0 = time.perf_counter()
                x = preprocess(frames, args.size, device, mean, std)
                sync(device)
                t1 = time.perf_counter()
                feats = encoder(x)
                sync(device)
                t2 = time.perf_counter()

                pooled = F.normalize(feats.float().mean(dim=1), dim=-1)[0]
                if ref is None:
                    ref = pooled
                d_ref = 1 - float(pooled @ ref)
                d_prev = 1 - float(pooled @ prev) if prev is not None else 0.0
                prev = pooled

                dts = np.diff(ts) * 1000
                row = {
                    "prep": (t1 - t0) * 1000,
                    "enc": (t2 - t1) * 1000,
                    "age": (t2 - ts[-1]) * 1000,
                    "d_ref": d_ref,
                    "d_prev": d_prev,
                    "frame_dt": float(dts.mean()),
                    "frame_dt_max": float(dts.max()),
                }
                for k, v in row.items():
                    stats[k].append(v)
                    win[k].append(v)

                now = time.perf_counter()
                if now - t_report >= 1.0:
                    cam_fps = (cam.count - frames_at_start) / (now - t_start)
                    print(
                        f"{now - t_start:5.1f} {len(win['enc']) / (now - t_report):8.1f} {cam_fps:8.1f} "
                        f"{np.mean(win['prep']):8.1f} {np.mean(win['enc']):7.1f} {np.mean(win['age']):7.1f} "
                        f"{np.mean(win['d_ref']):7.4f} {np.mean(win['d_prev']):7.4f}",
                        flush=True,
                    )
                    win = collections.defaultdict(list)
                    t_report = now
    finally:
        elapsed = time.perf_counter() - t_start
        cam_frames = cam.count - frames_at_start
        cam.close()

    n = len(stats["enc"])
    expected_dt = 1000 * args.stride / args.fps
    print("\n== summary")
    print(f"clips encoded        {n} in {elapsed:.1f} s = {n / elapsed:.1f} Hz (waited for a new frame {n_dup} times)")
    print(f"camera               {cam_frames / elapsed:.1f} fps delivered (asked {args.fps})")
    print(
        f"frame spacing in clip mean {np.mean(stats['frame_dt']):.1f} ms, "
        f"worst {np.max(stats['frame_dt_max']):.1f} ms (expected {expected_dt:.1f} ms)"
    )
    for k, label in [("prep", "preprocess"), ("enc", "encoder"), ("age", "newest-frame age")]:
        print(f"{label:<20} median {pct(stats[k], 50):6.1f} ms   p95 {pct(stats[k], 95):6.1f} ms")
    print(f"d_ref                min {np.min(stats['d_ref']):.4f}  max {np.max(stats['d_ref']):.4f}")
    print(f"d_prev               median {pct(stats['d_prev'], 50):.4f}  max {np.max(stats['d_prev']):.4f}")
    print(f"tokens/clip          {feats.shape[1]} x {feats.shape[2]}   peak device mem {peak_mem_gb(device):.2f} GB")
    if args.save:
        cv2.imwrite(args.save, frames[-1])
        print(f"saved last frame to {args.save}")


if __name__ == "__main__":
    main()
