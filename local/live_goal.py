"""Live image-goal readout with V-JEPA 2-AC: camera in, goal energy and best candidate action out.

Encodes a goal image once with the ViT-g encoder, then for every new camera frame:
  E_goal   mean |z_now - z_goal| over layer-normed patch tokens. This is the quantity the
           paper's planner minimizes. It should fall as the view approaches the goal.
           Scale on the C270: 0 for the identical frame, ~0.46 for the same static scene a
           moment later (sensor noise alone moves ViT-g patch tokens that much), ~0.71 for
           an unrelated scene. So the usable range is narrow, but it is stable to +/-0.005.
  d_pool   1 - cosine similarity of the mean-pooled tokens. Far better separated than
           E_goal (~0.005 same scene, 0.2-0.3 different view or unrelated scene) because pooling averages the
           per-patch noise away, at the cost of spatial detail. Good "am I there" signal.
  actions  rolls the action-conditioned predictor one step for a small set of candidate
           end-effector deltas (stay, +/-x, +/-y, +/-z) and reports the one whose predicted
           next latent is closest to the goal, plus the spread across candidates.

    python -m local.live_goal                               # goal = first camera frame
    python -m local.live_goal --goal local/runs/goal.jpg    # goal from a saved image
    python -m local.live_goal --save-goal local/runs/goal.jpg --seconds 20

Typical use by hand: start it pointed at the goal view (or save one), move the camera away
and back, and watch E_goal rise and fall.

Caveat: 2-AC was trained on DROID, a Franka arm seen from a fixed exocentric camera, with
actions as end-effector deltas. A hand-held or drone-mounted webcam is out of distribution
for the *action* readout, and the arm state fed to the predictor here is a fixed pose from
the bundled Franka example. Treat E_goal as a real signal and the best-action column as a
latency and plumbing check until there is a predictor trained on our own action space.
"""

import argparse
import collections
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from local.live_camera import Camera, pct, preprocess, set_v4l2_controls
from local.vjepa_local import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    REPO_ROOT,
    load_model,
    n_params,
    peak_mem_gb,
    pick_device,
    sync,
)

TRAJ = os.path.join(REPO_ROOT, "notebooks", "franka_example_traj.npz")


def candidate_actions(step: float, device, dtype):
    """[S, 1, 7] end-effector deltas (xyz, rpy, gripper): stay plus one step along each +/- axis."""
    labels, rows = ["stay"], [[0.0] * 7]
    for axis, name in enumerate("xyz"):
        for sign, s in ((1, "+"), (-1, "-")):
            row = [0.0] * 7
            row[axis] = sign * step
            rows.append(row)
            labels.append(f"{s}{name}")
    return labels, torch.tensor(rows, device=device, dtype=dtype).unsqueeze(1)


def encode_frame(encoder, x):
    """x: [1, 3, 1, H, W] -> layer-normed tokens [1, N, D]. The single frame fills a 2-frame tubelet."""
    h = encoder(x.repeat(1, 1, 2, 1, 1))
    return F.layer_norm(h, (h.size(-1),))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", default="/dev/video0")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--goal", default=None, help="goal image path (default: the first camera frame)")
    ap.add_argument("--save-goal", default=None, help="write the goal frame used to this jpg")
    ap.add_argument("--step", type=float, default=0.05, help="candidate action size in metres")
    ap.add_argument("--no-actions", action="store_true", help="skip the predictor, report E_goal only")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--exposure", type=int, default=None, help="lock manual exposure (C270 units of 100 us)")
    ap.add_argument("--mains-hz", type=int, default=50, choices=[50, 60])
    ap.add_argument("--random-init", action="store_true", help="skip weights (timing only)")
    args = ap.parse_args()

    device = pick_device(args.device)
    dtype = getattr(torch, args.dtype)
    size = 256  # 2-AC resolution: 16x16 = 256 tokens per frame
    print(f"device={device} dtype={dtype}")

    controls = {"exposure_dynamic_framerate": 0, "power_line_frequency": 1 if args.mains_hz == 50 else 2}
    controls.update(
        {"auto_exposure": 1, "exposure_time_absolute": args.exposure} if args.exposure else {"auto_exposure": 3}
    )
    set_v4l2_controls(str(args.camera), controls)

    if args.random_init:
        import src.hub.backbones as hub

        encoder, predictor = hub.vjepa2_ac_vit_giant(pretrained=False)
        encoder, predictor = encoder.to(device).eval(), predictor.to(device).eval()
    else:
        encoder, predictor = load_model("vjepa2_ac_vit_giant", device=device)
    print(f"encoder {n_params(encoder):.0f}M | predictor {n_params(predictor):.0f}M")
    tokens_per_frame = (size // encoder.patch_size) ** 2

    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    labels, actions = candidate_actions(args.step, device, torch.float32)
    S = actions.shape[0]
    # Fixed arm pose from the Franka example; see the caveat in the module docstring.
    state = torch.tensor(np.load(TRAJ)["states"][:, :1], dtype=torch.float32, device=device).repeat(S, 1, 1)

    cam = Camera(args.camera, args.width, args.height, args.fps, maxlen=8)
    t_wait = time.perf_counter()
    while cam.latest(1, 1) is None:
        if cam.error or time.perf_counter() - t_wait > 10:
            cam.close()
            sys.exit(f"camera delivered no frames: {cam.error or 'timeout'}")
        time.sleep(0.01)
    # The C270 returns black frames for ~1 s after opening, then auto-exposure ramps. Wait until
    # brightness is non-trivial and stable so the goal frame (and first decisions) are real.
    t_wait, last_mean = time.perf_counter(), -1.0
    while time.perf_counter() - t_wait < 5:
        time.sleep(0.25)
        m = float(cam.latest(1, 1)[1][0].mean())
        if m > 8 and abs(m - last_mean) < 0.02 * max(m, 1.0):
            break
        last_mean = m
    print(f"camera settled after {time.perf_counter() - t_wait:.2f} s, mean brightness {m:.0f}/255")

    if args.goal:
        goal_bgr = cv2.imread(args.goal)
        if goal_bgr is None:
            cam.close()
            sys.exit(f"cannot read goal image {args.goal}")
        print(f"goal: {args.goal} {goal_bgr.shape[1]}x{goal_bgr.shape[0]}")
    else:
        goal_bgr = cam.latest(1, 1)[1][0]
        print("goal: first camera frame")
    if args.save_goal:
        cv2.imwrite(args.save_goal, goal_bgr)
        print(f"saved goal frame to {args.save_goal}")

    autocast = torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype != torch.float32)
    stats = collections.defaultdict(list)
    best_counts = collections.Counter()
    last_t = None
    n_dup = 0
    cols = ["t", "Hz", "enc ms", "pred ms", "age ms", "E_goal", "d_pool", "best", "E_best", "spread"]
    print(" ".join(f"{c:>7}" for c in cols))

    try:
        with torch.inference_mode(), autocast:
            z_goal = encode_frame(encoder, preprocess(goal_bgr[None], size, device, mean, std)).float()
            g_pool = F.normalize(z_goal.mean(dim=1), dim=-1)[0]
            # warm-up, not counted
            z = encode_frame(encoder, preprocess(goal_bgr[None], size, device, mean, std))
            if not args.no_actions:
                predictor(z.float().repeat(S, 1, 1), actions, state)
            sync(device)

            t_start = t_report = time.perf_counter()
            frames_at_start = cam.count
            win = collections.defaultdict(list)
            win_best = collections.Counter()
            while time.perf_counter() - t_start < args.seconds:
                if cam.error:
                    raise RuntimeError(cam.error)
                ts, frames = cam.latest(1, 1)
                if ts[-1] == last_t:
                    n_dup += 1
                    time.sleep(0.001)
                    continue
                last_t = ts[-1]

                t0 = time.perf_counter()
                z = encode_frame(encoder, preprocess(frames, size, device, mean, std)).float()
                e_goal = float(torch.abs(z - z_goal).mean())
                # elementwise product, not `@`: autocast would run a matmul in bf16 and quantize this to ~0.004
                d_pool = 1 - float((F.normalize(z.mean(dim=1), dim=-1)[0] * g_pool).sum())
                sync(device)
                t1 = time.perf_counter()

                row = {"enc": (t1 - t0) * 1000, "e_goal": e_goal, "d_pool": d_pool}
                if not args.no_actions:
                    z1 = predictor(z.repeat(S, 1, 1), actions, state)[:, -tokens_per_frame:]
                    z1 = F.layer_norm(z1, (z1.size(-1),)).float()
                    energy = torch.abs(z1 - z_goal).mean(dim=[1, 2])
                    sync(device)
                    t2 = time.perf_counter()
                    best = int(energy.argmin())
                    row.update(
                        {
                            "pred": (t2 - t1) * 1000,
                            "e_best": float(energy[best]),
                            "spread": float(energy.max() - energy.min()),
                        }
                    )
                    best_counts[labels[best]] += 1
                    win_best[labels[best]] += 1
                row["age"] = (time.perf_counter() - ts[-1]) * 1000
                for k, v in row.items():
                    stats[k].append(v)
                    win[k].append(v)

                now = time.perf_counter()
                if now - t_report >= 1.0:
                    top = win_best.most_common(1)[0][0] if win_best else "-"
                    vals = [now - t_start, len(win["enc"]) / (now - t_report), np.mean(win["enc"])]
                    vals += [np.mean(win[k]) if win[k] else float("nan") for k in ("pred", "age")]
                    out = " ".join(f"{v:7.1f}" for v in vals)
                    out += f" {np.mean(win['e_goal']):7.4f} {np.mean(win['d_pool']):7.4f} {top:>7}"
                    tail = [np.mean(win[k]) if win[k] else float("nan") for k in ("e_best", "spread")]
                    out += "".join(f" {v:7.4f}" for v in tail)
                    print(out, flush=True)
                    win = collections.defaultdict(list)
                    win_best = collections.Counter()
                    t_report = now
    finally:
        elapsed = time.perf_counter() - t_start
        cam_fps = (cam.count - frames_at_start) / elapsed
        cam.close()

    n = len(stats["enc"])
    print("\n== summary")
    print(
        f"decisions            {n} in {elapsed:.1f} s = {n / elapsed:.1f} Hz "
        f"(camera {cam_fps:.1f} fps, idle waits {n_dup})"
    )
    rows = [("enc", "encode frame"), ("pred", f"predictor x{S}"), ("age", "frame age at output")]
    for k, label in rows:
        if stats[k]:
            print(f"{label:<20} median {pct(stats[k], 50):6.1f} ms   p95 {pct(stats[k], 95):6.1f} ms")
    e = stats["e_goal"]
    print(f"E_goal               min {np.min(e):.4f}  median {pct(e, 50):.4f}  max {np.max(e):.4f}")
    d = stats["d_pool"]
    print(f"d_pool               min {np.min(d):.4f}  median {pct(d, 50):.4f}  max {np.max(d):.4f}")
    if best_counts:
        print(
            f"candidate spread     median {pct(stats['spread'], 50):.4f}  "
            f"(E_best median {pct(stats['e_best'], 50):.4f})"
        )
        print("best action counts   " + "  ".join(f"{k}:{v}" for k, v in best_counts.most_common()))
    print(f"peak device mem      {peak_mem_gb(device):.2f} GB")


if __name__ == "__main__":
    main()
