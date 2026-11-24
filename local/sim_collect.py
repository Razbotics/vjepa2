"""Collect arm data in simulation, in the same format as arm_collect.py writes for the real arm.

The simulated SO-101 (local/sim_arm.py) does the same random walk: targets inside a box around the ready
pose, each at most --step-deg from the last, goals ramped at a speed limit, a dwell on arrival. Every
camera frame is saved with the measured joint angles. arm_encode.py, arm_predictor.py and the planner's
loaders read the result unchanged.

    cd ~/projects/vjepa2
    ~/projects/lerobot/.venv/bin/python -m local.sim_collect --seconds 120
    ~/projects/lerobot/.venv/bin/python -m local.sim_collect --joints elbow_flex wrist_flex wrist_roll gripper \
        --max-deg 10 --step-deg 6            # the real run's settings
    ~/projects/lerobot/.venv/bin/python -m local.sim_collect --seconds 3600 --seed 1

Differences from the real collector: all six joints move by default and the box is wider (no hardware at
risk), time is simulated (an hour of data takes a few minutes), and there are no load or temperature
stops. A target is rejected if it would put the gripper tip or a link origin within --min-height of the
table.

Output (default local/data/sim_arm/<timestamp>):
  frames/000000.jpg ...   one per sample
  log.npz                 t_frame, t_joint, pos_deg, goal_deg, torque, tip, target_id
  meta.json               joints, moving joints, ready pose, camera and motion settings, "sim": true
"""

import argparse
import json
import os
import time

import cv2
import numpy as np

from local.sim_arm import ASSETS, CAMERA, JOINTS, SimArm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--seconds", type=float, default=120.0, help="simulated seconds to record")
    ap.add_argument("--joints", nargs="+", default=JOINTS, choices=JOINTS, help="joints that move")
    ap.add_argument("--max-deg", type=float, default=20.0, help="half-width of the box around the ready pose")
    ap.add_argument("--step-deg", type=float, default=8.0, help="largest change of a target per joint")
    ap.add_argument("--margin-deg", type=float, default=3.0, help="stay this far inside the joint limits")
    ap.add_argument("--speed", type=float, default=17.6, help="goal ramp, deg/s (the real run's 200 ticks/s)")
    ap.add_argument("--dwell", type=float, default=0.3)
    ap.add_argument("--min-height", type=float, default=0.03, help="metres above the table")
    ap.add_argument("--kp", type=float, default=17.8, help="servo stiffness, N m/rad")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--jpeg-quality", type=int, default=92)
    args = ap.parse_args()

    out = args.out or os.path.join("local", "data", "sim_arm", time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(os.path.join(out, "frames"))
    rng = np.random.default_rng(args.seed)
    arm = SimArm(width=args.width, height=args.height, fps=args.fps, kp=args.kp)
    arm.reset(arm.ready)
    moving = np.array([j in args.joints for j in JOINTS])
    lo = np.maximum(arm.ready - args.max_deg, arm.range_deg[:, 0] + args.margin_deg)
    hi = np.minimum(arm.ready + args.max_deg, arm.range_deg[:, 1] - args.margin_deg)
    lo, hi = np.where(moving, lo, arm.ready), np.where(moving, hi, arm.ready)
    assert (lo <= hi).all(), "the ready pose is outside the joint limits"

    def new_target(prev):
        for _ in range(50):
            t = np.clip(prev + rng.uniform(-args.step_deg, args.step_deg, len(JOINTS)), lo, hi)
            t = np.where(moving, t, arm.ready)
            if arm.lowest(t) >= args.min_height:
                return t, True
        return arm.ready.copy(), False

    n = int(round(args.seconds * args.fps))
    step = args.speed / args.fps
    log = {k: [] for k in ("t_frame", "pos_deg", "goal_deg", "torque", "tip", "target_id")}
    goal, target = arm.ready.copy(), arm.ready.copy()
    target_id, dwell_left, rejected, started = 0, 0.0, 0, time.perf_counter()
    for i in range(n):
        if np.abs(goal - target).max() < 1e-6 and np.abs(arm.qvel_deg).max() < 2.0:
            dwell_left -= 1.0 / args.fps
            if dwell_left <= 0:
                target, ok = new_target(target)
                rejected += not ok
                target_id, dwell_left = target_id + 1, args.dwell
        goal = goal + np.clip(target - goal, -step, step)
        arm.step(goal)
        path = os.path.join(out, "frames", f"{i:06d}.jpg")
        cv2.imwrite(path, arm.render(), [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
        log["t_frame"].append(arm.t)
        log["pos_deg"].append(arm.q_deg)
        log["goal_deg"].append(goal.copy())
        log["torque"].append(arm.torque)
        log["tip"].append(arm.tip())
        log["target_id"].append(target_id)
    wall = time.perf_counter() - started

    arrays = {k: np.asarray(v) for k, v in log.items()}
    arrays["t_joint"] = arrays["t_frame"]
    np.savez(os.path.join(out, "log.npz"), **arrays)
    meta = {
        "sim": True,
        "joints": JOINTS,
        "moving": [j for j in JOINTS if j in args.joints],
        "ready_deg": arm.ready.tolist(),
        "box_deg": [lo.tolist(), hi.tolist()],
        "camera": CAMERA,
        "assets": ASSETS,
        "samples": n,
        "targets": target_id,
        "targets_rejected": rejected,
        "args": vars(args),
    }
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    q = arrays["pos_deg"]
    err = np.abs(q - arrays["goal_deg"])
    print(f"{n} frames = {args.seconds:.0f} s simulated in {wall:.1f} s ({n / wall:.0f} fps), {target_id} targets")
    print(f"{'joint':<14}{'min':>8}{'max':>8}{'goal err mean':>15}{'max':>7}   (deg)")
    for k, j in enumerate(JOINTS):
        print(f"{j:<14}{q[:, k].min():>8.1f}{q[:, k].max():>8.1f}{err[:, k].mean():>15.2f}{err[:, k].max():>7.2f}")
    peak, low = np.abs(arrays["torque"]).max(), arrays["tip"][:, 2].min()
    print(f"peak torque {peak:.2f} N m (limit 3.35), lowest tip {low:.3f} m")
    if rejected:
        print(f"{rejected} targets fell back to the ready pose (no safe target found)")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
