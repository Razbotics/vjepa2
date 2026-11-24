"""Closed-loop image-goal reaching on the SO-101 with the predictor from arm_predictor.py.

Runs on the desktop in the LeRobot environment (it also needs `timm` there for the V-JEPA encoder):

    cd ~/projects/vjepa2
    ~/projects/lerobot/.venv/bin/python -m local.arm_plan local/data/arm/20261124_011609
    ~/projects/lerobot/.venv/bin/python -m local.arm_plan local/data/arm/<run> --policy random

Each episode:
  1. The arm moves to a random pose in the collection box and the camera records the goal image there.
  2. It moves to a different random pose, the start.
  3. For --steps control steps of 0.5 s: encode the current frame, sample candidate joint-delta action
     sequences, roll each through the predictor, score by L1 distance of the predicted features to the
     goal image's features, refine with CEM, execute the first action.
The planner sees only images and its own joint angles. The goal's joint angles are recorded for scoring
and never shown to it. `--policy random` executes random actions of the same size as a baseline.

Safety: every command is clipped to the +/- --max-deg box around the ready pose that the collection runs
already explored, steps are limited to --max-step degrees per joint, and the arm is parked and released on
any exit. Before the first episode it checks that the live view matches the training run (camera not
moved); if not it parks without planning.
"""

import argparse
import glob
import json
import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from local.arm_collect import DEFAULT_PORT, READY_POSES, TICKS_PER_DEG
from local.arm_predictor import Predictor
from local.camera import Camera, set_v4l2_controls
from local.vjepa_local import IMAGENET_MEAN, IMAGENET_STD, load_model, pick_device


class Arm:
    """Minimal SO-101 follower session: staged move to a ready pose, raw goals, staged park."""

    def __init__(self, port, robot_id, speed=200, accel=10, max_load=700):
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

        robot = SO101Follower(SO101FollowerConfig(port=port, id=robot_id))
        self.bus, self.names = robot.bus, list(robot.bus.motors)
        cal = [robot.calibration[n] for n in self.names]
        self.rmin = np.array([c.range_min for c in cal], dtype=float)
        self.rmax = np.array([c.range_max for c in cal], dtype=float)
        self.mid = (self.rmin + self.rmax) / 2
        self.speed, self.accel, self.max_load = speed, accel, max_load
        self.torque_on, self.path, self.ready = False, [], None

    def read(self, reg="Present_Position"):
        d = self.bus.sync_read(reg, normalize=False)
        return np.array([d[n] for n in self.names], dtype=float)

    def deg(self, raw):
        return (np.asarray(raw) - self.mid) / TICKS_PER_DEG

    def raw(self, deg):
        return np.asarray(deg) * TICKS_PER_DEG + self.mid

    def write_goal(self, goal):
        self.bus.sync_write("Goal_Position", {n: int(round(g)) for n, g in zip(self.names, goal)}, normalize=False)

    def move_and_wait(self, goal, label=None, timeout=8.0, sag_tol=45):
        self.write_goal(goal)
        t0, last, still = time.perf_counter(), self.read(), 0
        while time.perf_counter() - t0 < timeout and still < 3:
            time.sleep(0.1)
            pos, load = self.read(), self.read("Present_Load")
            if np.abs(load).max() > self.max_load:
                raise RuntimeError(f"load {load.tolist()} above {self.max_load}")
            still = still + 1 if np.abs(pos - last).max() <= 1 else 0
            last = pos
        err = np.abs(last - goal).max()
        if err > sag_tol:
            raise RuntimeError(f"stopped {err:.0f} ticks from goal (blocked?)")
        if label:
            print(f"{label:<22} pos={last.astype(int).tolist()}")
        return last

    def start(self, ready_name, rest_tol=60):
        self.bus.connect()
        if self.read("Operating_Mode").any() or self.read("Torque_Enable").any():
            raise RuntimeError("servos must be in position mode with torque off at the start")
        start = self.read()
        pose = READY_POSES[ready_name]
        off = {j: int(start[self.names.index(j)] - v) for j, v in pose["rest"].items()}
        if max(abs(v) for v in off.values()) > rest_tol:
            raise RuntimeError(f"arm is not in the '{ready_name}' rest pose, offsets {off}")
        for n in self.names:
            self.bus.write("Acceleration", n, self.accel, normalize=False)
            self.bus.write("Goal_Velocity", n, self.speed, normalize=False)
        goal = np.clip(start, self.rmin, self.rmax)
        self.write_goal(goal)
        for n in self.names:
            self.bus.enable_torque(n)
            time.sleep(0.1)
        self.torque_on = True
        time.sleep(0.5)
        for j, v in pose["stages"]:
            self.path.append(goal.copy())
            goal[self.names.index(j)] = v
            self.move_and_wait(goal, f"to ready: {j}")
        self.ready = goal.copy()

    def park(self):
        if not self.bus.is_connected:
            return
        try:
            if self.torque_on:
                for g in ([self.ready] if self.ready is not None else []) + self.path[::-1]:
                    try:
                        self.move_and_wait(g, sag_tol=10**6)
                    except RuntimeError as e:
                        print("while parking:", e)
        finally:
            self.bus.disable_torque()
            print("torque enable", self.read("Torque_Enable").astype(int).tolist())
            self.bus.disconnect(disable_torque=False)


def encode_frames(frames, encoder, mean, std, zmu, zsd, device, jpeg_quality=92):
    """BGR frames -> standardized 4x4 feature grids [n, 16, D], matching arm_encode.py's preprocessing.

    The training frames went through a jpg on disk; round-trip the live ones the same way so inputs match.
    """
    frames = [cv2.imdecode(cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])[1], 1) for f in frames]
    x = torch.from_numpy(np.stack(frames)).to(device).flip(-1).permute(0, 3, 1, 2).float().div_(255.0)
    h, w = x.shape[-2:]
    s = min(h, w)
    x = x[..., (h - s) // 2 : (h - s) // 2 + s, (w - s) // 2 : (w - s) // 2 + s]
    x = F.interpolate(x, size=(256, 256), mode="bilinear", antialias=True, align_corners=False)
    x = ((x - mean) / std).unsqueeze(2).repeat(1, 1, 2, 1, 1)
    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16):
        z = encoder(x).float()
    b, t, d = z.shape
    side = int(round(t**0.5))
    g = F.adaptive_avg_pool2d(z.transpose(1, 2).reshape(b, d, side, side), 4).flatten(2).transpose(1, 2)
    return (g - zmu) / zsd


def cem_plan(model, z, q, z_goal, lo, hi, qmu, qsd, gen, horizon=2, samples=256, iters=3, elites=24, sigma=3.0,
             max_step=5.0):
    """CEM over joint-delta sequences [H, A] in degrees, scored by L1 of the predicted features to the goal.

    z, z_goal [1, 16, D] standardized; q current joint angles (deg); lo, hi box tensors (deg).
    Returns (first action in deg as numpy, lowest energy).
    """
    device = z.device
    q_t = torch.tensor(q, device=device, dtype=torch.float32)
    mean = torch.zeros(horizon, len(q), device=device)
    std = torch.full_like(mean, sigma)
    with torch.inference_mode():
        for _ in range(iters):
            a = mean + std * torch.randn((samples,) + mean.shape, device=device, generator=gen)
            a = a.clamp(-max_step, max_step)
            zc, qc = z.expand(samples, -1, -1), q_t.expand(samples, -1)
            for h in range(horizon):
                a[:, h] = torch.minimum(torch.maximum(qc + a[:, h], lo), hi) - qc  # stay in the box
                zc = model(zc, (qc - qmu) / qsd, a[:, h] / qsd)
                qc = qc + a[:, h]
            energy = (zc - z_goal).abs().mean(dim=(1, 2))
            best = a[energy.topk(elites, largest=False).indices]
            mean, std = best.mean(0), best.std(0).clamp_min(0.2)
    return mean[0].cpu().numpy(), float(energy.min())


class Eyes:
    """Camera -> standardized 4x4 feature grid, matching arm_encode.py's preprocessing."""

    def __init__(self, model, ckpt, camera, device, mains_hz=50):
        self.device = device
        self.encoder = load_model(model, device=device)[0]
        self.zmu, self.zsd = ckpt["zmu"].to(device), ckpt["zsd"].to(device)
        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
        set_v4l2_controls(
            camera,
            {"exposure_dynamic_framerate": 0, "power_line_frequency": 1 if mains_hz == 50 else 2, "auto_exposure": 3},
        )
        self.cam = Camera(camera, 640, 360, 30, maxlen=16)
        t0 = time.perf_counter()
        while self.cam.latest(1, 1) is None or self.cam.latest(1, 1)[1][0].mean() < 8:
            if self.cam.error or time.perf_counter() - t0 > 10:
                raise RuntimeError(f"camera delivered no usable frames: {self.cam.error or 'timeout'}")
            time.sleep(0.05)
        time.sleep(1.0)  # let auto-exposure settle
        self.last = None

    def see(self, n=3):
        """Mean feature grid [1, 16, D] over n fresh frames. Averaging cuts per-frame sensor noise."""
        start = self.cam.count
        while self.cam.count < start + n:
            time.sleep(0.003)
        frames = self.cam.latest(n, 1)[1]
        self.last = frames[-1]
        g = encode_frames(frames, self.encoder, self.mean, self.std, self.zmu, self.zsd, self.device)
        return g.mean(0, keepdim=True)


def scene_check(run, model, joints, eyes, q_now):
    """Compare the live view with training frames at similar joint angles. Returns (live, reference) L1."""
    log = np.load(os.path.join(run, "log.npz"))
    meta = json.load(open(os.path.join(run, "meta.json")))
    feat = np.load(os.path.join(run, f"feat_{model}.npz"))
    cols = [meta["joints"].index(j) for j in joints]
    q = log["pos_deg"][:, cols]
    n, dim = len(q), feat["grid"].shape[1] // 16
    z = torch.from_numpy(feat["grid"].astype(np.float32)).reshape(n, dim, 16).transpose(1, 2).to(eyes.device)
    z = (z - eyes.zmu) / eyes.zsd
    near = np.argsort(np.abs(q - q_now).mean(1))[:5]
    live = float(np.mean([(eyes.see(3) - z[i : i + 1]).abs().mean().item() for i in near]))
    ref = []
    rng = np.random.default_rng(0)
    for i in rng.choice(n, 200, replace=False):
        d = np.abs(q - q[i]).mean(1)
        d[max(0, i - 30) : i + 31] = np.inf
        ref.append((z[i] - z[int(d.argmin())]).abs().mean().item())
    return live, float(np.median(ref)), float(np.abs(q[near] - q_now).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="run folder holding pred_<model>.pt and feat_<model>.npz")
    ap.add_argument("--model", default="vjepa2_1_vit_large_384")
    ap.add_argument("--policy", default="plan", choices=["plan", "random"])
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--steps", type=int, default=14, help="control steps per episode")
    ap.add_argument("--horizon", type=int, default=2, help="actions per candidate sequence")
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--elites", type=int, default=24)
    ap.add_argument("--sigma", type=float, default=3.0, help="initial action std, degrees per joint")
    ap.add_argument("--max-step", type=float, default=5.0, help="largest joint change per step, degrees")
    ap.add_argument("--max-deg", type=float, default=10.0, help="box half-width around the ready pose")
    ap.add_argument("--min-start", type=float, default=4.0, help="start at least this far from the goal (deg)")
    ap.add_argument("--ready", default="table", choices=sorted(READY_POSES))
    ap.add_argument("--port", default=DEFAULT_PORT)
    ap.add_argument("--robot-id", default="my_follower_arm")
    ap.add_argument("--camera", default="/dev/video0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="plan even if the scene check fails")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = pick_device(None)
    rng = np.random.default_rng(args.seed)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    ckpt = torch.load(os.path.join(args.run, f"pred_{args.model}.pt"), map_location=device)
    joints, cfg = ckpt["joints"], ckpt["args"]
    qmu, qsd = ckpt["qmu"].to(device), ckpt["qsd"].to(device)
    dt = cfg["k"] / 30.0
    model = Predictor(ckpt["zmu"].shape[-1], len(joints), cfg["width"], cfg["depth"]).to(device).eval()
    model.load_state_dict(ckpt["state_dict"])
    out = args.out or os.path.join("local", "runs", f"plan_{args.policy}_" + time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out, exist_ok=True)

    eyes = Eyes(args.model, ckpt, args.camera, device)
    arm = Arm(args.port, args.robot_id)
    mv = [arm.names.index(j) for j in joints]
    results, reason = [], "completed"
    try:
        arm.start(args.ready)
        margin, span = 3 * TICKS_PER_DEG, args.max_deg * TICKS_PER_DEG
        centre = np.clip(arm.ready, arm.rmin + margin, arm.rmax - margin)
        lo = np.maximum(arm.rmin + margin, centre - span)[mv]
        hi = np.minimum(arm.rmax - margin, centre + span)[mv]
        lo_deg = (lo - arm.mid[mv]) / TICKS_PER_DEG
        hi_deg = (hi - arm.mid[mv]) / TICKS_PER_DEG

        def q_now():
            return arm.deg(arm.read())[mv]

        def go(q_deg, wait=True):
            goal = arm.ready.copy()
            goal[mv] = np.clip(np.asarray(q_deg) * TICKS_PER_DEG + arm.mid[mv], lo, hi)
            if wait:
                arm.move_and_wait(goal)
            else:
                arm.write_goal(goal)

        live, ref, _ = scene_check(args.run, args.model, joints, eyes, q_now())
        print(f"scene check: live vs training frames at the same pose L1 {live:.3f}, training pairs {ref:.3f}")
        if live > 1.5 * ref and not args.force:
            raise RuntimeError("the live view does not match the training run (camera or lighting changed)")

        lo_t = torch.tensor(lo_deg, device=device, dtype=torch.float32)
        hi_t = torch.tensor(hi_deg, device=device, dtype=torch.float32)

        def plan(z, q, z_goal):
            return cem_plan(model, z, q, z_goal, lo_t, hi_t, qmu, qsd, gen, args.horizon, args.samples, args.iters,
                            args.elites, args.sigma, args.max_step)

        print(f"\npolicy={args.policy}  joints={joints}  step={dt:.2f} s  horizon={args.horizon}")
        for ep in range(args.episodes):
            q_goal_cmd = rng.uniform(lo_deg, hi_deg)
            go(q_goal_cmd)
            time.sleep(0.3)
            z_goal, q_goal = eyes.see(8), q_now()
            cv2.imwrite(os.path.join(out, f"ep{ep}_goal.jpg"), eyes.last)
            noise = (eyes.see(3) - z_goal).abs().mean().item()
            for _ in range(200):
                q_start_cmd = rng.uniform(lo_deg, hi_deg)
                if np.abs(q_start_cmd - q_goal).mean() >= args.min_start:
                    break
            go(q_start_cmd)
            time.sleep(0.3)
            eyes.see(1)
            cv2.imwrite(os.path.join(out, f"ep{ep}_start.jpg"), eyes.last)

            trace = []
            for step in range(args.steps + 1):
                t0 = time.perf_counter()
                z, q = eyes.see(3), q_now()
                err = np.abs(q - q_goal)
                e_now = (z - z_goal).abs().mean().item()
                trace.append({"energy": e_now, "err": err.tolist()})
                if step == args.steps:
                    break
                if args.policy == "plan":
                    act, _ = plan(z, q, z_goal)
                else:
                    act = np.clip(rng.normal(0, args.sigma, len(joints)), -args.max_step, args.max_step)
                go(q + act, wait=False)
                time.sleep(max(0.0, dt - (time.perf_counter() - t0)))
            cv2.imwrite(os.path.join(out, f"ep{ep}_final.jpg"), eyes.last)
            e0, e1 = np.array(trace[0]["err"]), np.array(trace[-1]["err"])
            best_i = int(np.argmin([np.mean(t["err"]) for t in trace]))
            results.append({"goal_deg": q_goal.tolist(), "noise_floor": noise, "trace": trace})
            print(
                f"ep {ep}: joint error {e0.mean():.2f} -> {e1.mean():.2f} deg "
                f"(best {np.mean(trace[best_i]['err']):.2f} at step {best_i}) | per joint "
                + " ".join(f"{a:.1f}->{b:.1f}" for a, b in zip(e0, e1))
                + f" | energy {trace[0]['energy']:.3f} -> {trace[-1]['energy']:.3f} (noise {noise:.3f})"
            )
    except KeyboardInterrupt:
        reason = "interrupted"
    except Exception as e:  # noqa: BLE001 - park the arm whatever went wrong
        reason = f"stopped: {e}"
    finally:
        arm.park()
        eyes.cam.close()

    print(f"\n== {reason}")
    if results:
        s = np.array([np.mean(r["trace"][0]["err"]) for r in results])
        f = np.array([np.mean(r["trace"][-1]["err"]) for r in results])
        e0 = np.array([r["trace"][0]["energy"] for r in results])
        e1 = np.array([r["trace"][-1]["energy"] for r in results])
        print(f"episodes              {len(results)}  policy {args.policy}")
        closed = 100 * (1 - f.mean() / s.mean())
        print(f"mean joint error      {s.mean():.2f} -> {f.mean():.2f} deg  ({closed:.0f}% closed)")
        print(f"episodes that got closer  {(f < s).sum()} of {len(results)}")
        print(f"feature distance to goal  {e0.mean():.3f} -> {e1.mean():.3f}")
    with open(os.path.join(out, "result.json"), "w") as fh:
        json.dump({"args": vars(args), "joints": joints, "reason": reason, "episodes": results}, fh, indent=2)
    print(f"saved to {out}", len(glob.glob(os.path.join(out, "*.jpg"))), "images")


if __name__ == "__main__":
    main()
