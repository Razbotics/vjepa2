"""Collect onboard-camera flight data from the simulated Crazyflie (local/sim_drone.py).

    cd ~/projects/vjepa2
    ~/projects/lerobot/.venv/bin/python -m local.sim_drone_collect --seconds 60
    ~/projects/lerobot/.venv/bin/python -m local.sim_drone_collect --seconds 3600 --seed 1

The drone flies a random walk of waypoints, the drone's equivalent of sim_collect.py's arm random walk. Each
waypoint is a position and a heading, at most --step-m away (per axis, --step-z vertically) and
--step-yaw deg of turn from the last. It stays inside the room (--wall-margin from the walls, between
--z-min and --z-max), and the straight path to it keeps --clearance from every pillar and box; otherwise it
is drawn again. The controller's target slides to each waypoint along a minimum-jerk profile (smooth
speed-up and slow-down) at an average speed drawn from --speed-min..--speed-max, turning at up to
--yaw-rate, then holds for a random 0..--dwell s.

What the world model will see as an action is the controller's target, not the rotor thrusts: ref_pos,
ref_vel and ref_yaw are logged every frame, next to the measured state. If the drone ever touches
something or tilts past 60 deg, it is put back in the air at a safe hover and `segment` increments, so
training pairs can be kept from straddling the reset.

Output (default local/data/sim_drone/<timestamp>):
  frames/000000.jpg ...   onboard camera, one per sample
  log.npz                 t_frame, pos, quat (w x y z), rpy_deg, vel (world), rate (body), ref_pos, ref_vel,
                          ref_acc, ref_yaw, target (x y z yaw), motors, target_id, segment
  meta.json               room, camera, controller and motion settings, "sim": true
"""

import argparse
import json
import os
import time

import cv2
import numpy as np

from local.sim_drone import BOXES, CF2_DIR, FPV_TILT, PILLAR_R, PILLARS, ROOM, SimDrone


def clear_path(a, b, clearance):
    """True if the segment a->b (x, y, z) keeps `clearance` from every pillar and box."""
    pts = a + np.linspace(0.0, 1.0, max(2, int(np.linalg.norm(b - a) / 0.05) + 2))[:, None] * (b - a)
    for x, y, _ in PILLARS:  # 2 m tall, taller than any target
        if (np.hypot(pts[:, 0] - x, pts[:, 1] - y) < PILLAR_R + clearance).any():
            return False
    for x, y, s, _ in BOXES:
        near = (np.abs(pts[:, 0] - x) < s + clearance) & (np.abs(pts[:, 1] - y) < s + clearance)
        if (near & (pts[:, 2] < 2 * s + clearance)).any():
            return False
    return True


def min_jerk(tau):
    """Fraction of a rest-to-rest minimum-jerk move done at tau in [0, 1], and its 1st and 2nd derivatives."""
    tau = np.clip(tau, 0.0, 1.0)
    return (10 * tau**3 - 15 * tau**4 + 6 * tau**5, 30 * tau**2 - 60 * tau**3 + 30 * tau**4,
            60 * tau - 180 * tau**2 + 120 * tau**3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--seconds", type=float, default=60.0, help="simulated seconds to record")
    ap.add_argument("--step-m", type=float, default=0.8, help="largest horizontal change per axis, m")
    ap.add_argument("--step-z", type=float, default=0.3, help="largest vertical change, m")
    ap.add_argument("--step-yaw", type=float, default=45.0, help="largest heading change, deg")
    ap.add_argument("--speed-min", type=float, default=0.2, help="m/s, average over a leg")
    ap.add_argument("--speed-max", type=float, default=0.8)
    ap.add_argument("--yaw-rate", type=float, default=60.0, help="deg/s, average over a turn")
    ap.add_argument("--dwell", type=float, default=0.5, help="longest hold at a waypoint, s")
    ap.add_argument("--z-min", type=float, default=0.3)
    ap.add_argument("--z-max", type=float, default=2.0)
    ap.add_argument("--wall-margin", type=float, default=0.5)
    ap.add_argument("--clearance", type=float, default=0.3, help="metres from pillars and boxes")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--jpeg-quality", type=int, default=92)
    args = ap.parse_args()

    out = args.out or os.path.join("local", "data", "sim_drone", time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(os.path.join(out, "frames"))
    rng = np.random.default_rng(args.seed)
    drone = SimDrone(width=args.width, height=args.height, fps=args.fps)
    dt = drone.model.opt.timestep
    lim = ROOM - args.wall_margin
    home = np.array([0.0, 0.0, 1.0, 0.0])  # x, y, z, yaw (rad)
    assert clear_path(home[:3], home[:3], args.clearance), "home must be clear of obstacles"
    drone.hover_at(home[:3], home[3])

    def new_target(prev):
        for _ in range(200):
            d = np.concatenate([rng.uniform(-args.step_m, args.step_m, 2), rng.uniform(-args.step_z, args.step_z, 1),
                                np.radians(rng.uniform(-args.step_yaw, args.step_yaw, 1))])
            t = prev + d
            if (np.abs(t[:2]) <= lim).all() and args.z_min <= t[2] <= args.z_max and \
                    clear_path(prev[:3], t[:3], args.clearance):
                return t, True
        return prev.copy(), False  # boxed in: hold, and try again after the dwell

    def new_leg(start):
        target, ok = new_target(start)
        speed = rng.uniform(args.speed_min, args.speed_max)
        dur = max(np.linalg.norm(target[:3] - start[:3]) / speed,
                  abs(target[3] - start[3]) / np.radians(args.yaw_rate), 0.2)
        return target, dur, rng.uniform(0.0, args.dwell), ok

    n = int(round(args.seconds * args.fps))
    keys = ("t_frame", "pos", "quat", "rpy_deg", "vel", "rate", "ref_pos", "ref_vel", "ref_acc", "ref_yaw", "target",
            "motors", "target_id", "segment")
    log = {k: [] for k in keys}
    start = home.copy()
    target, dur, dwell, _ = new_leg(start)
    t_leg, target_id, segment, rejected, crashes = 0.0, 1, 0, 0, 0
    ref, v_ref, a_ref, yaw_ref, f = start[:3].copy(), np.zeros(3), np.zeros(3), start[3], np.zeros(4)
    started = time.perf_counter()
    for i in range(n):
        for _ in range(drone.substeps):
            s, ds, dds = min_jerk(t_leg / dur)
            move = target[:3] - start[:3]
            ref = start[:3] + s * move
            v_ref, a_ref = (ds / dur * move, dds / dur**2 * move) if t_leg < dur else (np.zeros(3), np.zeros(3))
            yaw_ref = start[3] + s * (target[3] - start[3])
            f = drone.tick(ref, v_ref, yaw_ref, a_ref=a_ref)
            t_leg += dt
            if t_leg >= dur + dwell:
                start = target
                target, dur, dwell, ok = new_leg(start)
                t_leg, target_id, rejected = 0.0, target_id + 1, rejected + (not ok)
        r = drone.rpy_deg
        if drone.touching() or abs(r[0]) > 60 or abs(r[1]) > 60:  # never expected; recover rather than stop
            crashes, segment = crashes + 1, segment + 1
            p = np.clip(drone.pos, [-lim, -lim, args.z_min], [lim, lim, args.z_max])
            start = np.append(p if clear_path(p, p, args.clearance) else home[:3], np.radians(r[2]))
            drone.hover_at(start[:3], start[3])
            target, dur, dwell, _ = new_leg(start)
            t_leg, target_id = 0.0, target_id + 1
        cv2.imwrite(os.path.join(out, "frames", f"{i:06d}.jpg"), drone.render("fpv"),
                    [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
        for k, v in (("t_frame", (i + 1) / args.fps), ("pos", drone.pos), ("quat", drone.data.qpos[3:7].copy()),
                     ("rpy_deg", drone.rpy_deg), ("vel", drone.vel), ("rate", drone.rate), ("ref_pos", ref.copy()),
                     ("ref_vel", v_ref.copy()), ("ref_acc", a_ref.copy()),
                     ("ref_yaw", yaw_ref), ("target", target.copy()),
                     ("motors", f.copy()), ("target_id", target_id), ("segment", segment)):
            log[k].append(v)
        if (i + 1) % (60 * int(args.fps)) == 0:
            el = time.perf_counter() - started
            print(f"  {(i + 1) / args.fps / 60:.0f} min simulated in {el:.0f} s, {target_id} waypoints, "
                  f"{crashes} resets", flush=True)
    wall = time.perf_counter() - started

    arrays = {k: np.asarray(v) for k, v in log.items()}
    np.savez(os.path.join(out, "log.npz"), **arrays)
    meta = {
        "sim": True,
        "robot": "crazyflie_2",
        "model": CF2_DIR,
        "camera": {"name": "fpv", "tilt_down_deg": FPV_TILT, "fovy_deg": 70, "width": args.width,
                   "height": args.height},
        "room_half_width_m": ROOM,
        "pillars": PILLARS, "boxes": BOXES,
        "timestep_s": dt, "substeps_per_frame": drone.substeps,
        "samples": n, "waypoints": target_id, "waypoints_rejected": rejected, "resets": crashes,
        "args": vars(args),
    }
    with open(os.path.join(out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    p, err = arrays["pos"], np.linalg.norm(arrays["pos"] - arrays["ref_pos"], axis=1)
    spd = np.linalg.norm(arrays["vel"], axis=1)
    tilt = np.abs(arrays["rpy_deg"][:, :2]).max(1)
    step = int(round(0.5 * args.fps))
    dyaw = np.abs((np.diff(arrays["rpy_deg"][::step, 2]) + 180) % 360 - 180)
    dpos = np.linalg.norm(np.diff(p[::step], axis=0), axis=1)
    print(f"{n} frames = {args.seconds:.0f} s simulated in {wall:.1f} s ({n / wall:.0f} fps), {target_id} waypoints, "
          f"{crashes} resets, {rejected} waypoints redrawn out")
    print(f"position covered  x {p[:, 0].min():+.2f}..{p[:, 0].max():+.2f}"
          f"  y {p[:, 1].min():+.2f}..{p[:, 1].max():+.2f}  z {p[:, 2].min():.2f}..{p[:, 2].max():.2f} m")
    print(f"speed mean {spd.mean():.2f} max {spd.max():.2f} m/s | tilt mean {tilt.mean():.1f} max {tilt.max():.1f} deg"
          f" | tracking error mean {100 * err.mean():.1f} max {100 * err.max():.1f} cm")
    print(f"per 0.5 s: moves {100 * dpos.mean():.1f} cm and turns {dyaw.mean():.1f} deg on average "
          f"(still {100 * (dpos < 0.01).mean():.0f}% of the time)")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
