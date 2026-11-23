"""Drive the SO-101 follower through small random moves and log measured joints with camera frames.

Runs in the LeRobot environment on the desktop, not the V-JEPA one (LeRobot pins torch < 2.12):

    cd ~/projects/vjepa2
    ~/projects/lerobot/.venv/bin/python -m local.arm_collect --seconds 30
    ~/projects/lerobot/.venv/bin/python -m local.arm_collect --joints wrist_roll gripper --max-deg 15
    ~/projects/lerobot/.venv/bin/python -m local.arm_collect --no-camera --seconds 10

What it does:
  1. Connects to the motor bus only (no EEPROM writes), sets a speed and acceleration limit,
     sets every goal to the current position and enables torque, so the arm holds its pose.
  2. Repeatedly picks a random target for the joints in --joints, within --max-deg of the start
     pose and the calibrated range, and at most --step-deg from the previous target.
  3. For every camera frame, reads the measured joint positions and loads and logs them with
     the frame. The servos stop a few ticks short of their targets (P-only control), so the
     measured positions are the ground truth, not the goals.
  4. Returns to the start pose and releases torque, also on Ctrl-C or a safety stop.

Output directory (default local/data/arm/<timestamp>):
  frames/000000.jpg ...   one per logged sample
  log.npz                 t_frame, t_joint, pos_raw, goal_raw, load, pos_deg, target_id
  meta.json               joint names, calibration, limits, camera and motion settings

Safety stops: a joint that makes no progress towards its goal for --stall-s seconds, a load
above --max-load, or a servo above --max-temp. All joints other than --joints only hold.
"""

import argparse
import json
import os
import queue
import sys
import threading
import time

import cv2
import numpy as np

from local.camera import Camera, set_v4l2_controls

DEFAULT_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B91028149-if00"
TICKS_PER_DEG = 4096 / 360.0


def ticks_to_deg(raw, calib):
    """Raw ticks -> degrees from the middle of the calibrated range (LeRobot's convention)."""
    mid = np.array([(c.range_min + c.range_max) / 2 for c in calib])
    return (np.asarray(raw) - mid) / TICKS_PER_DEG


class FrameWriter:
    """Writes jpgs on a background thread so disk time never delays the control loop."""

    def __init__(self, folder, quality):
        self.folder, self.quality = folder, quality
        self.q = queue.Queue(maxsize=256)
        self.dropped = 0
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            i, frame = item
            cv2.imwrite(os.path.join(self.folder, f"{i:06d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])

    def put(self, i, frame):
        try:
            self.q.put_nowait((i, frame))
        except queue.Full:
            self.dropped += 1

    def close(self):
        self.q.put(None)
        self.thread.join()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=DEFAULT_PORT)
    ap.add_argument("--robot-id", default="my_follower_arm", help="LeRobot calibration id")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--joints", nargs="+", default=["wrist_roll", "gripper"], help="joints allowed to move")
    ap.add_argument("--max-deg", type=float, default=10.0, help="how far a joint may wander from its start pose")
    ap.add_argument("--step-deg", type=float, default=6.0, help="largest change between consecutive targets")
    ap.add_argument("--margin-deg", type=float, default=3.0, help="keep this far inside the calibrated limits")
    ap.add_argument("--speed", type=int, default=200, help="servo speed limit, ticks/s (200 = 17.6 deg/s)")
    ap.add_argument("--accel", type=int, default=10, help="servo acceleration, units of 100 ticks/s^2")
    ap.add_argument("--dwell", type=float, default=0.3, help="seconds to rest at each target")
    ap.add_argument("--tol", type=int, default=12, help="ticks from goal that count as arrived")
    ap.add_argument("--stall-s", type=float, default=2.0)
    ap.add_argument("--max-load", type=int, default=700, help="abort above this servo load (0-1000)")
    ap.add_argument("--max-temp", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-camera", action="store_true", help="log joints only, at --rate Hz")
    ap.add_argument("--rate", type=float, default=30.0, help="log rate without a camera")
    ap.add_argument("--camera", default="/dev/video0")
    # 16:9 on purpose: the C270's 4:3 modes crop the sides of the sensor, 640x360 keeps the full 720p view.
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--jpeg-quality", type=int, default=92)
    ap.add_argument("--exposure", type=int, default=None, help="lock manual exposure (C270 units of 100 us)")
    ap.add_argument("--mains-hz", type=int, default=50, choices=[50, 60])
    args = ap.parse_args()

    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    robot = SO101Follower(SO101FollowerConfig(port=args.port, id=args.robot_id))
    bus = robot.bus
    names = list(bus.motors)
    unknown = [j for j in args.joints if j not in names]
    if unknown:
        sys.exit(f"unknown joints {unknown}; choose from {names}")
    calib = [robot.calibration[n] for n in names]
    lo = np.array([c.range_min for c in calib]) + args.margin_deg * TICKS_PER_DEG
    hi = np.array([c.range_max for c in calib]) - args.margin_deg * TICKS_PER_DEG
    moving = np.array([n in args.joints for n in names])
    rng = np.random.default_rng(args.seed)

    out = args.out or os.path.join("local", "data", "arm", time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(os.path.join(out, "frames"), exist_ok=True)

    def read(reg):
        d = bus.sync_read(reg, normalize=False)
        return np.array([d[n] for n in names])

    def write_goal(goal):
        bus.sync_write("Goal_Position", {n: int(g) for n, g in zip(names, goal)}, normalize=False)

    cam = writer = None
    if not args.no_camera:
        controls = {"exposure_dynamic_framerate": 0, "power_line_frequency": 1 if args.mains_hz == 50 else 2}
        controls.update(
            {"auto_exposure": 1, "exposure_time_absolute": args.exposure} if args.exposure else {"auto_exposure": 3}
        )
        set_v4l2_controls(str(args.camera), controls)
        cam = Camera(args.camera, args.width, args.height, args.fps, maxlen=4)
        t0 = time.perf_counter()
        # The C270 returns black frames for about a second after opening.
        while cam.latest(1, 1) is None or cam.latest(1, 1)[1][0].mean() < 8:
            if cam.error or time.perf_counter() - t0 > 10:
                cam.close()
                sys.exit(f"camera delivered no usable frames: {cam.error or 'timeout or black image'}")
            time.sleep(0.05)
        writer = FrameWriter(os.path.join(out, "frames"), args.jpeg_quality)

    bus.connect()  # handshake only; SO101Follower.connect() would also rewrite PID gains
    log = {k: [] for k in ("t_frame", "t_joint", "pos_raw", "goal_raw", "load", "target_id")}
    stop_reason, torque_on = "completed", False
    try:
        mode = read("Operating_Mode")
        if mode.any():
            raise RuntimeError(f"servos not all in position mode: {dict(zip(names, mode))}")
        start = read("Present_Position")
        # Joints resting on a hard stop sit slightly outside the calibrated range; never command past it.
        hold = np.clip(start, [c.range_min for c in calib], [c.range_max for c in calib])
        centre = np.clip(start, lo, hi)
        span = args.max_deg * TICKS_PER_DEG
        box_lo, box_hi = np.maximum(lo, centre - span), np.minimum(hi, centre + span)
        print("joints      ", names)
        print("start (raw) ", start.tolist())
        print("moving      ", [n for n, m in zip(names, moving) if m])
        print("box lo      ", np.where(moving, box_lo, hold).astype(int).tolist())
        print("box hi      ", np.where(moving, box_hi, hold).astype(int).tolist())

        for n in names:
            bus.write("Acceleration", n, args.accel, normalize=False)
            bus.write("Goal_Velocity", n, args.speed, normalize=False)
        goal = hold.copy()
        write_goal(goal)
        for n in names:
            bus.enable_torque(n)
            time.sleep(0.1)
        torque_on = True
        time.sleep(0.5)

        def new_target(prev):
            step = args.step_deg * TICKS_PER_DEG
            t = rng.uniform(np.maximum(box_lo, prev - step), np.minimum(box_hi, prev + step))
            return np.where(moving, np.round(t), hold)

        target_id, arrived_at = 0, None
        goal = new_target(goal)
        write_goal(goal)
        best_err, best_t = np.inf, time.perf_counter()
        t_start = t_report = t_temp = time.perf_counter()
        last_frame_t, i = None, 0
        period = 1.0 / args.rate
        while time.perf_counter() - t_start < args.seconds:
            if cam is not None:
                if cam.error:
                    raise RuntimeError(cam.error)
                ts, frames = cam.latest(1, 1)
                if ts[-1] == last_frame_t:
                    time.sleep(0.001)
                    continue
                last_frame_t, frame = ts[-1], frames[0]
            else:
                time.sleep(max(0.0, t_start + i * period - time.perf_counter()))
                last_frame_t, frame = time.perf_counter(), None

            pos, load = read("Present_Position"), read("Present_Load")
            now = time.perf_counter()
            for k, v in zip(log, (last_frame_t, now, pos, goal.copy(), load, target_id)):
                log[k].append(v)
            if writer is not None:
                writer.put(i, frame)
            i += 1

            err = np.abs(pos - goal)[moving].max()
            if np.abs(load).max() > args.max_load:
                raise RuntimeError(f"load {dict(zip(names, load.tolist()))} above {args.max_load}")
            if err < best_err - 3:
                best_err, best_t = err, now
            if err <= args.tol:
                arrived_at = arrived_at or now
                if now - arrived_at >= args.dwell:
                    goal, target_id, arrived_at = new_target(goal), target_id + 1, None
                    write_goal(goal)
                    best_err, best_t = np.inf, now
            elif now - best_t > args.stall_s:
                raise RuntimeError(f"stalled {err:.0f} ticks from goal: {dict(zip(names, (pos - goal).tolist()))}")

            if now - t_temp > 2.0:
                temp, t_temp = read("Present_Temperature"), now
                if temp.max() > args.max_temp:
                    raise RuntimeError(f"servo temperature {temp.tolist()} above {args.max_temp} C")
            if now - t_report >= 2.0:
                print(
                    f"t={now - t_start:5.1f}s samples={i:5d} targets={target_id:3d} "
                    f"pos={pos[moving].tolist()} |load|max={int(np.abs(load).max())}",
                    flush=True,
                )
                t_report = now
    except KeyboardInterrupt:
        stop_reason = "interrupted"
    except Exception as e:  # noqa: BLE001 - any failure must still park and release the arm
        stop_reason = f"safety stop: {e}"
    finally:
        try:
            if torque_on:
                write_goal(hold)
                t0 = time.perf_counter()
                while time.perf_counter() - t0 < 6 and np.abs(read("Present_Position") - hold).max() > args.tol:
                    time.sleep(0.05)
                end = read("Present_Position")
                print("return error (ticks)", (end - start).tolist(), "| temps", read("Present_Temperature").tolist())
        finally:
            bus.disable_torque()
            print("torque enable", read("Torque_Enable").tolist())
            bus.disconnect(disable_torque=False)
            if cam is not None:
                cam.close()
                writer.close()

    n = len(log["t_frame"])
    arrays = {k: np.array(v) for k, v in log.items()}
    if n:
        arrays["pos_deg"] = ticks_to_deg(arrays["pos_raw"], calib)
    np.savez(os.path.join(out, "log.npz"), **arrays)
    meta = {
        "joints": names,
        "moving": args.joints,
        "calibration": {n: vars(c) for n, c in zip(names, calib)},
        "start_raw": start.tolist() if n else None,
        "stop_reason": stop_reason,
        "samples": n,
        "frames_dropped_by_writer": writer.dropped if writer else None,
        "args": vars(args),
    }
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n== {stop_reason}")
    if n > 1:
        dur = arrays["t_joint"][-1] - arrays["t_joint"][0]
        lag = (arrays["t_joint"] - arrays["t_frame"]) * 1000
        travel = np.abs(np.diff(arrays["pos_raw"], axis=0)).sum(axis=0) / TICKS_PER_DEG
        print(f"samples        {n} in {dur:.1f} s = {(n - 1) / dur:.1f} Hz, {int(arrays['target_id'][-1])} targets")
        print(f"joint-vs-frame median {np.median(lag):.1f} ms  p95 {np.percentile(lag, 95):.1f} ms")
        print("travel (deg)   " + "  ".join(f"{nm}:{t:.0f}" for nm, t in zip(names, travel)))
        print(f"peak |load|    {np.abs(arrays['load']).max(axis=0).tolist()}")
    print(f"saved to       {out}")
    if stop_reason.startswith("safety"):
        sys.exit(1)


if __name__ == "__main__":
    main()
