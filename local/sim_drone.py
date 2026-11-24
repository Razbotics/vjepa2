"""Simulated Crazyflie 2 in MuJoCo: a room to fly in, an onboard camera, and a position controller.

    cd ~/projects/vjepa2
    ~/projects/lerobot/.venv/bin/python -m local.sim_drone            # flight test, writes a report
    ~/projects/lerobot/.venv/bin/python -m local.sim_drone --view     # same flight in the viewer

The airframe (meshes, mass 27 g, inertia) is MuJoCo Menagerie's bitcraze_crazyflie_2. Its actuators are
not used: they push the body with one thrust and three torques, and the torques top out at 1e-5 N m, which
spins the body up at 0.4 rad/s^2, about a thousandth of the real drone. They are replaced by the real
layout: four rotors at the tips of a 92 mm (motor to motor) X frame, each pushing along the body z axis and
twisting the body with its drag torque, with a first-order spin-up lag. Each rotor's command is its thrust
in newtons.

Control runs at the physics rate (about 510 Hz), as on the real flight controller: a position loop turns the
target position into a desired thrust direction, and a geometric attitude loop (Lee, Leok and McClamroch,
2010) turns that into body torques. The four rotor thrusts are then solved from total thrust and torques
(the "mixer"). Later, a learned planner will give targets to this controller, not raw rotor thrusts: the
drone cannot hold still on its own, so something fast must keep it upright underneath.
"""

import os
import sys

if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from local.sim_arm import ASSETS  # noqa: E402

CF2_DIR = os.path.join(ASSETS, "mujoco_menagerie", "bitcraze_crazyflie_2")
MASS = 0.027  # kg
G = 9.81
ARM = 0.0325  # m, rotor offset along body x and y (46 mm from the centre on the diagonal)
F_MAX = 0.15  # N per rotor, thrust-to-weight about 2.3
K_YAW = 0.006  # m, rotor drag torque per newton of thrust (commonly used identified value)
TAU_MOTOR = 0.03  # s, rotor spin-up time constant
# Rotors in the Crazyflie's order: front-right, back-right, back-left, front-left (body x forward, y left).
# Diagonal pairs spin the same way, so each rotor's reaction torque on the body alternates in sign.
ROTORS = np.array([[ARM, -ARM], [-ARM, -ARM], [-ARM, ARM], [ARM, ARM]])
SPIN = np.array([-1.0, 1.0, -1.0, 1.0])
ROOM = 3.0  # m, half-width of the square room
FPV_TILT = 15.0  # deg, onboard camera pitched down
# Room furniture: pillars (x, y, colour) are cylinders of radius 0.12 m, 2 m tall; boxes (x, y, half-size, colour)
PILLAR_R = 0.12
PILLARS = [(1.6, 1.2, "1 0.35 0.2"), (-1.4, 1.7, "0.2 0.6 1"), (1.9, -1.5, "1 0.85 0.15"),
           (-1.8, -1.1, "0.6 0.3 0.9"), (0.3, 2.3, "0.2 0.8 0.4"), (-0.4, -2.2, "0.9 0.3 0.6")]
BOXES = [(0.9, -0.6, 0.15, "0.85 0.5 0.3"), (-0.8, 0.5, 0.25, "0.35 0.45 0.55"), (2.3, 0.4, 0.2, "0.5 0.7 0.3")]


def _scene_xml():
    """Menagerie's cf2.xml with the rotors swapped in, wrapped in a furnished room."""
    src = open(os.path.join(CF2_DIR, "cf2.xml")).read()
    src = src.replace('meshdir="assets"', f'meshdir="{os.path.join(CF2_DIR, "assets")}"')
    head, rest = src.split("<actuator>", 1)
    _, tail = rest.split("</actuator>", 1)
    motors = "\n".join(
        f'    <general name="rotor{i + 1}" site="rotor{i + 1}" gear="0 0 1 0 0 {s * K_YAW}" ctrlrange="0 {F_MAX}" '
        f'dyntype="filterexact" dynprm="{TAU_MOTOR}" gainprm="1"/>'
        for i, s in enumerate(SPIN)
    )
    src = head + "<actuator>\n" + motors + "\n  </actuator>" + tail
    t = np.radians(FPV_TILT)
    c = f"0 -1 0 {np.sin(t):.4f} 0 {np.cos(t):.4f}"  # looks along body +x, pitched down
    sites = "".join(f'\n      <site name="rotor{i + 1}" pos="{x} {y} 0.015"/>' for i, (x, y) in enumerate(ROTORS))
    src = src.replace(
        '<site name="actuation"/>',
        f'<site name="actuation"/>{sites}\n      <camera name="fpv" pos="0.02 0 0.012" xyaxes="{c}" fovy="70"/>',
    )
    src = src.replace('<key name="hover" qpos="0 0 0.1 1 0 0 0" ctrl="0.26487 0 0 0"/>', "")

    r, h = ROOM, 1.25  # walls 2.5 m tall
    furniture = "".join(
        f'\n    <geom type="cylinder" size="{PILLAR_R} 1" pos="{x} {y} 1" rgba="{c} 1"/>' for x, y, c in PILLARS
    ) + "".join(
        f'\n    <geom type="box" size="{s} {s} {s}" pos="{x} {y} {s}" rgba="{c} 1"/>' for x, y, s, c in BOXES
    )
    room = f"""
  <visual>
    <headlight diffuse="0.5 0.5 0.5" ambient="0.35 0.35 0.35" specular="0 0 0"/>
    <global offwidth="1280" offheight="720"/>
    <quality shadowsize="4096"/>
  </visual>
  <asset>
    <texture type="2d" name="floor" builtin="checker" rgb1="0.55 0.5 0.45" rgb2="0.35 0.32 0.3" width="512"
      height="512" mark="edge" markrgb="0.2 0.2 0.2"/>
    <material name="floor" texture="floor" texrepeat="12 12" texuniform="false"/>
    <texture type="2d" name="wallN" builtin="checker" rgb1="0.8 0.78 0.72" rgb2="0.65 0.7 0.75"
      width="256" height="256"/>
    <material name="wallN" texture="wallN" texrepeat="6 3"/>
    <texture type="2d" name="wallS" builtin="checker" rgb1="0.78 0.82 0.7" rgb2="0.55 0.65 0.55"
      width="256" height="256"/>
    <material name="wallS" texture="wallS" texrepeat="6 3"/>
    <texture type="2d" name="wallE" builtin="checker" rgb1="0.85 0.75 0.7" rgb2="0.7 0.55 0.5"
      width="256" height="256"/>
    <material name="wallE" texture="wallE" texrepeat="6 3"/>
    <texture type="2d" name="wallW" builtin="checker" rgb1="0.75 0.75 0.85" rgb2="0.55 0.55 0.7"
      width="256" height="256"/>
    <material name="wallW" texture="wallW" texrepeat="6 3"/>
    <texture type="2d" name="ceiling" builtin="checker" rgb1="0.9 0.9 0.88" rgb2="0.8 0.8 0.78"
      width="256" height="256"/>
    <material name="ceiling" texture="ceiling" texrepeat="4 4"/>
  </asset>
  <worldbody>
    <light pos="0 0 2.45" dir="0 0 -1" diffuse="0.6 0.6 0.6" castshadow="true"/>
    <light pos="2 -2 2.2" dir="-0.5 0.5 -1" diffuse="0.3 0.3 0.3" castshadow="false"/>
    <geom name="floor" type="plane" size="{r} {r} 0.05" material="floor"/>
    <geom type="box" size="{r} 0.05 {h}" pos="0 {r} {h}" material="wallN"/>
    <geom type="box" size="{r} 0.05 {h}" pos="0 {-r} {h}" material="wallS"/>
    <geom type="box" size="0.05 {r} {h}" pos="{r} 0 {h}" material="wallE"/>
    <geom type="box" size="0.05 {r} {h}" pos="{-r} 0 {h}" material="wallW"/>
    <geom type="box" size="{r} {r} 0.05" pos="0 0 {2 * h + 0.05}" material="ceiling"/>{furniture}
    <camera name="corner" pos="-2.6 -2.6 2.0" xyaxes="0.707 -0.707 0 0.35 0.35 0.87" fovy="60"/>
  </worldbody>
"""
    return src.replace("</mujoco>", room + "</mujoco>")


def allocation():
    """[total thrust, roll torque, pitch torque, yaw torque] = A @ rotor thrusts."""
    x, y = ROTORS[:, 0], ROTORS[:, 1]
    return np.stack([np.ones(4), y, -x, SPIN * K_YAW])


def _vee(m):
    return np.array([m[2, 1], m[0, 2], m[1, 0]])


class Controller:
    """Position loop + geometric attitude loop + mixer. Returns the four rotor thrusts in newtons."""

    def __init__(self, kp=(6.0, 6.0, 8.0), kd=(4.0, 4.0, 5.0), k_att=(250.0, 250.0, 60.0),
                 k_rate=(22.0, 22.0, 12.0), max_tilt=30.0):
        self.kp, self.kd = np.array(kp), np.array(kd)
        self.k_att, self.k_rate = np.array(k_att), np.array(k_rate)
        self.J = np.array([2.3951e-5, 2.3951e-5, 3.2347e-5])
        self.mix = np.linalg.inv(allocation())
        self.tan_tilt = np.tan(np.radians(max_tilt))

    def __call__(self, p, v, R, w, p_ref, v_ref=np.zeros(3), yaw_ref=0.0, a_ref=np.zeros(3)):
        a = a_ref + self.kp * (p_ref - p) + self.kd * (v_ref - v)
        F = MASS * (a + np.array([0.0, 0.0, G]))
        F[2] = max(F[2], 0.2 * MASS * G)
        h = np.linalg.norm(F[:2])
        if h > self.tan_tilt * F[2]:  # don't ask for more than max_tilt of lean
            F[:2] *= self.tan_tilt * F[2] / h
        thrust = F @ R[:, 2]

        b3 = F / np.linalg.norm(F)
        b2 = np.cross(b3, [np.cos(yaw_ref), np.sin(yaw_ref), 0.0])
        b2 /= np.linalg.norm(b2)
        Rd = np.column_stack([np.cross(b2, b3), b2, b3])
        e_R = 0.5 * _vee(Rd.T @ R - R.T @ Rd)
        torque = self.J * (-self.k_att * e_R - self.k_rate * w) + np.cross(w, self.J * w)
        return np.clip(self.mix @ np.concatenate([[thrust], torque]), 0.0, F_MAX)


class SimDrone:
    def __init__(self, width=640, height=360, fps=30.0):
        self.model = mujoco.MjModel.from_xml_string(_scene_xml())
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.ctrl = Controller()
        self.fps = fps
        self.substeps = int(np.ceil(1.0 / (fps * self.model.opt.timestep)))  # 17 at 30 fps
        self.model.opt.timestep = 1.0 / (fps * self.substeps)  # 1.96 ms, so frames are exactly 1/fps apart
        self._gyro = self.model.sensor_adr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "body_gyro")]
        self._R = np.zeros(9)

    def reset(self, pos=(0.0, 0.0, 0.0), yaw=0.0, settle=0.5):
        """Place the drone, motors off; with z=0 it is dropped from 5 cm and settles on the floor."""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = pos
        if pos[2] <= 0:
            self.data.qpos[2] = 0.05
        self.data.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        for _ in range(int(settle / self.model.opt.timestep)):
            mujoco.mj_step(self.model, self.data)
        self.data.time = 0.0

    @property
    def pos(self):
        return self.data.qpos[:3].copy()

    @property
    def vel(self):
        return self.data.qvel[:3].copy()  # world frame for a free joint

    @property
    def R(self):
        mujoco.mju_quat2Mat(self._R, self.data.qpos[3:7])
        return self._R.reshape(3, 3).copy()

    @property
    def rate(self):
        return self.data.sensordata[self._gyro:self._gyro + 3].copy()  # body frame, rad/s

    @property
    def rpy_deg(self):
        R = self.R
        return np.degrees([np.arctan2(R[2, 1], R[2, 2]), -np.arcsin(np.clip(R[2, 0], -1, 1)),
                           np.arctan2(R[1, 0], R[0, 0])])

    def hover_at(self, pos, yaw=0.0):
        """Place the drone in the air, level and still, rotors already spinning at hover thrust."""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = pos
        self.data.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        self.data.act[:] = MASS * G / 4
        mujoco.mj_forward(self.model, self.data)

    def touching(self):
        """True if any part of the drone is in contact with the world."""
        n, b = self.data.ncon, self.model.geom_bodyid
        if n == 0:
            return False
        return bool(np.any((b[self.data.contact.geom1[:n]] == 1) | (b[self.data.contact.geom2[:n]] == 1)))

    def tick(self, p_ref, v_ref=np.zeros(3), yaw_ref=0.0, motors_off=False, a_ref=np.zeros(3)):
        """One physics step (2 ms) with the controller in the loop."""
        f = np.zeros(4) if motors_off else self.ctrl(self.pos, self.vel, self.R, self.rate, np.asarray(p_ref),
                                                     np.asarray(v_ref), yaw_ref, np.asarray(a_ref))
        self.data.ctrl[:] = f
        mujoco.mj_step(self.model, self.data)
        return f

    def render(self, camera="fpv"):
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render()[..., ::-1]


def mission(z0, alt=1.0):
    """Test flight as (target position, target yaw, seconds to hold before the next leg) legs."""
    return [
        ("take off", (0, 0, alt), 0.0, 4.0),
        ("square 1", (1, 0, alt), 0.0, 1.0),
        ("square 2", (1, 1, alt), 0.0, 1.0),
        ("square 3", (0, 1, alt), 0.0, 1.0),
        ("square 4", (0, 0, alt), 0.0, 1.0),
        ("yaw 90", (0, 0, alt), np.pi / 2, 3.0),
        ("push", (0, 0, alt), np.pi / 2, 3.0),
        ("land", (0, 0, z0 + 0.02), np.pi / 2, 1.5),
    ]


def fly(drone, live=None, alt=1.0, speed=0.5, push=0.05, on_frame=None):
    """Runs the test mission; the target moves along each leg at `speed`, then holds. Returns a log."""
    m, d = drone.model, drone.data
    drone.reset()
    z0 = drone.pos[2]
    dt = m.opt.timestep
    log = {k: [] for k in ("t", "pos", "ref", "rpy", "motors", "leg")}
    ref = drone.pos.copy()
    frame_every = drone.substeps
    n = 0
    log_legs = mission(z0, alt)
    for li, (name, goal, yaw, hold) in enumerate(log_legs):
        goal = np.array(goal, dtype=float)
        start, yaw0 = ref.copy(), (log["rpy"][-1][2] if log["rpy"] else 0.0)
        travel = np.linalg.norm(goal - start) / speed
        t_leg = 0.0
        while t_leg < travel + hold:
            s = min(1.0, t_leg / travel) if travel > 0 else 1.0
            ref = start + s * (goal - start)
            v_ref = (goal - start) / travel if 0 < s < 1 else np.zeros(3)
            yaw_ref = np.radians(yaw0) + min(1.0, t_leg / 1.5) * (yaw - np.radians(yaw0)) if name == "yaw 90" else yaw
            if name == "push" and 1.0 <= t_leg < 1.1:  # a 0.1 s sideways shove, like a flick of the hand
                d.xfrc_applied[1, :3] = [push, push, 0.0]
            else:
                d.xfrc_applied[1, :3] = 0.0
            f = drone.tick(ref, v_ref, yaw_ref)
            if n % 10 == 0:
                log["t"].append(d.time), log["pos"].append(drone.pos), log["ref"].append(ref.copy())
                log["rpy"].append(drone.rpy_deg), log["motors"].append(f), log["leg"].append(li)
            if n % frame_every == 0:
                if live is not None:
                    live(drone, name)
                if on_frame is not None:
                    on_frame(drone, name, t_leg)
            n += 1
            t_leg += dt
    for _ in range(int(1.0 / dt)):  # motors off on the ground
        f = drone.tick(drone.pos, motors_off=True)
        if n % 10 == 0:
            log["t"].append(d.time), log["pos"].append(drone.pos), log["ref"].append(drone.pos)
            log["rpy"].append(drone.rpy_deg), log["motors"].append(f), log["leg"].append(len(log_legs))
        if live is not None and n % frame_every == 0:
            live(drone, "landed")
        n += 1
    return {k: np.array(v) for k, v in log.items()}, z0


def main():
    import argparse
    import json
    import time

    import cv2

    ap = argparse.ArgumentParser()
    ap.add_argument("--view", action="store_true", help="fly the mission in the MuJoCo viewer, in real time")
    ap.add_argument("--loop", action="store_true", help="--view: repeat the mission until the window closes")
    ap.add_argument("--out", default=None)
    ap.add_argument("--video", action="store_true", help="also save flight.mp4: onboard | chase camera")
    args = ap.parse_args()

    if args.view:
        import mujoco.viewer

        drone = SimDrone(width=320, height=180)
        v = mujoco.viewer.launch_passive(drone.model, drone.data, show_left_ui=False, show_right_ui=False)
        v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING  # follows the drone; drag to orbit around it
        v.cam.trackbodyid = 1
        v.cam.distance, v.cam.azimuth, v.cam.elevation = 1.0, -150.0, -20.0
        clock = {"next": time.perf_counter()}

        def live(dr, name):
            if not v.is_running():
                raise SystemExit
            img = np.ascontiguousarray(dr.render("fpv")[..., ::-1])  # RGB inset: what the drone's camera sees
            v.set_images((mujoco.MjrRect(0, 0, img.shape[1] * 2, img.shape[0] * 2), cv2.resize(img, None, fx=2, fy=2)))
            p, r = dr.pos, dr.rpy_deg
            v.set_texts((mujoco.mjtFont.mjFONT_BIG, mujoco.mjtGridPos.mjGRID_TOPLEFT, name,
                         f"x {p[0]:+.2f}  y {p[1]:+.2f}  z {p[2]:.2f} m   roll {r[0]:+.0f}  pitch {r[1]:+.0f}"
                         f"  yaw {r[2]:+.0f} deg"))
            v.sync()
            clock["next"] += 1.0 / dr.fps
            now = time.perf_counter()
            if now > clock["next"] + 0.25:
                clock["next"] = now
            time.sleep(max(0.0, clock["next"] - now))

        try:
            while v.is_running():
                fly(drone, live=live)
                if not args.loop:
                    break
        except SystemExit:
            pass
        v.close()
        return

    drone = SimDrone()
    out = args.out or os.path.join("local", "runs", "sim_drone_test_" + time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out, exist_ok=True)
    shots = {}
    wanted = {("take off", 3.5), ("square 2", 1.5), ("yaw 90", 3.5), ("push", 1.05), ("push", 1.6)}

    writer = None
    if args.video:
        writer = cv2.VideoWriter(os.path.join(out, "flight.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), drone.fps,
                                 (2 * drone.renderer.width, drone.renderer.height))

    def grab(dr, name, t_leg):
        if writer is not None:
            frame = np.hstack([dr.render("fpv"), dr.render("track")])
            p = dr.pos
            cv2.putText(frame, f"{name}   z {p[2]:.2f} m", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            writer.write(frame)
        for leg, t in wanted:
            if leg == name and (leg, t) not in shots and t_leg >= t:
                shots[(leg, t)] = (dr.render("fpv"), dr.render("track"))

    t0 = time.perf_counter()
    log, z0 = fly(drone, on_frame=grab)
    wall = time.perf_counter() - t0
    if writer is not None:
        writer.release()

    legs = [m[0] for m in mission(z0)]
    err = np.linalg.norm(log["pos"] - log["ref"], axis=1)
    tilt = np.degrees(np.arccos(np.clip(np.cos(np.radians(log["rpy"][:, 0])) * np.cos(np.radians(log["rpy"][:, 1])),
                                        -1, 1)))
    res = {"rest_height_m": float(z0), "sim_seconds": float(log["t"][-1]), "wall_seconds": wall}
    print(f"rest height {z0 * 100:.1f} cm | {res['sim_seconds']:.1f} s of flight simulated in {wall:.1f} s "
          f"(with 2 cameras at 30 fps)")
    print(f"{'leg':10s} {'track err mean/max (cm)':>24s} {'max tilt (deg)':>15s} {'motors min-max (N)':>20s}")
    for li, name in enumerate(legs):
        k = log["leg"] == li
        res[name] = {"err_mean_cm": float(100 * err[k].mean()), "err_max_cm": float(100 * err[k].max()),
                     "tilt_max_deg": float(tilt[k].max()),
                     "motor_min": float(log["motors"][k].min()), "motor_max": float(log["motors"][k].max())}
        r = res[name]
        print(f"{name:10s} {r['err_mean_cm']:12.1f} / {r['err_max_cm']:5.1f} {r['tilt_max_deg']:15.1f} "
              f"{r['motor_min']:11.3f} - {r['motor_max']:.3f}")
    hov = (log["leg"] == 0) & (log["t"] > log["t"][log["leg"] == 0][-1] - 2.0)  # last 2 s of the take-off hold
    res["hover_rms_mm"] = float(1000 * np.sqrt((err[hov] ** 2).mean()))
    res["hover_thrust_per_rotor_N"] = float(log["motors"][hov].mean())
    print(f"hover: position error RMS {res['hover_rms_mm']:.1f} mm, thrust per rotor "
          f"{res['hover_thrust_per_rotor_N']:.4f} N (weight/4 = {MASS * G / 4:.4f} N)")
    yaw_end = log["rpy"][log["leg"] == legs.index("yaw 90")][-1, 2]
    print(f"yaw after the turn {yaw_end:.1f} deg | resting height after landing {log['pos'][-1, 2] * 100:.1f} cm")
    res["yaw_end_deg"] = float(yaw_end)

    rows = [np.hstack(shots[k]) for k in sorted(shots, key=lambda k: (legs.index(k[0]), k[1]))]
    cv2.imwrite(os.path.join(out, "views.jpg"), np.vstack(rows))
    json.dump(res, open(os.path.join(out, "result.json"), "w"), indent=2)
    np.savez_compressed(os.path.join(out, "log.npz"), **log)
    print(f"saved to {out} (views.jpg: onboard camera | chase camera)")


if __name__ == "__main__":
    main()
