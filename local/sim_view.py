"""Open the simulated SO-101 in MuJoCo's interactive viewer, on the screen of the machine it runs on.

    # at the desktop itself
    ~/projects/lerobot/.venv/bin/python -m local.sim_view
    ~/projects/lerobot/.venv/bin/python -m local.sim_view --demo     # random walk, as sim_collect.py does

    # from the laptop, onto the desktop's screen (someone must be logged in there)
    ssh aorus 'cd projects/vjepa2 && DISPLAY=:0 XAUTHORITY=/run/user/1000/gdm/Xauthority \
        nohup ~/projects/lerobot/.venv/bin/python -m local.sim_view --demo > /tmp/sim_view.log 2>&1 &'

The scene is the same one sim_collect.py records (table, lamp, black arm), opened on the same side view
the dataset camera uses. The arm starts at the ready pose and holds it. Without --demo, drag the sliders in
the right-hand "Control" panel to move each servo goal (radians); double-click a link and Ctrl+right-drag
to push it. Mouse: left-drag rotates, right-drag pans, scroll zooms. Close the window to quit.
"""

import argparse
import os
import time

os.environ["MUJOCO_GL"] = "glfw"  # a real window, not the headless EGL used for recording

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402
import numpy as np  # noqa: E402

from local.sim_arm import READY_DEG, build_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="drive a random walk around the ready pose")
    ap.add_argument("--max-deg", type=float, default=20.0)
    ap.add_argument("--speed", type=float, default=17.6, help="goal ramp in demo mode, deg/s")
    ap.add_argument("--kp", type=float, default=17.8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model, cam = build_model(kp=args.kp)
    data = mujoco.MjData(model)
    ready = np.radians(READY_DEG)
    data.qpos[:] = ready
    data.ctrl[:] = ready
    mujoco.mj_forward(model, data)

    rng = np.random.default_rng(args.seed)
    lo = np.maximum(ready - np.radians(args.max_deg), model.jnt_range[:, 0])
    hi = np.minimum(ready + np.radians(args.max_deg), model.jnt_range[:, 1])
    target, step = ready.copy(), np.radians(args.speed) * model.opt.timestep

    with mujoco.viewer.launch_passive(model, data) as v:
        v.cam.lookat[:] = cam["lookat"]
        v.cam.distance, v.cam.azimuth, v.cam.elevation = cam["distance"], cam["azimuth"], cam["elevation"]
        while v.is_running():
            t0 = time.perf_counter()
            with v.lock():
                for _ in range(10):
                    if args.demo:
                        if np.abs(data.ctrl - target).max() < 1e-9 and np.abs(data.qvel).max() < 0.03:
                            target = np.clip(target + np.radians(rng.uniform(-8, 8, model.nu)), lo, hi)
                        data.ctrl[:] += np.clip(target - data.ctrl, -step, step)
                    mujoco.mj_step(model, data)
            v.sync()
            time.sleep(max(0.0, 10 * model.opt.timestep - (time.perf_counter() - t0)))  # real time


if __name__ == "__main__":
    main()
