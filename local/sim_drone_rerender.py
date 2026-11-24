"""Render an existing sim drone run again from its logged poses, with another onboard camera.

    cd ~/projects/vjepa2
    ~/projects/lerobot/.venv/bin/python -m local.sim_drone_rerender local/data/sim_drone/hour_seed1 \
        local/data/sim_drone/hour_seed1_level --camera fpv_level

The flight is the same (log.npz is copied unchanged); only the images differ. Each frame puts the drone at its
logged position and attitude and renders it. Nothing else in the scene moves, so this gives exactly the frames
the other camera would have recorded during the original flight.
"""

import argparse
import json
import os
import shutil
import time

import cv2
import mujoco
import numpy as np

from local.sim_drone import SimDrone


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--camera", default="fpv_level", choices=["fpv", "fpv_level"])
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.src, "meta.json")))
    log = np.load(os.path.join(args.src, "log.npz"))
    os.makedirs(os.path.join(args.dst, "frames"))
    shutil.copy(os.path.join(args.src, "log.npz"), args.dst)
    ma = meta["args"]
    drone = SimDrone(width=ma["width"], height=ma["height"], fps=ma["fps"])
    pos, quat = log["pos"], log["quat"]
    t0 = time.perf_counter()
    for i in range(len(pos)):
        drone.data.qpos[:3], drone.data.qpos[3:7] = pos[i], quat[i]
        mujoco.mj_kinematics(drone.model, drone.data)
        cv2.imwrite(os.path.join(args.dst, "frames", f"{i:06d}.jpg"), drone.render(args.camera),
                    [cv2.IMWRITE_JPEG_QUALITY, ma["jpeg_quality"]])
        if (i + 1) % 18000 == 0:
            print(f"  {i + 1} of {len(pos)} frames in {time.perf_counter() - t0:.0f} s", flush=True)
    meta["camera"]["name"] = args.camera
    meta["rerendered_from"] = os.path.abspath(args.src)
    with open(os.path.join(args.dst, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"{len(pos)} frames with camera {args.camera} in {time.perf_counter() - t0:.0f} s -> {args.dst}")


if __name__ == "__main__":
    main()
