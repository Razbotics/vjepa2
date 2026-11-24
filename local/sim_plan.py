"""Closed-loop image-goal reaching on the simulated SO-101, the same loop arm_plan.py runs on the real arm.

    cd ~/projects/vjepa2
    ~/projects/lerobot/.venv/bin/python -m local.sim_plan local/data/sim_arm/hour_seed1
    ~/projects/lerobot/.venv/bin/python -m local.sim_plan local/data/sim_arm/hour_seed1 --policy random
    ~/projects/lerobot/.venv/bin/python -m local.sim_plan local/data/sim_arm/hour_seed1 --pred train600s

Each episode:
  1. The arm is placed at a random pose in the training box and the camera records the goal image.
  2. It is placed at a different random pose (at least --min-start deg away on average), the start.
  3. For --steps control steps of 0.5 s: render and encode the current frame, plan with CEM through the
     predictor (arm_plan.cem_plan), command the first action. The servo goal ramps at the collection
     speed, so actions look like the ones in the training data.
The planner sees only images and its own joint angles; the goal's angles are kept for scoring. Goals and
starts depend only on --seed, so different policies and predictors face the same episodes.

Reports joint error and gripper-tip distance to the goal, before and after. Writes goal, start and final
frames and a montage to local/runs/sim_plan_<policy>_<pred>_<ts>/.

--view also plays the run in MuJoCo's viewer at real-time speed, with the goal pose drawn as a green
ghost arm. The ghost exists only in the viewer: the planner's camera renders its own scene and never sees
it. From the laptop, onto the desktop's screen:

    ssh aorus 'cd projects/vjepa2 && DISPLAY=:0 XAUTHORITY=/run/user/1000/gdm/Xauthority nohup \
        ~/projects/lerobot/.venv/bin/python -m local.sim_plan local/data/sim_arm/hour_seed1 --view \
        > /tmp/sim_plan_view.log 2>&1 &'
"""

import argparse
import json
import os
import time

import cv2
import numpy as np
import torch

from local.arm_plan import cem_plan, encode_frames
from local.arm_predictor import Predictor
from local.sim_arm import CAMERA, JOINTS, SimArm
from local.vjepa_local import IMAGENET_MEAN, IMAGENET_STD, load_model, pick_device


class Live:
    """Shows the simulation in MuJoCo's viewer, paced to real time, with the goal pose as a ghost arm."""

    GHOST_RGBA = (0.25, 0.9, 0.35, 0.35)

    def __init__(self, arm, camera):
        import mujoco
        import mujoco.viewer

        self.mj, self.arm = mujoco, arm
        self.v = mujoco.viewer.launch_passive(arm.model, arm.data, show_left_ui=False, show_right_ui=False)
        self.v.cam.lookat[:] = camera["lookat"]
        self.v.cam.distance, self.v.cam.azimuth, self.v.cam.elevation = (
            camera["distance"], camera["azimuth"], camera["elevation"])
        self.ghost = mujoco.MjData(arm.model)
        self.dt, self.next = 1.0 / arm.fps, time.perf_counter()

    def running(self):
        return self.v.is_running()

    def set_goal(self, q_deg):
        """Draw the arm's moving parts at q_deg, see-through green, into the viewer's extra scene."""
        mj, scn = self.mj, self.v.user_scn
        self.ghost.qpos[:] = np.radians(q_deg)
        mj.mj_kinematics(self.arm.model, self.ghost)
        with self.v.lock():
            scn.ngeom = 0
            mj.mjv_addGeoms(self.arm.model, self.ghost, mj.MjvOption(), mj.MjvPerturb(),
                            mj.mjtCatBit.mjCAT_DYNAMIC, scn)
            for i in range(scn.ngeom):
                scn.geoms[i].rgba[:] = self.GHOST_RGBA

    def text(self, left, right):
        mj = self.mj
        self.v.set_texts((mj.mjtFont.mjFONT_BIG, mj.mjtGridPos.mjGRID_TOPLEFT, left, right))

    def frame(self):
        """Call after each simulated frame: show it, and wait until its real-time slot."""
        self.v.sync()
        self.next += self.dt
        now = time.perf_counter()
        if now > self.next + 0.25:  # fell behind (planning, encoding): restart the clock, don't fast-forward
            self.next = now
        time.sleep(max(0.0, self.next - now))

    def hold(self, seconds):
        self.v.sync()
        time.sleep(seconds)
        self.next = time.perf_counter()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="sim run folder holding pred_<model>[_<pred>].pt")
    ap.add_argument("--model", default="vjepa2_1_vit_large_384")
    ap.add_argument("--pred", default="", help="checkpoint tag, e.g. train600s (default: trained on all)")
    ap.add_argument("--policy", default="plan", choices=["plan", "random"])
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--steps", type=int, default=20, help="control steps per episode")
    ap.add_argument("--horizon", type=int, default=2)
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--elites", type=int, default=24)
    ap.add_argument("--sigma", type=float, default=3.0)
    ap.add_argument("--max-step", type=float, default=5.0)
    ap.add_argument("--min-start", type=float, default=6.0)
    ap.add_argument("--min-height", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=100, help="episode seed (training runs used 0 and 1)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--view", action="store_true", help="play the run live in the MuJoCo viewer")
    ap.add_argument("--hold", type=float, default=1.5, help="--view: seconds to pause on each new goal")
    args = ap.parse_args()

    device = pick_device(None)
    meta = json.load(open(os.path.join(args.run, "meta.json")))
    tag = args.model + (f"_{args.pred}" if args.pred else "")
    ckpt = torch.load(os.path.join(args.run, f"pred_{tag}.pt"), map_location=device)
    joints, cfg = ckpt["joints"], ckpt["args"]
    assert joints == JOINTS, f"predictor joints {joints} must be all six"
    qmu, qsd, zmu, zsd = (ckpt[k].to(device) for k in ("qmu", "qsd", "zmu", "zsd"))
    k = cfg["k"]
    model = Predictor(zmu.shape[-1], len(joints), cfg["width"], cfg["depth"]).to(device).eval()
    model.load_state_dict(ckpt["state_dict"])
    encoder = load_model(args.model, device=device)[0]
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    margs = meta["args"]
    arm = SimArm(width=margs["width"], height=margs["height"], fps=margs["fps"], kp=margs["kp"])
    lo, hi = (np.array(b) for b in meta["box_deg"])
    lo_t, hi_t = (torch.tensor(b, device=device, dtype=torch.float32) for b in (lo, hi))
    ramp = margs["speed"] / margs["fps"]
    live = Live(arm, meta.get("camera", CAMERA)) if args.view else None

    def see():
        frame = arm.render()
        return encode_frames([frame], encoder, mean, std, zmu, zsd, device), frame

    ep_rng = np.random.default_rng(args.seed)  # goals and starts: identical across policies
    act_rng = np.random.default_rng(args.seed + 1)
    gen = torch.Generator(device=device).manual_seed(args.seed)

    def random_pose(away_from=None):
        for _ in range(1000):
            q = ep_rng.uniform(lo, hi)
            if arm.lowest(q) < args.min_height:
                continue
            if away_from is None or np.abs(q - away_from).mean() >= args.min_start:
                return q
        raise RuntimeError("no valid pose found")

    out = args.out or os.path.join(
        "local", "runs", f"sim_plan_{args.policy}_{args.pred or 'all'}_" + time.strftime("%Y%m%d_%H%M%S")
    )
    os.makedirs(out, exist_ok=True)
    print(f"policy={args.policy}  predictor={tag}  step={k / margs['fps']:.2f} s  steps={args.steps}")
    results, rows = [], []
    for ep in range(args.episodes):
        arm.reset(random_pose())
        z_goal, img_goal = see()
        q_goal, tip_goal = arm.q_deg, arm.tip()
        arm.reset(random_pose(away_from=q_goal))
        _, img_start = see()
        if live:
            live.set_goal(q_goal)
            live.text(f"episode {ep + 1}/{args.episodes}  ({args.policy})", "green ghost = goal pose")
            live.hold(args.hold)

        trace, goal_cmd, plan_ms = [], arm.q_deg, []
        for step in range(args.steps + 1):
            z, _ = see()
            q = arm.q_deg
            trace.append({
                "err": np.abs(q - q_goal).tolist(),
                "tip_cm": float(100 * np.linalg.norm(arm.tip() - tip_goal)),
                "energy": float((z - z_goal).abs().mean()),
            })
            if live:
                live.text(f"episode {ep + 1}/{args.episodes}  step {step}/{args.steps}  ({args.policy})",
                          f"tip {trace[-1]['tip_cm']:.1f} cm from goal")
            if step == args.steps:
                break
            if args.policy == "plan":
                t0 = time.perf_counter()
                act, _ = cem_plan(model, z, q, z_goal, lo_t, hi_t, qmu, qsd, gen, args.horizon, args.samples,
                                  args.iters, args.elites, args.sigma, args.max_step)
                plan_ms.append(1000 * (time.perf_counter() - t0))
            else:
                act = np.clip(act_rng.normal(0, args.sigma, len(joints)), -args.max_step, args.max_step)
            target = np.clip(q + act, lo, hi)
            for _ in range(k):  # 0.5 s, goal ramped at the collection speed
                goal_cmd = goal_cmd + np.clip(target - goal_cmd, -ramp, ramp)
                arm.step(goal_cmd)
                if live:
                    live.frame()
        _, img_final = see()
        e0, e1 = np.array(trace[0]["err"]), np.array(trace[-1]["err"])
        results.append({"goal_deg": q_goal.tolist(), "trace": trace})
        if ep < 6:
            rows.append(np.hstack([img_goal, img_start, img_final]))
        ms = f" | plan {np.mean(plan_ms):.0f} ms" if plan_ms else ""
        print(
            f"ep {ep:2d}: joints {e0.mean():5.2f} -> {e1.mean():5.2f} deg | tip {trace[0]['tip_cm']:4.1f} -> "
            f"{trace[-1]['tip_cm']:4.1f} cm | per joint " + " ".join(f"{a:.1f}->{b:.1f}" for a, b in zip(e0, e1)) + ms
        )
        if live:
            live.hold(args.hold)
            if not live.running():  # window closed: stop and report what ran
                break

    s = np.array([np.mean(r["trace"][0]["err"]) for r in results])
    f = np.array([np.mean(r["trace"][-1]["err"]) for r in results])
    t0 = np.array([r["trace"][0]["tip_cm"] for r in results])
    t1 = np.array([r["trace"][-1]["tip_cm"] for r in results])
    pj0 = np.array([r["trace"][0]["err"] for r in results]).mean(0)
    pj1 = np.array([r["trace"][-1]["err"] for r in results]).mean(0)
    summary = {
        "joint_err_start": float(s.mean()), "joint_err_final": float(f.mean()),
        "tip_cm_start": float(t0.mean()), "tip_cm_final": float(t1.mean()),
        "tip_cm_final_median": float(np.median(t1)), "within_2cm": int((t1 < 2).sum()),
        "closer": int((f < s).sum()), "per_joint_start": pj0.tolist(), "per_joint_final": pj1.tolist(),
    }
    print(f"\n== {len(results)} episodes, policy {args.policy}, predictor {tag}")
    print(f"mean joint error   {s.mean():.2f} -> {f.mean():.2f} deg  ({100 * (1 - f.mean() / s.mean()):.0f}% closed)")
    print(f"gripper tip        {t0.mean():.1f} -> {t1.mean():.1f} cm (median {np.median(t1):.1f}), "
          f"{(t1 < 2).sum()} of {len(results)} end within 2 cm")
    print("per joint (deg)    " + "  ".join(f"{j} {a:.1f}->{b:.1f}" for j, a, b in zip(JOINTS, pj0, pj1)))
    print(f"episodes that got closer  {(f < s).sum()} of {len(results)}")
    with open(os.path.join(out, "result.json"), "w") as fh:
        json.dump({"args": vars(args), "summary": summary, "episodes": results}, fh, indent=2)
    montage = np.vstack(rows)
    cv2.imwrite(os.path.join(out, "montage.jpg"), cv2.resize(montage, (montage.shape[1] // 2, montage.shape[0] // 2)))
    print(f"saved to {out} (montage: goal | start | final)")


if __name__ == "__main__":
    main()
