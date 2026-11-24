"""Closed-loop image-goal flight on the simulated Crazyflie: the drone version of sim_plan.py.

    cd ~/projects/vjepa2
    ~/projects/lerobot/.venv/bin/python -m local.drone_plan local/data/sim_drone/hour_seed1
    ~/projects/lerobot/.venv/bin/python -m local.drone_plan local/data/sim_drone/hour_seed1 --policy random
    ~/projects/lerobot/.venv/bin/python -m local.drone_plan local/data/sim_drone/hour_seed1 --video 3

Each episode:
  1. The drone hovers at a random goal (position and heading, clear of the furniture) and its onboard camera
     records the goal image.
  2. It is moved to a start --min-dist..--max-dist metres away (horizontally, plus a little height) and
     turned up to --max-turn deg, so the goal view overlaps the start view, with a clear straight path.
  3. For --steps steps of 0.5 s: render and encode the onboard frame, plan through the predictor
     (local/drone_predictor.py; the search is described in plan_actions), and command the first action.
     An action moves the flight controller's target forward, left, up (heading frame) and turns it; the
     target slides there at constant speed over the step, and the controller underneath flies the drone.
     The predicted next state carries the drone's velocity from one imagined step to the next.
The planner sees only images and its own onboard state (velocity, height, tilt, target offset); the goal's
position is kept for scoring. Episodes depend only on --seed, so policies face the same ones.

Reports distance and heading error to the goal, before and after. Writes result.json, a montage
(goal | start | final onboard view) and, with --video N, the first N episodes as mp4 (goal | onboard | chase)
to local/runs/drone_plan_<policy>_<ts>/.
"""

import argparse
import json
import os
import time

import cv2
import numpy as np
import torch

from local.arm_plan import encode_frames
from local.drone_predictor import ACTION, STATE, DronePredictor, heading_frame
from local.sim_drone import ROOM, SimDrone
from local.sim_drone_collect import clear_path
from local.vjepa_local import IMAGENET_MEAN, IMAGENET_STD, load_model, pick_device


def plan_actions(model, z, s, z_goal, gen, horizon=2, samples=256, iters=4, init=None, max_std=2.5):
    """Search over action sequences [H, 4] in standardized units (1 = a typical step in the training data).

    z, z_goal [1, 16, D]; s [1, S] standardized state; init [H, 4] a starting guess (the previous step's plan,
    shifted). Returns (best sequence [H, 4], its energy).

    Energy: L1 to the goal averaged over every step of the sequence. Scored on the last step alone, "go
    forward, then come back" ties with "stay", so the first action, the one that gets executed, is left
    almost free; at the goal that made the drone fly off.

    Search: keep the best sequence found so far and sample around it with a spread that halves each round
    (1, 0.5, 0.25, 0.125). "Stay" and the starting guess are always candidates. Plain CEM, which recentres
    on the average of the 24 best samples, failed here: the good region is a narrow valley in 8 dimensions,
    the best samples scatter along its sides, and their average lands on a slope (at the goal, rounds went
    0.11 -> 0.51 -> 0.66 in energy while "stay" scored 0.11).
    """
    device = z.device
    best = torch.zeros(horizon, len(ACTION), device=device)
    with torch.inference_mode():
        def energy_of(a):
            zc, sc = z.expand(len(a), -1, -1), s.expand(len(a), -1)
            e = torch.zeros(len(a), device=device)
            for h in range(horizon):
                zc, sc = model(zc, sc, a[:, h])
                e += (zc - z_goal).abs().mean(dim=(1, 2)) / horizon
            return e

        seeds = torch.stack([best] + ([init] if init is not None else []))
        e = energy_of(seeds)
        best, best_e = seeds[int(e.argmin())], float(e.min())
        spread = 1.0
        for _ in range(iters):
            a = (best + spread * torch.randn((samples, horizon, len(ACTION)), device=device, generator=gen))
            a = a.clamp(-max_std, max_std)
            e = energy_of(a)
            if float(e.min()) < best_e:
                best, best_e = a[int(e.argmin())], float(e.min())
            spread /= 2
    return best, best_e


def wrap180(d):
    return (d + 180.0) % 360.0 - 180.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="sim drone run folder holding pred_<model>.pt")
    ap.add_argument("--model", default="vjepa2_1_vit_large_384")
    ap.add_argument("--policy", default="plan", choices=["plan", "random"])
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--steps", type=int, default=20, help="control steps of 0.5 s per episode")
    ap.add_argument("--horizon", type=int, default=2)
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--min-dist", type=float, default=0.3, help="start at least this far from the goal, m")
    ap.add_argument("--max-dist", type=float, default=0.8)
    ap.add_argument("--max-turn", type=float, default=30.0, help="largest heading difference start-goal, deg")
    ap.add_argument("--clearance", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=100, help="episode seed (collection used 0 and 1)")
    ap.add_argument("--video", type=int, default=0, help="record the first N episodes as mp4")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = pick_device(None)
    meta = json.load(open(os.path.join(args.run, "meta.json")))
    ckpt = torch.load(os.path.join(args.run, f"pred_{args.model}.pt"), map_location=device, weights_only=False)
    cfg = ckpt["args"]
    k = cfg["k"]
    zmu, zsd = ckpt["zmu"].to(device), ckpt["zsd"].to(device)
    smu, ssd, asd = (torch.as_tensor(ckpt[x], dtype=torch.float32, device=device) for x in ("smu", "ssd", "asd"))
    model = DronePredictor(zmu.shape[-1], len(STATE), len(ACTION), cfg["width"], cfg["depth"]).to(device).eval()
    model.load_state_dict(ckpt["state_dict"])
    encoder = load_model(args.model, device=device)[0]
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    ma = meta["args"]
    drone = SimDrone(width=ma["width"], height=ma["height"], fps=ma["fps"])
    assert drone.substeps == meta["substeps_per_frame"], "frame timing differs from the training run"
    lim = ROOM - ma["wall_margin"]
    camera = meta["camera"]["name"]  # fly with the camera the predictor was trained on
    z_lo, z_hi = ma["z_min"], ma["z_max"]

    def see():
        frame = drone.render(camera)
        return encode_frames([frame], encoder, mean, std, zmu, zsd, device), frame

    def state_now(ref):
        yaw = np.radians(drone.rpy_deg[2])
        p = drone.pos
        s = np.concatenate([heading_frame(drone.vel[None], np.array([yaw]))[0], [p[2]], drone.rpy_deg[:2],
                            heading_frame((ref - p)[None], np.array([yaw]))[0]])
        return (torch.as_tensor(s, dtype=torch.float32, device=device)[None] - smu) / ssd

    ep_rng = np.random.default_rng(args.seed)
    act_rng = np.random.default_rng(args.seed + 1)
    gen = torch.Generator(device=device).manual_seed(args.seed)

    def free_spot():
        for _ in range(1000):
            p = np.array([*ep_rng.uniform(-lim, lim, 2), ep_rng.uniform(z_lo + 0.2, z_hi - 0.2)])
            if clear_path(p, p, args.clearance):
                return p, np.radians(ep_rng.uniform(-180, 180))
        raise RuntimeError("no free spot")

    def start_near(goal, gyaw):
        for _ in range(1000):
            d, ang = ep_rng.uniform(args.min_dist, args.max_dist), ep_rng.uniform(-np.pi, np.pi)
            p = goal + np.array([d * np.cos(ang), d * np.sin(ang), ep_rng.uniform(-0.2, 0.2)])
            if (np.abs(p[:2]) <= lim).all() and z_lo <= p[2] <= z_hi and clear_path(p, goal, args.clearance):
                return p, gyaw + np.radians(ep_rng.uniform(-args.max_turn, args.max_turn))
        return None, None

    out = args.out or os.path.join("local", "runs", f"drone_plan_{args.policy}_" + time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out, exist_ok=True)
    print(f"policy={args.policy}  predictor={args.model}  camera={camera}  step={k / drone.fps:.2f} s  "
          f"steps={args.steps}  "
          f"start {args.min_dist}-{args.max_dist} m and up to {args.max_turn:.0f} deg from the goal")
    results, rows, plan_ms = [], [], []
    for ep in range(args.episodes):
        while True:
            goal, gyaw = free_spot()
            start, syaw = start_near(goal, gyaw)
            if start is not None:
                break
        drone.hover_at(goal, gyaw)
        z_goal, img_goal = see()
        drone.hover_at(start, syaw)
        _, img_start = see()
        ref, yaw_ref = start.copy(), syaw
        plan = None
        writer = None
        if ep < args.video:
            h, w = img_goal.shape[:2]
            writer = cv2.VideoWriter(os.path.join(out, f"episode{ep}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                     drone.fps, (3 * w, h))
            label = img_goal.copy()
            cv2.putText(label, "goal", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        trace = []
        for step in range(args.steps + 1):
            z, frame = see()
            p, yaw_deg = drone.pos, drone.rpy_deg[2]
            trace.append({"dist_m": float(np.linalg.norm(p - goal)),
                          "heading_err_deg": float(abs(wrap180(yaw_deg - np.degrees(gyaw)))),
                          "energy": float((z - z_goal).abs().mean())})
            if step == args.steps:
                break
            if args.policy == "plan":
                t0 = time.perf_counter()
                init = None if plan is None else torch.cat([plan[1:], torch.zeros_like(plan[:1])])
                plan, _ = plan_actions(model, z, state_now(ref), z_goal, gen, args.horizon, args.samples, args.iters,
                                       init)
                plan_ms.append(1000 * (time.perf_counter() - t0))
                act = (plan[0] * asd).cpu().numpy()
            else:
                act = np.clip(act_rng.normal(0, 1, len(ACTION)), -2.5, 2.5) * asd.cpu().numpy()
            # the action is a move of the controller target in the heading frame at t
            yaw = np.radians(yaw_deg)
            c, s_ = np.cos(yaw), np.sin(yaw)
            move = np.array([c * act[0] - s_ * act[1], s_ * act[0] + c * act[1], act[2]])
            new_ref = ref + move
            new_ref[:2] = np.clip(new_ref[:2], -lim, lim)
            new_ref[2] = np.clip(new_ref[2], z_lo, z_hi)
            if not clear_path(ref, new_ref, 0.15):  # never steer the target into furniture
                new_ref = ref.copy()
            new_yaw = yaw_ref + np.radians(act[3])
            v_ref = (new_ref - ref) * drone.fps / k
            for i in range(k):
                frac = (i + 1) / k
                r, yr = ref + frac * (new_ref - ref), yaw_ref + frac * (new_yaw - yaw_ref)
                for _ in range(drone.substeps):
                    drone.tick(r, v_ref, yr)
                if writer is not None:
                    writer.write(np.hstack([label, drone.render(camera), drone.render("track")]))
            ref, yaw_ref = new_ref, new_yaw
            if drone.touching():
                print(f"  ep {ep}: touched something at step {step}")
        if writer is not None:
            writer.release()
        _, img_final = see()
        results.append({"goal": goal.tolist(), "goal_yaw_deg": float(np.degrees(gyaw)), "start": start.tolist(),
                        "trace": trace})
        if ep < 6:
            rows.append(np.hstack([img_goal, img_start, img_final]))
        t0, t1 = trace[0], trace[-1]
        ms = f" | plan {np.mean(plan_ms[-args.steps:]):.0f} ms" if plan_ms else ""
        print(f"ep {ep:2d}: distance {100 * t0['dist_m']:5.1f} -> {100 * t1['dist_m']:5.1f} cm | heading "
              f"{t0['heading_err_deg']:4.1f} -> {t1['heading_err_deg']:4.1f} deg | energy {t0['energy']:.3f} -> "
              f"{t1['energy']:.3f}{ms}")

    d0 = np.array([r["trace"][0]["dist_m"] for r in results]) * 100
    d1 = np.array([r["trace"][-1]["dist_m"] for r in results]) * 100
    h0 = np.array([r["trace"][0]["heading_err_deg"] for r in results])
    h1 = np.array([r["trace"][-1]["heading_err_deg"] for r in results])
    dmin = np.array([min(t["dist_m"] for t in r["trace"]) for r in results]) * 100
    ok = (d1 < 20) & (h1 < 15)
    summary = {"dist_cm_start": float(d0.mean()), "dist_cm_final": float(d1.mean()),
               "dist_cm_final_median": float(np.median(d1)), "dist_cm_closest": float(dmin.mean()),
               "heading_deg_start": float(h0.mean()), "heading_deg_final": float(h1.mean()),
               "within_20cm_15deg": int(ok.sum()), "closer": int((d1 < d0).sum()),
               "plan_ms": float(np.mean(plan_ms)) if plan_ms else None}
    print(f"\n== {len(results)} episodes, policy {args.policy}")
    print(f"distance to goal   {d0.mean():.1f} -> {d1.mean():.1f} cm (median {np.median(d1):.1f}; closest on the way "
          f"{dmin.mean():.1f})")
    print(f"heading error      {h0.mean():.1f} -> {h1.mean():.1f} deg")
    print(f"ended within 20 cm and 15 deg: {ok.sum()} of {len(results)} | got closer: {(d1 < d0).sum()} of "
          f"{len(results)}")
    with open(os.path.join(out, "result.json"), "w") as fh:
        json.dump({"args": vars(args), "summary": summary, "episodes": results}, fh, indent=2)
    montage = np.vstack(rows)
    cv2.imwrite(os.path.join(out, "montage.jpg"), cv2.resize(montage, (montage.shape[1] // 2, montage.shape[0] // 2)))
    print(f"saved to {out} (montage: goal | start | final)")


if __name__ == "__main__":
    main()
