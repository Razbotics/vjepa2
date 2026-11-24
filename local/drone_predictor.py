"""Train an action-conditioned predictor on frozen V-JEPA features of a sim_drone_collect.py run.

    .venv/bin/python -m local.drone_predictor local/data/sim_drone/hour_seed1 --model vjepa2_1_vit_large_384

The drone's version of arm_predictor.py (same network, same split, same evaluations), with drone-shaped
state and action. Everything is expressed in the drone's heading frame (x forward, y left, z up, rotated by
the measured heading at t), because the camera sees the world from there: "20 cm forward" should look the
same whichever way the drone faces.

  features z_t   the 4x4 grid of ViT tokens of onboard frame t (16 tokens x D), standardized
  state s_t      what the drone knows about itself from onboard sensors: velocity (heading frame, m/s),
                 height (m), roll and pitch (deg), and where its controller target is relative to it
                 (heading frame, m). A drone has momentum, so unlike the arm its future depends on how fast
                 it is already moving.
  action a_t     how the controller's target moves over the step: forward, left and up (m, heading frame
                 at t) and heading change (deg). This is what a planner would command; the flight
                 controller underneath turns it into rotor thrusts.
  target         z_{t+k} (L1, residual on z_t) and s_{t+k} (L1, residual on s_t). Predicting the next state
                 lets a rollout or a planner chain steps without knowing the real future state; for the arm
                 the next state was simply state + action.

Split: the first 80% of the run trains, the last 20% tests, after a 1 s gap. Evaluations on the test part:
  1. feature error      L1 against "copy" (no change) and the same network trained without actions
  2. decoded pose       a linear probe (fit on training frames) reads position, height, tilt and heading out
                        of the predicted features; error against the real future pose. "real future" decodes
                        the real z_{t+k} and is the probe's own floor. Absolute position reads out only to
                        ~90 cm, too coarse to judge 10-20 cm steps: for motion, 3. is the measure (a linear
                        probe on the pair (z_t, z_{t+k}) could not read the step's motion even from real
                        frames, so it was dropped)
  3. action ranking     the true action among --candidates others from the test set, scored by how close its
                        predicted future lands to the real one; how far the winner is from the true action
  4. rollout            --rollout steps feeding predicted features and predicted state back in, with the real
                        action sequence
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from local.arm_predictor import Predictor, make_probe
from local.vjepa_local import pick_device

STATE = ["v_fwd", "v_left", "v_up", "height", "roll", "pitch", "tgt_fwd", "tgt_left", "tgt_up"]
ACTION = ["d_fwd", "d_left", "d_up", "d_yaw"]


class DronePredictor(Predictor):
    """Predictor with separate state and action sizes, and a head that predicts the next state."""

    def __init__(self, dim, n_state, n_act, width=384, depth=4, use_action=True):
        super().__init__(dim, n_act, width, depth, use_action=use_action)
        self.state = nn.Linear(n_state, width)
        self.state_out = nn.Linear(width, n_state)
        nn.init.zeros_(self.state_out.weight)
        nn.init.zeros_(self.state_out.bias)

    def forward(self, z, state, action):
        """z [B, 16, D], state [B, S], action [B, A] -> (next z [B, 16, D], next state [B, S])."""
        if not self.use_action:
            action = torch.zeros_like(action)
        x = torch.cat([self.inp(z), self.state(state)[:, None], self.act(action)[:, None]], dim=1) + self.pos
        x = self.norm(self.blocks(x))
        return z + self.out(x[:, : z.shape[1]]), state + self.state_out(x[:, z.shape[1]])


def heading_frame(v, yaw):
    """Rotate world-frame xy of v [N, 3] into the frame of heading yaw [N] (rad): x forward, y left."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.column_stack([c * v[:, 0] + s * v[:, 1], -s * v[:, 0] + c * v[:, 1], v[:, 2]])


def train(model, pairs, t_train, steps, batch, lr, noise, device, seed):
    g = torch.Generator(device=device).manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    model.train()
    t0, run_z, run_s = time.perf_counter(), 0.0, 0.0
    for step in range(1, steps + 1):
        z, s, a, zn, sn = pairs(t_train[torch.randint(0, len(t_train), (batch,), device=device, generator=g)])
        zp, sp = model(z + noise * torch.randn(z.shape, device=device, generator=g), s, a)
        lz, ls = F.l1_loss(zp, zn), F.l1_loss(sp, sn)
        loss = lz + 0.1 * ls
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        run_z, run_s = run_z + lz.item(), run_s + ls.item()
        if step % (steps // 5) == 0:
            m = steps // 5
            print(f"    step {step:5d}  train L1 features {run_z / m:.4f}  state {run_s / m:.4f}  "
                  f"({time.perf_counter() - t0:.0f} s)")
            run_z, run_s = 0.0, 0.0
    model.eval()
    return model


def pose_errors(p, y):
    """Errors between pose vectors [x, y, z, roll, pitch, sin yaw, cos yaw]: position cm, height cm,
    tilt deg (mean of |roll|, |pitch| errors), heading deg."""
    pos = 100 * (p[:, :3] - y[:, :3]).norm(dim=1)
    hgt = 100 * (p[:, 2] - y[:, 2]).abs()
    tilt = (p[:, 3:5] - y[:, 3:5]).abs().mean(1)
    d = torch.atan2(p[:, 5], p[:, 6]) - torch.atan2(y[:, 5], y[:, 6])
    head = torch.rad2deg(torch.atan2(torch.sin(d), torch.cos(d)).abs())
    return pos, hgt, tilt, head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--model", default="vjepa2_1_vit_large_384", help="which feat_<model>.npz to use")
    ap.add_argument("--k", type=int, default=15, help="frames per step (15 at 30 Hz = 0.5 s)")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--width", type=int, default=384)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--noise", type=float, default=0.05, help="input noise in standardized feature units")
    ap.add_argument("--candidates", type=int, default=64)
    ap.add_argument("--rollout", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-seconds", type=float, default=None, help="train on only the first N seconds")
    ap.add_argument("--max-test", type=int, default=4000, help="evaluate on at most this many test starts")
    ap.add_argument("--probe-max", type=int, default=8000, help="fit the probe on at most this many frames")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    log = np.load(os.path.join(args.run, "log.npz"))
    meta = json.load(open(os.path.join(args.run, "meta.json")))
    assert meta.get("robot") == "crazyflie_2", "this is for sim_drone_collect.py runs"
    feat = np.load(os.path.join(args.run, f"feat_{args.model}.npz"))
    assert np.array_equal(feat["idx"], np.arange(len(log["t_frame"]))), "features must cover every frame"
    assert log["segment"].max() == 0, "runs with resets need pairs kept inside a segment (not handled yet)"
    n, k = len(feat["idx"]), args.k
    fps = (n - 1) / float(log["t_frame"][-1] - log["t_frame"][0])
    gap = int(round(fps))

    # state, action and pose, all per frame
    rpy = log["rpy_deg"].astype(np.float64)
    yaw = np.radians(rpy[:, 2])
    state = np.column_stack([heading_frame(log["vel"], yaw), log["pos"][:, 2], rpy[:, :2],
                             heading_frame(log["ref_pos"] - log["pos"], yaw)])
    pose = np.column_stack([log["pos"], rpy[:, :2], np.sin(yaw), np.cos(yaw)])
    ref, ref_yaw = log["ref_pos"].astype(np.float64), np.degrees(log["ref_yaw"].astype(np.float64))

    def actions_np(t0, t1):
        return np.column_stack([heading_frame(ref[t1] - ref[t0], yaw[t0]), ref_yaw[t1] - ref_yaw[t0]])

    split = int(n * 0.8)
    tr_end = split - gap - k
    if args.train_seconds is not None:
        tr_end = min(tr_end, int(round(args.train_seconds * fps)))
    tr_t = torch.arange(0, tr_end, device=device)
    te_t = torch.arange(split, n - k * args.rollout, device=device)
    if len(te_t) > args.max_test:
        te_t = te_t[torch.linspace(0, len(te_t) - 1, args.max_test, device=device).long()]

    # features: standardized, float16, on the GPU, gathered by index
    grid = feat["grid"]
    dim = grid.shape[1] // 16
    sub = torch.from_numpy(grid[: split - gap : max(1, (split - gap) // 5000)].astype(np.float32))
    sub = sub.reshape(len(sub), dim, 16).transpose(1, 2)
    zmu = sub.mean(dim=(0, 1), keepdim=True).to(device)
    zsd = sub.std(dim=(0, 1), keepdim=True).clamp_min(1e-6).to(device)
    z_all = torch.empty(n, 16, dim, dtype=torch.float16, device=device)
    for c in range(0, n, 4096):
        zc = torch.from_numpy(grid[c : c + 4096].astype(np.float32)).to(device).reshape(-1, dim, 16).transpose(1, 2)
        z_all[c : c + 4096] = ((zc - zmu) / zsd).half()
    del grid, sub

    tr_np = np.arange(split - gap)
    smu, ssd = state[tr_np].mean(0), state[tr_np].std(0)
    a_tr = actions_np(tr_np[:-k], tr_np[:-k] + k)
    asd = a_tr.std(0)  # scale only: a zero action stays zero
    tt = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)  # noqa: E731
    s_all, pose_all = tt((state - smu) / ssd), tt(pose)
    a_next = np.zeros((n, len(ACTION)))
    a_next[: n - k] = actions_np(np.arange(n - k), np.arange(k, n)) / asd
    a_all = tt(a_next)  # a_all[t]: the action from t to t + k
    smu_t, ssd_t, asd_t = tt(smu), tt(ssd), tt(asd)

    def pairs(t):
        return z_all[t].float(), s_all[t], a_all[t], z_all[t + k].float(), s_all[t + k]

    print(f"run {args.run} | {args.model} dim {dim} | {n} frames at {fps:.1f} Hz | step {k} frames = {k / fps:.2f} s")
    print(f"train pairs {len(tr_t)} ({len(tr_t) / fps:.0f} s)  test starts {len(te_t)} in the last "
          f"{(n - split) / fps:.0f} s")
    at = (a_all[te_t] * asd_t).abs().mean(0).tolist()
    print(f"mean |action| per step on test: forward {100 * at[0]:.1f} cm, left {100 * at[1]:.1f} cm, "
          f"up {100 * at[2]:.1f} cm, heading {at[3]:.1f} deg")

    models = {}
    for name, use_action in (("with actions", True), ("no actions", False)):
        print(f"\n  training predictor, {name}")
        m = DronePredictor(dim, len(STATE), len(ACTION), args.width, args.depth, use_action=use_action).to(device)
        if name == "with actions":
            print(f"    {sum(p.numel() for p in m.parameters()) / 1e6:.1f}M parameters")
        models[name] = train(m, pairs, tr_t, args.steps, args.batch, args.lr, args.noise, device, args.seed)
    model = models["with actions"]

    pt = torch.arange(0, split - gap, max(1, (split - gap) // args.probe_max), device=device)
    probe = make_probe(z_all[pt].float().flatten(1), pose_all[pt])
    res = {"model": args.model, "k": k, "state": STATE, "action": ACTION, "train_pairs": len(tr_t),
           "test_starts": len(te_t)}
    with torch.inference_mode():
        z, s, a, zn, sn = pairs(te_t)
        y_next = pose_all[te_t + k]
        out = {"no actions": models["no actions"](z, s, a), "with actions": model(z, s, a)}
        preds = {"copy (no change)": z, **{name: o[0] for name, o in out.items()}}

        print("\n== 1. feature error on held-out time (L1, standardized units; lower is better)")
        for name, p in preds.items():
            e = F.l1_loss(p, zn).item()
            res[f"feat_l1/{name}"] = e
            print(f"  {name:<20} {e:.4f}")
        print("   next-state error (m/s, m, deg), copy -> with actions:")
        for i, sname in enumerate(STATE):
            ec = ((s[:, i] - sn[:, i]) * ssd_t[i]).abs().mean().item()
            ea = ((out["with actions"][1][:, i] - sn[:, i]) * ssd_t[i]).abs().mean().item()
            res[f"state_mae/{sname}"] = {"copy": ec, "with actions": ea}
            print(f"     {sname:<9} {ec:.3f} -> {ea:.3f}")

        print("\n== 2. pose decoded from predicted features, error vs the real future pose (mean)")
        print(f"  {'':<20}{'position cm':>13}{'height cm':>11}{'tilt deg':>10}{'heading deg':>13}")
        rows = dict(preds)
        rows["real future (floor)"] = zn
        for name, p in rows.items():
            e = [v.mean().item() for v in pose_errors(probe(p.flatten(1)), y_next)]
            res[f"decoded/{name}"] = dict(zip(["pos_cm", "height_cm", "tilt_deg", "heading_deg"], e))
            print(f"  {name:<20}{e[0]:>13.1f}{e[1]:>11.1f}{e[2]:>10.2f}{e[3]:>13.1f}")

        print(f"\n== 3. action ranking: true action among {args.candidates} candidates, scored by predicted future")
        gen = torch.Generator(device=device).manual_seed(args.seed)
        pick = torch.randperm(len(te_t), device=device, generator=gen)[:400]
        hits, rank_sum, chosen, rand = 0, 0, [], []
        for i in pick.tolist():
            others = torch.randint(0, len(te_t), (args.candidates - 1,), device=device, generator=gen)
            cand = torch.cat([a[i : i + 1], a[others]])  # index 0 is the true action
            zp, _ = model(z[i : i + 1].expand(len(cand), -1, -1), s[i : i + 1].expand(len(cand), -1), cand)
            score = (zp - zn[i : i + 1]).abs().mean(dim=(1, 2))
            best = int(score.argmin())
            hits += best == 0
            rank_sum += int((score < score[0]).sum()) + 1
            d = (cand - cand[0:1]) * asd_t
            dist, dyaw = 100 * d[:, :3].norm(dim=1), d[:, 3].abs()
            chosen.append((dist[best].item(), dyaw[best].item()))
            rand.append((dist[1:].mean().item(), dyaw[1:].mean().item()))
        (cd, cy), (rd, ry) = np.mean(chosen, 0), np.mean(rand, 0)
        res.update({"rank/top1": hits / len(pick), "rank/mean_rank": rank_sum / len(pick),
                    "rank/chosen_err": [cd, cy], "rank/random_err": [rd, ry]})
        print(f"  true action ranked first   {100 * hits / len(pick):.1f}%  (chance {100 / args.candidates:.1f}%)")
        print(f"  mean rank of true action   {rank_sum / len(pick):.1f} of {args.candidates}")
        print(f"  chosen action is off by    {cd:.1f} cm and {cy:.1f} deg  (a random pick: {rd:.1f} cm, {ry:.1f} deg)")

        print(f"\n== 4. rollout, {args.rollout} steps = {args.rollout * k / fps:.1f} s, predicted features and state "
              f"fed back in")
        zc, sc = z, s
        for step in range(1, args.rollout + 1):
            t0, t1 = te_t + k * (step - 1), te_t + k * step
            zc, sc = model(zc, sc, a_all[t0])
            real, y = z_all[t1].float(), pose_all[t1]
            l1, l1c = F.l1_loss(zc, real).item(), F.l1_loss(z, real).item()
            em = [v.mean().item() for v in pose_errors(probe(zc.flatten(1)), y)]
            ec = [v.mean().item() for v in pose_errors(probe(z.flatten(1)), y)]
            res[f"rollout/{step}"] = {"l1": l1, "l1_copy": l1c, "decoded": em, "decoded_copy": ec}
            print(f"  step {step}: feature L1 {l1:.4f} (copy {l1c:.4f})   position {em[0]:.0f} cm (copy {ec[0]:.0f})"
                  f"  heading {em[3]:.1f} deg (copy {ec[3]:.1f})")

    tag = args.model + (f"_train{args.train_seconds:g}s" if args.train_seconds is not None else "")
    torch.save(
        {"state_dict": model.state_dict(), "args": vars(args), "state": STATE, "action": ACTION,
         "zmu": zmu.cpu(), "zsd": zsd.cpu(), "smu": smu, "ssd": ssd, "asd": asd},
        os.path.join(args.run, f"pred_{tag}.pt"),
    )
    with open(os.path.join(args.run, f"pred_{tag}.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nsaved pred_{tag}.pt and .json in {args.run}")


if __name__ == "__main__":
    main()
