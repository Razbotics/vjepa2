"""Train a small action-conditioned predictor on frozen V-JEPA features of an arm_collect.py run.

    python -m local.arm_predictor local/data/arm/20261124_011609
    python -m local.arm_predictor local/data/arm/<run> --model vjepa2_1_vit_large_384 --steps 4000

Needs `feat_<model>.npz` from arm_encode.py. Setup, following V-JEPA 2-AC but in joint space:
  state    z_t: the 4x4 grid of pooled patch tokens of frame t (16 tokens x D), standardized
  proprio  measured joint angles at t (the joints that moved)
  action   measured joint-angle change between t and t+k frames (default k = 15, 0.5 s; at 0.2 s the
           arm moves only ~0.5 deg per step, which is lost in sensor noise)
  target   z_{t+k}, L1 loss, predicted as a residual on z_t

Split: the first 80% of the run trains, the last 20% tests, with a 1 s gap. Everything reported is on the
test part. Evaluations:
  1. feature error     L1 between predicted and real future features, against "copy" (predict no
                       change) and the same network trained without actions.
  2. decoded joints    a linear probe (fit on training frames) reads joint angles out of the predicted
                       features; error against the real future angles. "copy" decodes z_t, "real" decodes
                       the real z_{t+k} and is the probe's own floor.
  3. action ranking    the planning question. Given z_t and the real future, score the true action and
                       63 other actions from the test set by how close their predicted future lands.
                       Reports how often the true action wins and how far the winner is from it.
  4. rollout           feed predictions back in for 5 steps with the real action sequence.

Long runs (the simulator gives hours): features stay on the GPU in float16 and training batches are
gathered by index; the evaluations use at most --max-test evenly spaced test starts and the probe is fit
on at most --probe-max training frames. --train-seconds N trains the predictor on only the first N seconds
of the training part; the test part, the feature standardization and the probe stay the same, so runs with
different N are directly comparable (a data-scaling curve). Outputs are tagged _train<N>s.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from local.vjepa_local import pick_device


class Predictor(nn.Module):
    def __init__(self, dim, n_act, width=384, depth=4, heads=6, n_tokens=16, dropout=0.1, use_action=True):
        super().__init__()
        self.use_action = use_action
        self.inp = nn.Linear(dim, width)
        self.act = nn.Linear(n_act, width)
        self.state = nn.Linear(n_act, width)
        self.pos = nn.Parameter(torch.zeros(1, n_tokens + 2, width))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(width, heads, width * 4, dropout, "gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, z, state, action):
        """z [B, 16, D], state [B, A], action [B, A] -> predicted next z [B, 16, D]."""
        if not self.use_action:
            action = torch.zeros_like(action)
        x = torch.cat([self.inp(z), self.state(state)[:, None], self.act(action)[:, None]], dim=1) + self.pos
        x = self.norm(self.blocks(x))
        return z + self.out(x[:, : z.shape[1]])


def make_probe(x, y, lambdas=(0.1, 1.0, 10.0, 100.0, 1000.0)):
    """Ridge probe features -> joint angles, dual form. Returns a function on new features.

    The ridge strength is chosen on the last 20% of the given (training) frames, then refit on all of them.
    """

    def fit(xf, yf, lam):
        mu, sd = xf.mean(0, keepdim=True), xf.std(0, keepdim=True).clamp_min(1e-6)
        xs = ((xf - mu) / sd).double()
        ymu = yf.mean(0, keepdim=True).double()
        k = xs @ xs.T / xs.shape[1]
        alpha = torch.linalg.solve(k + lam * torch.eye(len(k), device=k.device, dtype=k.dtype), yf.double() - ymu)
        w = (xs.T @ alpha / xs.shape[1]).float()  # primal weights, so prediction is one matmul
        return lambda xn: ((xn - mu) / sd) @ w + ymu.float()

    cut = int(len(x) * 0.8)
    errs = [(fit(x[:cut], y[:cut], lam)(x[cut:]) - y[cut:]).abs().mean().item() for lam in lambdas]
    return fit(x, y, lambdas[int(np.argmin(errs))])


def train(model, pairs, t_train, steps, batch, lr, noise, device, seed):
    """pairs(t) -> (z, state, action, next z) for start frames t; t_train holds the allowed starts."""
    g = torch.Generator(device=device).manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    model.train()
    t0, running = time.perf_counter(), 0.0
    for step in range(1, steps + 1):
        z, s, a, zn = pairs(t_train[torch.randint(0, len(t_train), (batch,), device=device, generator=g)])
        zin = z + noise * torch.randn(z.shape, device=device, generator=g)
        loss = F.l1_loss(model(zin, s, a), zn)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        running += loss.item()
        if step % (steps // 5) == 0:
            print(f"    step {step:5d}  train L1 {running / (steps // 5):.4f}  ({time.perf_counter() - t0:.0f} s)")
            running = 0.0
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--model", default="vjepa2_1_vit_base_384", help="which feat_<model>.npz to use")
    ap.add_argument("--k", type=int, default=15, help="frames per step (15 at 30 Hz = 0.5 s)")
    ap.add_argument("--steps", type=int, default=3000)
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
    feat = np.load(os.path.join(args.run, f"feat_{args.model}.npz"))
    assert np.array_equal(feat["idx"], np.arange(len(log["t_frame"]))), "features must cover every frame"
    joints = [j for j in meta["joints"] if j in meta["moving"]]
    cols = [meta["joints"].index(j) for j in joints]
    n = len(feat["idx"])
    dim = feat["grid"].shape[1] // 16
    # arm_encode stored the grid as [D, 4, 4] flattened; make it [N, 16 tokens, D]
    grid = feat["grid"]  # float16 [N, D*16]
    q_all = torch.from_numpy(log["pos_deg"][:, cols].astype(np.float32)).to(device)
    fps = (n - 1) / float(log["t_frame"][-1] - log["t_frame"][0])
    k, gap = args.k, int(round(fps))

    split = int(n * 0.8)
    tr_end = split - gap - k
    if args.train_seconds is not None:
        tr_end = min(tr_end, int(round(args.train_seconds * fps)))
    tr_t = torch.arange(0, tr_end, device=device)
    te_t = torch.arange(split, n - k * args.rollout, device=device)
    if len(te_t) > args.max_test:
        te_t = te_t[torch.linspace(0, len(te_t) - 1, args.max_test, device=device).long()]
    # standardization from the whole training part (a subsample is plenty for a mean and std)
    sub = torch.from_numpy(grid[: split - gap : max(1, (split - gap) // 5000)].astype(np.float32))
    sub = sub.reshape(len(sub), dim, 16).transpose(1, 2)
    zmu = sub.mean(dim=(0, 1), keepdim=True).to(device)
    zsd = sub.std(dim=(0, 1), keepdim=True).clamp_min(1e-6).to(device)
    z_all = torch.empty(n, 16, dim, dtype=torch.float16, device=device)
    for c in range(0, n, 4096):
        zc = torch.from_numpy(grid[c : c + 4096].astype(np.float32)).to(device).reshape(-1, dim, 16).transpose(1, 2)
        z_all[c : c + 4096] = ((zc - zmu) / zsd).half()
    del grid, sub
    qmu, qsd = q_all[: split - gap].mean(0, keepdim=True), q_all[: split - gap].std(0, keepdim=True)
    s_all = (q_all - qmu) / qsd

    def pairs(t, step=k):
        return z_all[t].float(), s_all[t], (q_all[t + step] - q_all[t]) / qsd, z_all[t + step].float()

    print(f"run {args.run} | {args.model} dim {dim} | {n} frames at {fps:.1f} Hz | step {k} frames = {k / fps:.2f} s")
    test_s = (n - split) / fps
    print(f"train pairs {len(tr_t)} ({len(tr_t) / fps:.0f} s)  test starts {len(te_t)} in the last {test_s:.0f} s")
    da = (q_all[te_t + k] - q_all[te_t]).abs()
    print("joints " + ", ".join(joints))
    print("mean |joint change| per step on test (deg): " + "  ".join(f"{v:.2f}" for v in da.mean(0).tolist()))

    models = {}
    for name, use_action in (("with actions", True), ("no actions", False)):
        print(f"\n  training predictor, {name}")
        m = Predictor(dim, len(joints), args.width, args.depth, use_action=use_action).to(device)
        if name == "with actions":
            print(f"    {sum(p.numel() for p in m.parameters()) / 1e6:.1f}M parameters")
        models[name] = train(m, pairs, tr_t, args.steps, args.batch, args.lr, args.noise, device, args.seed)
    model = models["with actions"]

    pt = torch.arange(0, split - gap, max(1, (split - gap) // args.probe_max), device=device)
    probe = make_probe(z_all[pt].float().flatten(1), q_all[pt])
    res = {"joints": joints, "model": args.model, "k": k, "train_pairs": len(tr_t), "test_starts": len(te_t)}
    with torch.inference_mode():
        z, s, a, zn = pairs(te_t)
        q_next = q_all[te_t + k]
        preds = {"copy (no change)": z, "no actions": models["no actions"](z, s, a), "with actions": model(z, s, a)}

        print("\n== 1. feature error on held-out time (L1, standardized units; lower is better)")
        for name, p in preds.items():
            e = F.l1_loss(p, zn).item()
            res[f"feat_l1/{name}"] = e
            print(f"  {name:<18} {e:.4f}")

        print("\n== 2. joint angles decoded from predicted features, error vs real future angles (deg)")
        print(f"  {'':<18}" + "".join(f"{j:>12}" for j in joints) + f"{'mean':>10}")
        rows = dict(preds)
        rows["real future (floor)"] = zn
        for name, p in rows.items():
            e = (probe(p.flatten(1)) - q_next).abs().mean(0)
            res[f"decoded_mae/{name}"] = e.tolist()
            print(f"  {name:<18}" + "".join(f"{v:>12.2f}" for v in e.tolist()) + f"{e.mean().item():>10.2f}")

        print(f"\n== 3. action ranking: true action among {args.candidates} candidates, scored by predicted future")
        gen = torch.Generator(device=device).manual_seed(args.seed)
        pick = torch.randperm(len(te_t), device=device, generator=gen)[:400]
        hits, chosen_err, random_err, rank_sum = 0, [], [], 0
        for i in pick.tolist():
            others = torch.randint(0, len(te_t), (args.candidates - 1,), device=device, generator=gen)
            cand = torch.cat([a[i : i + 1], a[others]])  # index 0 is the true action
            zp = model(z[i : i + 1].expand(len(cand), -1, -1), s[i : i + 1].expand(len(cand), -1), cand)
            score = (zp - zn[i : i + 1]).abs().mean(dim=(1, 2))
            best = int(score.argmin())
            hits += best == 0
            rank_sum += int((score < score[0]).sum()) + 1
            err = ((cand - cand[0:1]) * qsd).abs().mean(1)  # deg, averaged over joints
            chosen_err.append(err[best].item())
            random_err.append(err[1:].mean().item())
        res["rank/top1"] = hits / len(pick)
        res["rank/mean_rank"] = rank_sum / len(pick)
        res["rank/chosen_action_err_deg"] = float(np.mean(chosen_err))
        res["rank/random_action_err_deg"] = float(np.mean(random_err))
        print(f"  true action ranked first   {100 * hits / len(pick):.1f}%  (chance {100 / args.candidates:.1f}%)")
        print(f"  mean rank of true action   {rank_sum / len(pick):.1f} of {args.candidates}")
        ce, re = np.mean(chosen_err), np.mean(random_err)
        print(f"  chosen action is off by    {ce:.2f} deg per joint  (a random pick: {re:.2f})")

        print(f"\n== 4. rollout, {args.rollout} steps = {args.rollout * k / fps:.1f} s, feeding predictions back in")
        zc, sc = z, s
        for step in range(1, args.rollout + 1):
            t0, t1 = te_t + k * (step - 1), te_t + k * step
            act = (q_all[t1] - q_all[t0]) / qsd
            zc = model(zc, sc, act)
            sc = s_all[t1]  # equals previous state + action, because actions are measured joint deltas
            real = z_all[t1].float()
            l1, l1c = F.l1_loss(zc, real).item(), F.l1_loss(z, real).item()
            dm = (probe(zc.flatten(1)) - q_all[t1]).abs().mean().item()
            dc = (probe(z.flatten(1)) - q_all[t1]).abs().mean().item()
            res[f"rollout/{step}"] = {"l1": l1, "l1_copy": l1c, "decoded_mae": dm, "decoded_mae_copy": dc}
            print(f"  step {step}: feature L1 {l1:.4f} (copy {l1c:.4f})   decoded joints {dm:.2f} deg (copy {dc:.2f})")

    tag = args.model + (f"_train{args.train_seconds:g}s" if args.train_seconds is not None else "")
    torch.save(
        {"state_dict": model.state_dict(), "args": vars(args), "joints": joints, "zmu": zmu.cpu(), "zsd": zsd.cpu(),
         "qmu": qmu.cpu(), "qsd": qsd.cpu()},
        os.path.join(args.run, f"pred_{tag}.pt"),
    )
    with open(os.path.join(args.run, f"pred_{tag}.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nsaved pred_{tag}.pt and .json in {args.run}")


if __name__ == "__main__":
    main()
