"""Encode an arm_collect.py run with frozen V-JEPA encoders and probe the features for joint angles.

Question answered: do the encoder's features carry the arm's joint state? If a linear map from features
to measured joint angles works on held-out time, a predictor trained on those features has something to
predict. If it does not, no amount of predictor training will help.

    python -m local.arm_encode local/data/arm/20261124_011609
    python -m local.arm_encode local/data/arm/<run> --models vjepa2_1_vit_base_384 --stride 3
    python -m local.arm_encode local/data/arm/<run> --probe-only

Runs in the V-JEPA environment (GPU). For each model it writes `feat_<model>.npz` into the run folder:
  idx     frame indices encoded
  pooled  [N, D]       mean over all patch tokens
  grid    [N, 16 * D]  tokens averaged in a 4x4 spatial grid, which keeps coarse position

Each frame is encoded alone (duplicated to fill the 2-frame tubelet), centre-cropped to a square.

Probe: ridge regression from features to the measured angle of every joint that moved, evaluated with
blocked cross-validation in time (5 contiguous blocks, a 1 s gap dropped next to each test block) so that
neighbouring frames never sit on both sides of the split. The ridge strength is chosen on the tail of the
training part, never on the test block. Baselines: predicting the training mean, and the same ridge on
raw 32x32 grey pixels. Long runs are probed on at most --probe-max evenly spaced frames (the dual-form
ridge builds a frames x frames matrix); the 1 s gap is kept in time, not in samples.
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

from local.vjepa_local import IMAGENET_MEAN, IMAGENET_STD, load_model, n_params, peak_mem_gb, pick_device, sync

DEFAULT_MODELS = ["vjepa2_1_vit_base_384", "vjepa2_1_vit_large_384", "vjepa2_ac_vit_giant"]
LAMBDAS = [1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]


def load_batch(files, size, device, mean, std):
    """jpg paths -> [B, 3, 2, size, size], centre-cropped, ImageNet-normalized."""
    x = torch.from_numpy(np.stack([cv2.imread(f) for f in files])).to(device)
    x = x.flip(-1).permute(0, 3, 1, 2).float().div_(255.0)
    h, w = x.shape[-2:]
    s = min(h, w)
    top, left = (h - s) // 2, (w - s) // 2
    x = x[..., top : top + s, left : left + s]
    x = F.interpolate(x, size=(size, size), mode="bilinear", antialias=True, align_corners=False)
    x = (x - mean) / std
    return x.unsqueeze(2).repeat(1, 1, 2, 1, 1)


def encode_run(run, model, files, idx, size, batch, device, dtype):
    out = load_model(model, device=device)
    encoder = out[0] if isinstance(out, tuple) else out
    del out
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    pooled, grid = [], []
    autocast = torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype != torch.float32)
    t0 = time.perf_counter()
    with torch.inference_mode(), autocast:
        for k in range(0, len(idx), batch):
            x = load_batch([files[i] for i in idx[k : k + batch]], size, device, mean, std)
            z = encoder(x).float()  # [B, N, D]
            b, n, d = z.shape
            side = int(round(n**0.5))
            assert side * side == n, f"expected a square token grid for one tubelet, got {n} tokens"
            pooled.append(z.mean(dim=1).cpu())
            g = F.adaptive_avg_pool2d(z.transpose(1, 2).reshape(b, d, side, side), 4)
            grid.append(g.flatten(1).cpu())
    sync(device)
    dt = time.perf_counter() - t0
    pooled, grid = torch.cat(pooled).numpy(), torch.cat(grid).numpy()
    path = os.path.join(run, f"feat_{model}.npz")
    np.savez(path, idx=idx, pooled=pooled.astype(np.float16), grid=grid.astype(np.float16))
    print(
        f"{model:<26} {n_params(encoder):5.0f}M  {len(idx)} frames in {dt:5.1f} s = {len(idx) / dt:6.1f} fps  "
        f"dim {pooled.shape[1]}  peak mem {peak_mem_gb(device):.1f} GB  -> {os.path.basename(path)}"
    )
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def ridge_fit_predict(xtr, ytr, xte, lam):
    """Dual-form ridge on standardized features: cheap when there are more dimensions than samples."""
    mu, sd = xtr.mean(0, keepdim=True), xtr.std(0, keepdim=True).clamp_min(1e-6)
    xtr, xte = (xtr - mu) / sd, (xte - mu) / sd
    ymu = ytr.mean(0, keepdim=True)
    k = xtr @ xtr.T / xtr.shape[1]
    alpha = torch.linalg.solve(k + lam * torch.eye(len(k), device=k.device, dtype=k.dtype), ytr - ymu)
    return (xte @ xtr.T / xtr.shape[1]) @ alpha + ymu


def probe(x, y, gap, device, folds=5):
    """Blocked CV in time. Returns per-joint (R2, MAE, MAE of the mean predictor), averaged over folds."""
    x = torch.as_tensor(x, dtype=torch.float64, device=device)
    y = torch.as_tensor(y, dtype=torch.float64, device=device)
    n = len(x)
    edges = np.linspace(0, n, folds + 1).astype(int)
    r2s, maes, base = [], [], []
    for f in range(folds):
        a, b = edges[f], edges[f + 1]
        tr = np.r_[0 : max(0, a - gap), min(n, b + gap) : n]
        te = np.arange(a, b)
        cut = int(len(tr) * 0.8)
        fit, val = tr[:cut], tr[cut:]
        errs = [float(((ridge_fit_predict(x[fit], y[fit], x[val], lam) - y[val]) ** 2).mean()) for lam in LAMBDAS]
        pred = ridge_fit_predict(x[tr], y[tr], x[te], LAMBDAS[int(np.argmin(errs))])
        ss_res = ((pred - y[te]) ** 2).sum(0)
        ss_tot = ((y[te] - y[tr].mean(0)) ** 2).sum(0)
        r2s.append((1 - ss_res / ss_tot).cpu().numpy())
        maes.append((pred - y[te]).abs().mean(0).cpu().numpy())
        base.append((y[te] - y[tr].mean(0)).abs().mean(0).cpu().numpy())
    return np.mean(r2s, 0), np.mean(maes, 0), np.mean(base, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="folder written by arm_collect.py")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--stride", type=int, default=1, help="encode every Nth frame")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--probe-only", action="store_true", help="reuse saved features")
    ap.add_argument("--probe-max", type=int, default=12000, help="probe on at most this many frames")
    args = ap.parse_args()

    device = pick_device(args.device)
    log = np.load(os.path.join(args.run, "log.npz"))
    meta = json.load(open(os.path.join(args.run, "meta.json")))
    files = sorted(glob.glob(os.path.join(args.run, "frames", "*.jpg")))
    assert len(files) == len(log["t_frame"]), "frame count does not match the log"
    idx = np.arange(0, len(files), args.stride)
    print(f"run {args.run}: {len(files)} frames, encoding {len(idx)} at {args.size}px on {device}")

    if not args.probe_only:
        for model in args.models:
            encode_run(args.run, model, files, idx, args.size, args.batch, device, getattr(torch, args.dtype))

    joints = [j for j in meta["joints"] if j in meta["moving"]]
    cols = [meta["joints"].index(j) for j in joints]
    fps = (len(log["t_frame"]) - 1) / (log["t_frame"][-1] - log["t_frame"][0])

    pstride = -(-len(idx) // args.probe_max)
    pidx = idx[::pstride]
    if pstride > 1:
        print(f"probing on every {pstride}th encoded frame: {len(pidx)} frames")
    rows = []
    g = np.stack([cv2.resize(cv2.imread(files[i], cv2.IMREAD_GRAYSCALE)[:, 140:500], (32, 32)) for i in pidx])
    rows.append(("raw pixels 32x32", g.reshape(len(pidx), -1).astype(np.float64), pidx))
    for model in args.models:
        path = os.path.join(args.run, f"feat_{model}.npz")
        if not os.path.exists(path):
            print(f"(no features for {model}, skipping)")
            continue
        z = np.load(path)
        short = model.replace("vjepa2_1_", "2.1 ").replace("vjepa2_", "2 ").replace("_384", "")
        keep = np.isin(z["idx"], pidx)
        rows.append((f"{short} pooled", z["pooled"][keep], z["idx"][keep]))
        rows.append((f"{short} grid4x4", z["grid"][keep], z["idx"][keep]))

    print(f"\n== linear probe, blocked 5-fold CV in time, {len(joints)} joints, angles in degrees")
    y_all = log["pos_deg"][:, cols]
    print(f"{'joint range (deg)':<28}" + "".join(f"{np.ptp(y_all[:, k]):>22.1f}" for k in range(len(joints))))
    print(f"{'features':<28}" + "".join(f"{j + ' R2 / MAE':>22}" for j in joints))
    summary = {}
    for name, x, ix in rows:
        gap = max(1, int(round(fps / (ix[1] - ix[0]))))  # 1 second
        r2, mae, base = probe(x, y_all[ix], gap, device)
        if "mean" not in summary:
            summary["mean"] = base
            print(f"{'training mean':<28}" + "".join(f"{0.0:>13.2f} / {b:5.2f}" for b in base))
        summary[name] = (r2, mae)
        print(f"{name:<28}" + "".join(f"{a:>13.2f} / {b:5.2f}" for a, b in zip(r2, mae)))
    out = {}
    for k, val in summary.items():
        out[k] = [np.asarray(v).tolist() for v in (val if isinstance(val, tuple) else (val,))]
    with open(os.path.join(args.run, "probe.json"), "w") as f:
        json.dump({"joints": joints, "results": out}, f, indent=2)


if __name__ == "__main__":
    main()
