"""Smoke test: load a V-JEPA checkpoint, encode a real video clip, report latency and memory.

Examples (from the repo root, with .venv active):

    python -m local.smoke_inference --model vjepa2_1_vit_base_384 --frames 16 --size 256
    python -m local.smoke_inference --model vjepa2_1_vit_large_384 --frames 32 --size 384
    python -m local.smoke_inference --model vjepa2_ac_vit_giant --frames 8 --size 256
"""

import argparse
import glob
import os
import time

import torch

from local.vjepa_local import REPO_ROOT, SPECS, load_model, load_video, n_params, peak_mem_gb, pick_device, sync


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vjepa2_1_vit_base_384", choices=sorted(SPECS))
    ap.add_argument("--ckpt", default=None, help="path to .pt (default: checkpoints/<spec file>)")
    ap.add_argument("--video", default=None, help="mp4 path (default: first clip under local/data)")
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--device", default=None, help="mps | cpu | cuda (default: auto)")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    device = pick_device(args.device)
    dtype = getattr(torch, args.dtype)
    video = args.video or sorted(glob.glob(os.path.join(REPO_ROOT, "local/data/**/*.mp4"), recursive=True))[0]
    print(f"device={device} dtype={dtype} video={os.path.relpath(video, REPO_ROOT)}")

    # img_size is fixed by each hub builder; RoPE is interpolated so other input sizes still work.
    encoder, predictor = load_model(args.model, args.ckpt, device, num_frames=args.frames)
    print(f"encoder {n_params(encoder):.0f}M params | predictor {n_params(predictor):.0f}M params")

    x = load_video(video, args.frames, args.size).to(device)
    print(f"input {tuple(x.shape)}  (B C T H W)")

    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype != torch.float32):
        for i in range(args.runs):
            sync(device)
            t0 = time.perf_counter()
            feats = encoder(x)
            sync(device)
            dt = time.perf_counter() - t0
            print(f"run {i}: encoder {dt * 1000:.0f} ms -> {tuple(feats.shape)} {feats.dtype}")

    print(f"tokens/clip = {feats.shape[1]} ; peak device mem = {peak_mem_gb(device):.2f} GB")
    finite = torch.isfinite(feats).all().item()
    print(f"feature stats: mean={feats.float().mean():.4f} std={feats.float().std():.4f} finite={finite}")


if __name__ == "__main__":
    main()
