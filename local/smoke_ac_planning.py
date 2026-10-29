"""Smoke test for V-JEPA 2-AC: encode the bundled Franka trajectory, evaluate an action
energy grid, and plan one step with CEM. Mirrors notebooks/energy_landscape_example.ipynb
but prints timing per stage so we can size the planning loop on this machine.

    python -m local.smoke_ac_planning                      # bf16 on MPS, checkpoint from checkpoints/
    python -m local.smoke_ac_planning --random-init        # architecture/timing only, no weights
    python -m local.smoke_ac_planning --samples 100 --cem-steps 5 --rollout 2
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import src.hub.backbones as hub
from app.vjepa_droid.transforms import make_transforms
from local.vjepa_local import REPO_ROOT, load_model, n_params, peak_mem_gb, pick_device, sync
from notebooks.utils.mpc_utils import poses_to_diff
from notebooks.utils.world_model_wrapper import WorldModel

TRAJ = os.path.join(REPO_ROOT, "notebooks", "franka_example_traj.npz")


def forward_target(encoder, c, tokens_per_frame, normalize_reps=True):
    """Encode every frame of clip c [B, C, T, H, W] independently (frame duplicated to a 2-frame tubelet)."""
    B, C, T, H, W = c.size()
    c = c.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
    h = encoder(c)
    h = h.view(B, T, -1, h.size(-1)).flatten(1, 2)
    if normalize_reps:
        h = F.layer_norm(h, (h.size(-1),))
    return h


def energy_grid(predictor, h, states, tokens_per_frame, nsamples=5, grid_size=0.075):
    """Roll the predictor one step for an nsamples^3 grid of xyz deltas; return L1 energy vs. the real next frame."""
    axis = np.linspace(-grid_size, grid_size, nsamples)
    actions = torch.tensor(
        [[da, db, dc, 0, 0, 0, 0] for da in axis for db in axis for dc in axis], device=h.device, dtype=h.dtype
    ).unsqueeze(
        1
    )  # [S, 1, 7]
    S = actions.shape[0]
    z0 = h[:, :tokens_per_frame].repeat(S, 1, 1)
    s0 = states[:, :1].to(h.device, h.dtype).repeat(S, 1, 1)
    z1 = predictor(z0, actions, s0)[:, -tokens_per_frame:]
    z1 = F.layer_norm(z1, (z1.size(-1),))
    target = h[:, tokens_per_frame : 2 * tokens_per_frame].expand(S, -1, -1)
    energy = torch.abs(z1 - target).mean(dim=[1, 2])
    return actions[:, 0, :3], energy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--random-init", action="store_true", help="skip weight loading (shape/timing check only)")
    ap.add_argument("--samples", type=int, default=25)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--cem-steps", type=int, default=2)
    ap.add_argument("--rollout", type=int, default=2)
    ap.add_argument("--grid", type=int, default=5, help="energy grid is grid^3 actions")
    args = ap.parse_args()

    device = pick_device(args.device)
    dtype = getattr(torch, args.dtype)
    print(f"device={device} dtype={dtype}")

    t0 = time.time()
    if args.random_init:
        encoder, predictor = hub.vjepa2_ac_vit_giant(pretrained=False)
        encoder, predictor = encoder.to(device).eval(), predictor.to(device).eval()
    else:
        encoder, predictor = load_model("vjepa2_ac_vit_giant", device=device)
    print(f"encoder {n_params(encoder):.0f}M | predictor {n_params(predictor):.0f}M | loaded {time.time() - t0:.1f}s")

    crop_size = 256
    tokens_per_frame = int((crop_size // encoder.patch_size) ** 2)
    transform = make_transforms(
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(1.0, 1.0),
        random_resize_scale=(1.0, 1.0),
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
        crop_size=crop_size,
    )

    traj = np.load(TRAJ)
    np_clips, np_states = traj["observations"], traj["states"]
    gt_action = np.asarray(poses_to_diff(np_states[0, 0], np_states[0, 1]))
    clips = transform(np_clips[0]).unsqueeze(0).to(device)
    states = torch.tensor(np_states, dtype=torch.float32)  # npz is float64, which MPS does not support
    print(f"trajectory: clips {tuple(clips.shape)} states {tuple(states.shape)} gt xyz={np.round(gt_action[:3], 3)}")

    ctx = torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype != torch.float32)
    with torch.inference_mode(), ctx:
        # 1) encode all frames of the trajectory
        for i in range(2):
            sync(device)
            t = time.perf_counter()
            h = forward_target(encoder, clips, tokens_per_frame)
            sync(device)
            print(f"encode {clips.shape[2]} frames: {(time.perf_counter() - t) * 1000:.0f} ms (run {i})")
        h = h.float()

        # 2) energy landscape over an action grid (one predictor step, batch = grid^3)
        sync(device)
        t = time.perf_counter()
        deltas, energy = energy_grid(predictor, h, states, tokens_per_frame, nsamples=args.grid)
        sync(device)
        best = energy.argmin().item()
        print(
            f"energy grid {args.grid}^3={len(energy)} actions: {(time.perf_counter() - t) * 1000:.0f} ms | "
            f"argmin xyz={deltas[best].float().cpu().numpy().round(3)} energy={energy[best]:.4f} "
            f"(min {energy.min():.4f} / max {energy.max():.4f})"
        )

        # 3) CEM planning for the next action
        wm = WorldModel(
            encoder=encoder,
            predictor=predictor,
            tokens_per_frame=tokens_per_frame,
            transform=transform,
            mpc_args={
                "rollout": args.rollout,
                "samples": args.samples,
                "topk": args.topk,
                "cem_steps": args.cem_steps,
                "momentum_mean": 0.15,
                "momentum_mean_gripper": 0.15,
                "momentum_std": 0.75,
                "momentum_std_gripper": 0.15,
                "maxnorm": 0.075,
                "verbose": False,
            },
            normalize_reps=True,
            device=device,
        )
        z_n, z_goal = h[:, :tokens_per_frame], h[:, -tokens_per_frame:]
        s_n = states[:, :1].to(device)
        for i in range(2):
            sync(device)
            t = time.perf_counter()
            action = wm.infer_next_action(z_n, s_n, z_goal)
            sync(device)
            dt = time.perf_counter() - t
            print(
                f"CEM plan (rollout={args.rollout} samples={args.samples} steps={args.cem_steps}): {dt * 1000:.0f} ms "
                f"-> first-step xyz={action[0, :3].float().cpu().numpy().round(3)} (run {i})"
            )

    print(f"peak device mem = {peak_mem_gb(device):.2f} GB")


if __name__ == "__main__":
    main()
