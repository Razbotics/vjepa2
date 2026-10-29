# V-JEPA 2 on this Mac — getting started

Local notes for running and hacking on `facebookresearch/vjepa2` on an Apple-silicon
laptop until the CUDA box arrives. Everything here lives on the `mac-dev` git branch;
`main` tracks upstream untouched so you can rebase later.

Machine this was validated on: Apple M5 Pro, 16 cores, 24 GB unified memory, macOS 26,
Python 3.12 (uv-managed), torch 2.15 with the MPS backend.

## Layout

```
vjepa2/
├── .venv/                 # uv virtualenv (ignored)
├── checkpoints/           # downloaded .pt files (ignored)
├── local/                 # everything Mac/laptop specific (tracked on mac-dev)
│   ├── README.md          # this file
│   ├── vjepa_local.py     # build model + load a local checkpoint, video loading, timing helpers
│   ├── smoke_inference.py # encode a clip, print latency / memory / feature stats
│   ├── configs/           # training configs sized for one laptop
│   └── data/              # sample clips + CSVs (ignored)
├── src/                   # the package: models, masks, datasets, utils
├── app/                   # training loops (vjepa, vjepa_2_1, vjepa_droid = action-conditioned)
├── evals/                 # attentive-probe evals
└── configs/               # upstream cluster configs (8-32 nodes, keep for reference)
```

## Setup (already done, here for reproducibility)

```bash
cd ~/Razbotics/vjepa2
uv venv .venv --python 3.12
grep -v '^decord$' requirements.txt > /tmp/reqs.txt
uv pip install --python .venv/bin/python -r /tmp/reqs.txt -r requirements-test.txt decord2
uv pip install --python .venv/bin/python --no-deps -e .
source .venv/bin/activate
```

Why the dance around `decord`: upstream's video reader has no macOS build. `decord2`
(3.4.0) ships a Python 3.12 arm64 wheel and keeps the `import decord` name, so no
source changes are needed. `eva-decord` only has wheels up to Python 3.11.

Known wart: `decord2` and `opencv-python` each bundle their own ffmpeg `libavdevice`
and macOS prints an "implemented in both" objc warning at import. It has not caused a
crash in these tests. If it ever does, swap `opencv-python` for `opencv-python-headless`
or build decord against the system ffmpeg.

## Checkpoints

Downloaded into `checkpoints/` (resumable with `curl -L -C -`):

| File | Model | Size | Use |
|---|---|---|---|
| `vjepa2_1_vitb_dist_vitG_384.pt` | V-JEPA 2.1 ViT-B/16, 384px | 1.6 GB | fastest full-quality encoder for laptop work |
| `vjepa2_1_vitl_dist_vitG_384.pt` | V-JEPA 2.1 ViT-L/16, 384px | 4.8 GB | the encoder PLAN.md targets |
| `vjepa2-ac-vitg.pt` | V-JEPA 2-AC (ViT-g encoder + action-conditioned predictor) | 11 GB | Phase 1 planning baseline |

Others (same CDN, `https://dl.fbaipublicfiles.com/vjepa2/<file>`): `vitl.pt`, `vith.pt`,
`vitg.pt`, `vitg-384.pt` (V-JEPA 2), `vjepa2_1_vitg_384.pt`, `vjepa2_1_vitG_384.pt` (2.1
giant/gigantic). The checkpoint files are full training states (encoder, EMA/target
encoder, predictor, optimizer), which is why they are 3-4x the parameter count.

Upstream bug fixed on `mac-dev`: `src/hub/backbones.py` shipped with
`VJEPA_BASE_URL = "http://localhost:8300"`, so `torch.hub.load(...)` could not download
anything. It now defaults to the public CDN and honours a `VJEPA_BASE_URL` env var.

## Inference

```bash
python -m local.smoke_inference --model vjepa2_1_vit_base_384 --frames 16 --size 256
python -m local.smoke_inference --model vjepa2_1_vit_large_384 --frames 64 --size 384
python -m local.smoke_inference --model vjepa2_ac_vit_giant --frames 8 --size 256
```

`local/vjepa_local.py::load_model(name, ckpt_path, device, num_frames=...)` returns
`(encoder, predictor)` built from the upstream hub constructors with weights loaded from
an explicit path (no torch-hub cache copy). The `SPECS` table there records which key in
each checkpoint holds the encoder to use (`ema_encoder` for 2.1 B/L, `target_encoder`
for the rest, `encoder` for 2-AC).

Measured on the M5 Pro, bf16 autocast, batch 1, encoder only, after warm-up:

| Model | Frames x px | Tokens | Latency | Peak GPU mem |
|---|---|---|---|---|
| 2.1 ViT-B | 16 x 256 | 2 048 | 54 ms | 0.6 GB |
| 2.1 ViT-B | 64 x 256 | 8 192 | 270 ms | 1.5 GB |
| 2.1 ViT-B | 32 x 384 | 9 216 | 315 ms | 1.5 GB |
| 2.1 ViT-B | 64 x 384 (native) | 18 432 | 1.1 s | 2.5 GB |
| 2.1 ViT-L | 16 x 256 | 2 048 | 146 ms | 2.1 GB |
| 2.1 ViT-L | 64 x 256 | 8 192 | 764 ms | 2.1 GB |
| 2.1 ViT-L | 64 x 384 (native) | 18 432 | 2.7 s | 3.1 GB |

Takeaways: both distilled 2.1 encoders run at full resolution well inside unified
memory; ViT-L costs roughly 2.5-3x ViT-B at every setting. Attention cost grows with
the square of the token count, so for a planning loop keep clips short (8-16 frames)
and 256px, which is what V-JEPA 2-AC uses anyway. The 2-AC ViT-g encoder is ~1B params
(2 GB of weights in bf16) and fits; see `local/smoke_ac_planning.py` for the measured
planning-step latency.

## Action-conditioned planning (V-JEPA 2-AC)

```bash
python -m local.smoke_ac_planning                      # bf16 on MPS, weights from checkpoints/
python -m local.smoke_ac_planning --random-init        # architecture + timing only
python -m local.smoke_ac_planning --samples 100 --cem-steps 5 --rollout 2
```

This mirrors `notebooks/energy_landscape_example.ipynb` on the bundled Franka
trajectory (`notebooks/franka_example_traj.npz`): encode the frames, score a 5x5x5 grid
of xyz deltas with one predictor step, then plan with the repo's CEM `WorldModel`
wrapper (`notebooks/utils/`). The predictor takes `(context tokens, actions [S,T,7],
poses [S,T,7])` and returns next-frame tokens; CEM batches all samples into one call.

Measured on the M5 Pro with the released weights (bf16 autocast, fp32 weights, 256px,
256 tokens per frame):

| Stage | Time | Note |
|---|---|---|
| Build + load ViT-g encoder (1.0B) + AC predictor (305M) | 13 s | 11 GB checkpoint, mmap load |
| Encode 2 frames | 108 ms | ~55 ms per frame |
| Energy grid, 125 actions, 1 predictor step | 1.7 s | batch of 125 |
| CEM plan, rollout 2, 25 samples, 2 steps | 2.0 s | notebook's "CPU-sized" settings |
| Peak device memory | 8.1 GB | fp32 weights; cast to bf16 to roughly halve |

Sanity check that the weights are doing real work: the ground-truth first action on the
Franka trajectory is xyz = (+0.092, +0.031, +0.084). The energy grid's minimum lands on
the all-positive corner (+0.075, +0.075, +0.075) with energy 0.417 against a maximum of
0.488, and two CEM plans returned first steps of (+0.031, +0.047, +0.044) and
(+0.050, +0.035, +0.049). With random weights the same grid is flat to four decimals.

So the Phase-1 planning loop runs here, but at seconds per action, which is exactly the
latency gap PLAN.md Phase 2 is about. Scaling CEM to the paper's 400 samples x 10 steps
would be ~minutes per action on this GPU; use the laptop for correctness and small
sweeps, the CUDA box for the real latency numbers.

## Training on the laptop

The upstream trainer assumed CUDA + NCCL + multi-rank. Patches on `mac-dev`:

- `app/vjepa_2_1/train.py`: picks `cuda` > `mps` > `cpu`; `torch.autocast(device_type=...)`
  instead of `torch.cuda.amp.autocast`; device-agnostic peak-memory logging; skips
  `DistributedDataParallel` when `world_size == 1`; embed-dim lookup covers tiny/small/base
  (upstream only knew L/g/G).
- `app/vjepa_2_1/utils.py`: `GradScaler` is created disabled when CUDA is absent (bf16
  does not need it anyway).
- `app/vjepa_2_1/models/predictor.py`: deep-supervision tap layers fall back to four evenly
  spaced blocks for predictor depths not in the hard-coded table.
- `src/utils/distributed.py`: `gloo` backend when CUDA is absent.
- `notebooks/utils/mpc_utils.py`: `compute_new_pose` casts the numpy float64 pose to the
  tensor dtype before moving it to the device (MPS has no float64).

Smoke config: `local/configs/vits_mac_smoke.yaml` (ViT-S encoder, 6-layer predictor,
16 frames at 256px, batch 2, 10 Kinetics clips, 2 epochs x 4 iterations).

```bash
python -m app.main --fname local/configs/vits_mac_smoke.yaml --devices mps
# single-process, easier to step through in a debugger:
python -m app.main --fname local/configs/vits_mac_smoke.yaml --debugmode 1
```

Logs and checkpoints land in `local/runs/<name>/` (`log_r0.csv`, `e<N>.pth.tar`,
`latest.pth.tar`, `params-pretrain.yaml`). The CSV header is written twice; that is an
upstream `CSVLogger` quirk, skip one row when parsing.

Measured smoke run (ViT-S + 6-layer predictor, 2 x 16 x 256px clips per step, bf16):

| | |
|---|---|
| Encoder / predictor params | 22 M / 12.6 M |
| Step time, data warm | 0.55-1.0 s |
| Step time, first batch (decord decode) | 7-8 s |
| Loss, epoch 1 -> 2 | 0.855 -> 0.822 |
| `mem` column (MPS driver-allocated, includes allocator cache) | ~13-14 GB |
| Checkpoint size (encoder + EMA encoder + predictor + AdamW state) | 480 MB |

Two things to keep in mind from that run. Data loading is the bottleneck at this scale
(the first batch of every loader refresh costs several seconds of video decoding), so
raise `num_workers` or pre-decode clips to tensors for real experiments. And the `mem`
figure is the Metal driver's view, which keeps freed blocks cached; the live working
set is far smaller. Use `torch.mps.current_allocated_memory()` if you need the exact
number, and `PYTORCH_MPS_HIGH_WATERMARK_RATIO` if the cache ever crowds out other apps.

Dataset CSV format: one line per clip, space separated, `"<abs path> <label>"`
(label is ignored by pretraining). `local/data/kinetics_mini.csv` is the sample.

Config notes for laptop runs:
- Drop the `img_data` block. The trainer splits ranks into image and video workers and
  divides by the number of image ranks, which is zero with one process.
- `pin_mem: false` (pinned memory is a CUDA concept).
- `ipe` (iterations per epoch) is independent of dataset size; the loader restarts.
- `meta.load_checkpoint: false` for a fresh run; `read_checkpoint: <path>` to resume.

## Gotchas

- **Case-insensitive filesystem.** Upstream has both `configs/train_2_1/vitG16` (2B) and
  `vitg16` (1B), likewise `configs/eval_2_1/vitG-384` and `vitg-384`. On APFS they collide.
  The lowercase (1B) copies are flagged `git update-index --skip-worktree` so `git status`
  stays clean; the on-disk `vitG16` / `vitG-384` contain the 2B configs. Need the 1B ones?
  `git show main:configs/train_2_1/vitg16/pretrain-256px-16f.yaml`.
- `torch.backends.cuda.sdp_kernel()` in `src/models/utils/modules.py` is deprecated but
  works as a no-op on MPS. SDPA itself runs on MPS.
- `app/main.py --devices` accepts any string; it only sets `CUDA_VISIBLE_DEVICES`. Pass one
  entry (`mps`) to get one process.
- V-JEPA 2-AC and the `app/vjepa_droid` trainer have not been patched yet; the same
  four CUDA assumptions will apply there.

## Suggested next steps for the drone world-model plan

1. Wire `load_model("vjepa2_ac_vit_giant")` into an image-goal MPC loop (the energy
   landscape notebook in `notebooks/` is the reference) and measure planning latency per
   step on MPS. ViT-g is ~1B params, ~2 GB in bf16, so it fits.
2. Fine-tune / post-train on the laptop only with ViT-S/B encoders and short clips; keep
   the ViT-L and ViT-g runs for the CUDA box.
3. When the GPU arrives nothing in `local/` needs to change: `pick_device()` prefers CUDA
   and the trainer patches keep the upstream CUDA/NCCL path.
