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

Downloaded into `checkpoints/` (resumable with `curl -L -C -`). Since 2026-11-24 they live on the GPU
desktop only; the Mac keeps none, so anything that loads a model runs on aorus.

| File | Model | Size | Use |
|---|---|---|---|
| `vjepa2_1_vitb_dist_vitG_384.pt` | V-JEPA 2.1 ViT-B/16, 384px | 1.6 GB | fastest full-quality encoder for laptop work |
| `vjepa2_1_vitl_dist_vitG_384.pt` | V-JEPA 2.1 ViT-L/16, 384px | 4.8 GB | the encoder PLAN.md targets |
| `vjepa2-ac-vitg.pt` | V-JEPA 2-AC (ViT-g encoder + action-conditioned predictor) | 11 GB | Phase 1 planning baseline; unused by the SO-101 pipeline, deleted on both machines 2026-11-24 (re-download from the CDN below if needed) |

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

## Remote GPU desktop (`aorus`)

The CUDA box is the `aorus` entry in `~/.ssh/config` (AORUS-PRO, Ubuntu 22.04,
RTX 4060 Ti 16 GB, 12 cores, 31 GB RAM, 192.168.29.2 on the home LAN; driver
580.196.07, CUDA 13.0). Key auth with
`~/.ssh/id_ed25519`; the repo lives at `~/projects/vjepa2` there with its own `.venv`
and `checkpoints/`.

```bash
local/remote/run_on_aorus.sh                       # rsync, then an interactive shell in the repo
local/remote/run_on_aorus.sh nvidia-smi
local/remote/run_on_aorus.sh python -m pytest tests -q
local/remote/run_on_aorus.sh python -m local.smoke_inference --model vjepa2_1_vit_large_384 --frames 64 --size 384
local/remote/run_on_aorus.sh python -m app.main --fname local/configs/vits_mac_smoke.yaml --devices cuda:0
```

Code moves through git. `origin` is the fork `github.com/Razbotics/vjepa2` on both
machines and `upstream` is `facebookresearch/vjepa2`; `mac-dev` tracks `origin/mac-dev`,
`main` tracks `upstream/main` (so `git pull` on `main` brings in Meta's changes, then
`git merge main` on `mac-dev`). The helper pushes the current branch to the fork, then
on the desktop fetches and hard-resets that branch to `origin/<branch>`, and runs the
command inside the remote venv with output streaming back. Only committed work travels;
it warns when the tree is dirty. `SYNC=rsync` copies the working tree instead
(uncommitted edits included, minus `.venv`, `checkpoints/`, `local/runs`, `local/data`)
for quick experiments; `NO_SYNC=1` skips syncing; `REMOTE=` / `REMOTE_DIR=` retarget it.
Edit and commit on the Mac, run on the desktop; results stay in
`~/projects/vjepa2/local/runs` there. The desktop pulls the public fork over HTTPS, so
it needs no GitHub credentials; pushing is only ever done from the Mac.

Setup notes, in the order they bit:
- **Secure Boot.** Ubuntu's DKMS-built NVIDIA modules are signed with a per-machine
  Machine Owner Key (`/var/lib/shim-signed/mok/MOK.der`). Until that key is enrolled in
  firmware the kernel logs "Loading of module with unavailable key is rejected" and
  `nvidia-smi` cannot find the driver. Fix: `sudo update-secureboot-policy --enroll-key`
  (or `sudo mokutil --import /var/lib/shim-signed/mok/MOK.der`), set a one-time
  password, reboot, and at the blue MOK Manager screen choose Enroll MOK, Continue
  ("View key 0" only shows the cert), Yes, type the password, Reboot. Needs a keyboard
  and monitor on the box that one time. The screen times out in ~10 s and boots normally;
  the request stays queued (`MokNew` in `/sys/firmware/efi/efivars`) until the next boot.
  Check with `mokutil --list-enrolled | grep AORUS`.
- **Driver.** Upgraded from 560-open to `nvidia-driver-580-open` (NVIDIA CUDA apt repo).
  580 is the production branch and the minimum for the CUDA 13 runtime bundled in
  PyPI's default `torch` wheel (`torch==2.15.0` pulls `nvidia-cuda-runtime==13.2`).
  With the old 560 driver only the `cu126` index (`torch 2.14.1+cu126`) would have worked.
- **Headless + no sleep.** `sudo bash local/remote/ubuntu_headless_nosleep.sh` masks the
  systemd sleep targets, sets `IdleAction=ignore` for logind, and disables suspend and
  blanking for both the GDM login screen and the user's GNOME session (via the user's
  session bus at `/run/user/<uid>/bus`; `dbus-launch` is not installed there). Run it
  *before* any long install: the box suspended itself on GNOME's 20-minute idle timer in
  the middle of the driver install, which killed the SSH session and the MOK dialog with
  it. In the BIOS also enable "power on after AC loss". The box does not need a monitor
  for CUDA work.
- **Network.** The desktop is on Wi-Fi via a USB dongle (~5 MB/s). The onboard
  `eno1` port has no cable. Plug one in: the 17 GB of checkpoints and the CUDA torch
  wheels (~3 GB) are painful over Wi-Fi, and the laptop talks to it over the same link.
- Working away from home later: install Tailscale on both machines and point the
  `aorus` HostName at its tailnet name; nothing else changes.

### GPU numbers (RTX 4060 Ti 16 GB, torch 2.15.0+cu132)

Same scripts, same configs, run through `run_on_aorus.sh` on 2026-11-04. Encoder only,
bf16 autocast, batch 1, after warm-up:

| Model | Frames x px | Tokens | Mac M5 Pro (MPS) | aorus (CUDA) | Speed-up |
|---|---|---|---|---|---|
| 2.1 ViT-B | 16 x 256 | 2 048 | 54 ms, 0.6 GB | 26 ms, 0.5 GB | 2.1x |
| 2.1 ViT-B | 64 x 256 | 8 192 | 270 ms, 1.5 GB | 140 ms, 0.7 GB | 1.9x |
| 2.1 ViT-B | 64 x 384 (native) | 18 432 | 1.1 s, 2.5 GB | 480 ms, 1.1 GB | 2.3x |
| 2.1 ViT-L | 16 x 256 | 2 048 | 146 ms, 2.1 GB | 73 ms, 1.3 GB | 2.0x |
| 2.1 ViT-L | 64 x 256 | 8 192 | 764 ms, 2.1 GB | 385 ms, 1.6 GB | 2.0x |
| 2.1 ViT-L | 64 x 384 (native) | 18 432 | 2.7 s, 3.1 GB | 1.32 s, 2.1 GB | 2.0x |

V-JEPA 2-AC planning (`local/smoke_ac_planning.py`, real weights, bf16):

| Stage | Mac M5 Pro (MPS) | aorus (CUDA) |
|---|---|---|
| Load encoder + predictor from the 11 GB checkpoint | 13 s | 15 s (over Wi-Fi-copied file, NVMe) |
| Encode 2 frames | 108 ms | 78 ms |
| Energy grid, 125 actions, 1 predictor step | 1.7 s | 1.04 s |
| CEM plan, rollout 2, 25 samples, 2 steps | 2.0 s | 1.2 s |
| Peak device memory | 8.1 GB | 6.0 GB |

Same sanity check on both machines: grid argmin at (+0.075, +0.075, +0.075) with energy
0.413 vs max 0.486, CEM first steps (+0.062, +0.046, +0.043) and (+0.030, +0.053, +0.055)
against ground truth (+0.092, +0.031, +0.084).

Other runs: `pytest tests` 26 passed in 53 s (the Mac skips the 3 CUDA-only tests);
`vits_mac_smoke.yaml --devices cuda:0` reaches the same loss (0.855 -> 0.822) with the
`gpu` column at 340-640 ms and 2.3 GB, but the `data` column at ~1 s, so this config is
data-bound on both machines. Raise `num_workers` and batch size before reading anything
into GPU step time. The Mac `mem` column is driver-allocated memory while the CUDA one is
`torch.cuda.max_memory_allocated`, so those two are not the same quantity.

A 4060 Ti is only ~2x the M5 Pro on these encoders: both are memory-bandwidth-bound at
batch 1 and the GPU is not fed. The gap widens with work per call: a CEM plan at
200 samples x 5 steps x rollout 2 takes 25 s per action on aorus at 9.9 GB peak (and
lands exactly on the +0.075 corner), a setting the laptop cannot hold in memory. That is
still ~25x off real time for the paper's 400 x 10 planner, so predictor batching and a
smaller predictor are the Phase-2 levers, not a bigger GPU.

## Live camera (Logitech C270 on aorus)

The C270 is plugged into the desktop as `/dev/video0` (USB 046d:0825, 720p MJPG).
`local/live_camera.py` grabs frames in a background thread, keeps a rolling window,
preprocesses on the GPU and encodes the newest clip as fast as the device allows.
Headless: one line per second plus a summary.

```bash
local/remote/run_on_aorus.sh python -m local.live_camera                      # ViT-L, 16 frames @256
local/remote/run_on_aorus.sh python -m local.live_camera --model vjepa2_1_vit_base_384
local/remote/run_on_aorus.sh python -m local.live_camera --frames 8 --stride 4 --exposure 150
```

Measured 2026-11-23, RTX 4060 Ti, bf16, 256 px, camera at 29.8 fps:

| Model | Clip | Preprocess | Encoder | Newest-frame age | Loop rate | Peak mem |
|---|---|---|---|---|---|---|
| 2.1 ViT-L | 16 frames, stride 2 (1 s) | 11 ms | 72 ms | 107 ms (p95 121) | 11 Hz | 1.4 GB |
| 2.1 ViT-B | 16 frames, stride 2 (1 s) | 11 ms | 25 ms | 61 ms (p95 75) | 23 Hz | 0.6 GB |
| 2.1 ViT-L | 8 frames, stride 4 (0.9 s) | 6 ms | 48 ms | 73 ms (p95 88) | 17.5 Hz | 1.3 GB |

"Newest-frame age" is wall time from the capture of the last frame in the clip to the
encoder returning, i.e. the staleness a planner would act on. It is roughly
preprocess + encoder + up to one camera frame of waiting. `d_ref` / `d_prev` are cosine
distances of mean-pooled features against the first / previous clip: ~0.000 with the
camera still, a few hundredths when the scene changes.

Camera gotchas, all handled by the script on every start because V4L2 controls reset on
replug and reboot:
- `exposure_dynamic_framerate` defaults to on and silently drops the camera to ~20 fps in
  a dim room. The script turns it off; the model assumes a constant time step.
- `power_line_frequency` defaults to 60 Hz; India is 50 Hz (`--mains-hz`).
- Do not set `CAP_PROP_BUFFERSIZE=1` in OpenCV: the C270 then delivers exactly 15 fps.
- `--exposure N` locks manual exposure (units of 100 us). Above ~330 the exposure no
  longer fits in a 33 ms frame and the rate falls (400 gave 25 fps). 150 is sharp but
  dark indoors; auto-exposure is the default.
- `v4l2-ctl` comes from `sudo apt-get install v4l-utils`. Device access over SSH works
  through the logged-in seat's ACL; add the user to the `video` group if the box is ever
  switched to a text-only boot.

### Live image-goal readout (V-JEPA 2-AC)

`local/live_goal.py` encodes a goal image once with the 2-AC ViT-g encoder, then for every
new camera frame reports the latent distance to the goal and asks the action-conditioned
predictor which of 7 candidate moves (stay, +/-x, +/-y, +/-z by 5 cm) lands closest to it.

```bash
local/remote/run_on_aorus.sh python -m local.live_goal                             # goal = current view
local/remote/run_on_aorus.sh python -m local.live_goal --goal local/runs/goal.jpg  # goal from a file
local/remote/run_on_aorus.sh python -m local.live_goal --no-actions                # distance only
```

Measured 2026-11-23 on the RTX 4060 Ti, bf16, 256 px, one frame per decision:

| Mode | Encode | Predictor x7 | Frame age at output | Rate | Peak mem |
|---|---|---|---|---|---|
| Goal distance + 7 candidate actions | 80 ms | 83 ms | 180 ms (p95 197) | 6.1 Hz | 5.1 GB |
| Goal distance only | 79 ms | n/a | 97 ms (p95 113) | 12.5 Hz | 5.0 GB |

What the two distance columns read, same camera, same room:

| Goal image | `E_goal` (patch-token L1) | `d_pool` (pooled cosine) | Best action |
|---|---|---|---|
| The identical frame | 0 | 0 | n/a |
| Live view, static scene | 0.46 (0.44-0.48) | 0.004 (0.002-0.011) | stay, 67 of 68 decisions |
| Frame from before the camera was moved | 0.70 | 0.21 | +z every time |
| Unrelated video frame | 0.71 | 0.30 | +z every time |

Things this taught us:
- **`E_goal` has a high noise floor.** The quantity the paper's planner minimizes reads
  0.46 between two frames of a static scene, against 0.71 for an unrelated one. Sensor
  noise alone moves ViT-g patch tokens that far: Gaussian noise of sigma 1/255 on a clean
  image already gives 0.27, sigma 2 gives 0.33, a 4 px shift 0.49. The bundled Franka
  example bottoms out at 0.41 for the same reason. So the planner works inside a band about
  0.25 wide. It is stable (+/-0.02), so it is usable, but it is not a distance-to-goal in
  any metric sense. Averaging 4 frames only lowers the floor to 0.40.
- **Pooling fixes the separation but not the planning.** Mean-pooling tokens before
  comparing gives 0.004 vs 0.2-0.3, a ~50x gap, because per-patch noise averages out. That
  is the right signal for "have I arrived", but it discards the spatial layout the
  predictor is supposed to steer.
- **The predictor does prefer staying when it is already at the goal** (spread across
  candidates 0.05, "stay" lowest in 67 of 68 decisions). Away from the goal it always says
  +z with a small spread (0.01-0.02); that is not meaningful. 2-AC was trained on a Franka
  arm from a fixed exocentric camera and is fed a fixed arm pose here, so for a moving
  camera the action column is a plumbing and latency check only.
- **Latency.** 6 Hz for encode plus 7 one-step rollouts is already half of the 300 ms
  decision budget in PLAN.md, with a 1B-param encoder and no CEM. A distilled encoder (the
  2.1 ViT-B runs in 25 ms) with a predictor trained on our own actions is the route.
- **Camera warm-up.** The C270 returns black frames for about a second after opening; the
  script waits for brightness to settle before taking a goal frame. Compute `d_pool` with an
  elementwise product, not `@`: under autocast the matmul runs in bf16 and quantizes it.

## SO-101 arm on aorus

The SO-101 follower (no leader arm) is on `aorus` at `/dev/ttyACM0`; scripts use the stable
`/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B91028149-if00`. LeRobot 0.6.2 (commit `ae2b57f`, same as
the Mac checkout) lives in its own environment, `~/projects/lerobot/.venv`, because it pins `torch<2.12`
and the V-JEPA environment runs 2.15. Calibration id is `my_follower_arm`; the file was copied from the Mac
to `~/.cache/huggingface/lerobot/calibration/robots/so_follower/` and matches the servo EEPROMs.

`local/arm_collect.py` drives the joints named in `--joints` through small random moves around the start
pose and logs, for every camera frame, the measured joint positions, goals and loads:

```bash
local/remote/run_on_aorus.sh '~/projects/lerobot/.venv/bin/python' -m local.arm_collect --seconds 20
```

Output goes to `local/data/arm/<timestamp>/` (`frames/*.jpg`, `log.npz`, `meta.json`). First run, wrist
roll and gripper only, at 640x480 (the default is now 640x360, see below):

| | |
|---|---|
| Log rate | 30.1 Hz, one sample per camera frame, no frames dropped |
| Joint read after frame capture | median 3.3 ms, p95 4.2 ms |
| Settled error at each target | mean 3.5 ticks, max 5 ticks (0.44 deg) |
| Peak servo load | 76 of 1000 |
| Size | 24 MB for 20 s |

Things to know:

- The servos run P = 16, I = 0, D = 32 and stop up to 5 ticks short of the goal. Use `pos_raw` or `pos_deg`
  (measured) as ground truth, never `goal_raw`.
- The script connects the motor bus only. `SO101Follower.connect()` also rewrites gains and re-enables
  torque on every joint, which is not wanted for a first contact.
- Speed and acceleration limits are RAM registers and reset on power-cycle, so they are set every run.
- In the folded rest pose `shoulder_lift` and `elbow_flex` sit a few ticks outside the calibrated range, on
  their hard stops. Goals are clipped to the range.
- Camera field of view: the C270 has no zoom control and 1280x720 is its widest mode. 640x360 shows the
  same view scaled down; the 4:3 modes (640x480) crop the left and right edges, and 1280x960 silently falls
  back to 1280x720. `preprocess` then centre-crops a square, so the encoder only sees the middle 56% of the
  width.
#### Live preview for aiming the camera

```bash
local/remote/camera_preview.sh        # then open http://localhost:8765 on the laptop
```

Starts `local/camera_preview.py` on the desktop and forwards the port over SSH (the server binds to
localhost only). The page shows the feed at about 15 fps with a yellow square marking the centre crop the
encoder sees, plus mean brightness and the share of blown-out pixels. `/snapshot.jpg` returns one frame.
Ctrl-C stops it; it must be stopped before `arm_collect.py` or the `live_*` scripts, because only one
program can hold the camera.

#### Ready pose and the side camera

`--ready table` first moves the arm from its rest pose to a ready pose (elbow, then wrist, one joint at a
time), random-walks around that pose, and leaves by the same path in reverse:

```bash
local/remote/run_on_aorus.sh '~/projects/lerobot/.venv/bin/python' -m local.arm_collect \
    --ready table --joints elbow_flex wrist_flex wrist_roll gripper --seconds 30
```

The pose (raw ticks: elbow 2782, wrist flex 1438, others as at rest) holds the gripper level about 12 cm
above the board. It was chosen by stepping the arm with the camera watching from the side. The run refuses
to start unless the arm rests within `--rest-tol` ticks of the rest pose stored in `READY_POSES`, because
the approach path is only known to be clear from there.

First 30 s run at the ready pose, four joints moving +/-10 deg, 640x360:

| | |
|---|---|
| Samples | 901 at 30.0 Hz, 901 frames on disk, none dropped |
| Targets visited | 34 |
| Settled error, elbow | mean +13 ticks, max 27 (2.4 deg), always short on the way up |
| Settled error, wrist flex | mean 6 ticks, max 12 |
| Settled error, wrist roll and gripper | mean 3 to 4 ticks, max 6 |
| Peak load | 160 of 1000 (elbow) |
| Size | 31 MB |

- Camera placement: look across the arm from the side, 60 to 80 cm away. Views from above or behind lose
  the gripper as soon as it lifts, because it moves toward the lens and out of the top of the frame.
- Direction of travel: decreasing `elbow_flex` raises the gripper, increasing `wrist_flex` lowers it.
  Elbow -35 deg from rest is the top of the frame in the side view; shoulder moves have not been used in
  collection yet.
- Sag is direction dependent: lifting against gravity the elbow stops about 23 ticks short, lowering it
  lands on target. Arrival is therefore "within `--tol`, or stopped within `--sag-tol`"; a joint that
  stops further away than that for `--stall-s` is a stall and ends the run.
- `shoulder_pan` and `shoulder_lift` only hold position so far.
- Side view against behind view, same 2-minute run and joints: mean pixel change inside the encoder's crop
  square is 11.8 (of 255) from the side and 4.8 from behind. From behind the elbow and wrist move along the
  line of sight and are mostly hidden. First clean side-view run: `local/data/arm/20261124_011609` on aorus
  (3,601 samples, 128 MB, nobody in view, gripper inside the square at every extreme).

### Do the encoder features carry the arm's joint state?

`local/arm_encode.py` encodes every frame of a run with frozen encoders (single frame, centre crop, 256 px)
and fits a ridge regression from features to the measured angle of each joint that moved. Evaluation is
blocked 5-fold cross-validation in time with a 1 s gap, so neighbouring frames never straddle the split.

```bash
local/remote/run_on_aorus.sh python -m local.arm_encode local/data/arm/20261124_011609
```

Run `20261124_011609` (side view, 3,601 frames, four joints moving +/-10 deg). R2 / mean absolute error in
degrees on held-out time; "pooled" is the mean over tokens, "grid" keeps a 4x4 spatial layout:

| Features | Encode speed | elbow_flex | wrist_flex | wrist_roll | gripper |
|---|---|---|---|---|---|
| joint range in the run | | 15.9 deg | 17.8 deg | 19.3 deg | 9.8 deg |
| training mean | | 0.00 / 3.46 | 0.00 / 4.14 | 0.00 / 4.54 | 0.00 / 2.07 |
| raw 32x32 grey pixels | | 1.00 / 0.21 | 0.98 / 0.44 | 0.85 / 1.38 | 0.09 / 1.75 |
| 2.1 ViT-B pooled | 237 fps | 0.98 / 0.40 | 0.96 / 0.73 | 0.86 / 1.44 | 0.59 / 1.23 |
| 2.1 ViT-B grid | | 0.99 / 0.23 | 0.98 / 0.42 | 0.94 / 0.90 | 0.77 / 0.89 |
| 2.1 ViT-L pooled | 112 fps | 0.99 / 0.33 | 0.97 / 0.64 | 0.92 / 1.14 | 0.66 / 1.09 |
| 2.1 ViT-L grid | | 0.99 / 0.22 | 0.99 / 0.37 | 0.95 / 0.86 | 0.81 / 0.82 |
| 2-AC ViT-g pooled | 44 fps | 0.98 / 0.39 | 0.95 / 0.77 | 0.86 / 1.49 | 0.40 / 1.49 |
| 2-AC ViT-g grid | | 0.99 / 0.28 | 0.97 / 0.56 | 0.91 / 1.21 | 0.57 / 1.26 |

Reading it:

- The joint state is linearly readable from the features: elbow and wrist flex to well under a degree.
- Raw pixels do just as well on elbow and wrist flex. With a fixed camera and background those two joints
  move large dark shapes, so this run cannot show an advantage for the encoder there.
- The encoders pull ahead on the subtle joints: gripper opening goes from R2 0.09 (pixels) to 0.81 (ViT-L
  grid), wrist roll from 0.85 to 0.95.
- Keeping spatial layout beats mean pooling for every model and joint. A predictor should work on patch
  tokens or a pooled grid, not one pooled vector.
- ViT-L is best, ViT-B is close behind at twice the speed, and the 1B-parameter 2-AC ViT-g encoder is the
  weakest of the three here, clearly so on the gripper.
- Limits: one scene, one session, one lighting. This says the state is present, not that it survives a
  change of background or light. That needs a second run under different conditions as the test set.

### Action-conditioned predictor on frozen features

`local/arm_predictor.py` trains a 7.7M-parameter transformer that takes the 4x4 feature grid of a frame,
the measured joint angles and an action (the measured joint-angle change over the next 0.5 s), and predicts
the feature grid 0.5 s later. Same idea as V-JEPA 2-AC, with joint-space actions and our own data.

```bash
local/remote/run_on_aorus.sh python -m local.arm_predictor local/data/arm/20261124_011609
local/remote/run_on_aorus.sh python -m local.arm_predictor local/data/arm/20261124_011609 \
    --model vjepa2_1_vit_large_384
```

Trained on the first 95 s of run `20261124_011609`, tested on the last 22 s (1 s gap). Training takes about
two minutes per model. Test results, 0.5 s step, average joint change per step about 1.1 deg:

| | ViT-B features | ViT-L features |
|---|---|---|
| Feature L1: copy (predict no change) | 0.277 | 0.215 |
| Feature L1: same network without actions | 0.271 | 0.215 |
| Feature L1: with actions | 0.241 | 0.184 |
| Decoded joint error: copy | 1.43 deg | 1.41 deg |
| Decoded joint error: without actions | 1.91 deg | 1.80 deg |
| Decoded joint error: with actions | 1.08 deg | 0.88 deg |
| Decoded joint error: real future features (probe floor) | 0.70 deg | 0.69 deg |
| True action ranked first of 64 (chance 1.6%) | 15.5% | 21.5% |
| Mean rank of the true action | 11.5 | 8.8 |
| Chosen action off by (random pick: 1.80 deg) | 0.79 deg | 0.69 deg |
| Rollout 2.5 s, decoded joint error (copy 2.9 deg) | 1.67 deg | 1.26 deg |

"Decoded joint error" reads joint angles out of the predicted features with a linear probe fitted on the
training frames and compares them with the real future angles.

Reading it:

- The predictor uses the action. Without it the network is no better than copying; with it, the decoded
  error on ViT-L falls from 1.41 to 0.88 deg against a floor of 0.69.
- It can rank actions, which is what planning needs: the action it prefers is off by 0.7 deg per joint
  where a random candidate is off by 1.8. Exact top-1 is low because many candidates differ by less than
  the noise.
- Errors grow slowly over a 2.5 s rollout and stay well under the copy baseline.
- ViT-L features are better than ViT-B on every measure here.
- Feature L1 moves little in absolute terms because most of it is per-frame sensor noise the predictor
  cannot and should not predict. Judge the model by the decoded and ranking rows.
- At a 0.2 s step the arm moves about 0.5 deg per step and the action signal is nearly lost in that noise
  (decoded error equal to copy), hence the 0.5 s default.
- Limits: 95 s of training data, four joints in one plane, one scene. The test is offline. Nothing has yet
  chosen actions on the real arm.

### Closed-loop image-goal reaching on the arm

`local/arm_plan.py` uses the predictor to drive the real arm to a goal image. Per episode the arm visits a
random pose, the camera records the goal image there, the arm moves to a different random pose, and then
for 14 steps of 0.5 s a CEM planner (256 samples, 3 iterations, horizon 2) picks the joint-delta action
whose predicted features land closest to the goal image's features. The planner gets images and its own
joint angles; the goal's joint angles are kept only for scoring.

```bash
local/remote/run_on_aorus.sh '~/projects/lerobot/.venv/bin/python' -m local.arm_plan local/data/arm/20261124_011609
local/remote/run_on_aorus.sh '~/projects/lerobot/.venv/bin/python' -m local.arm_plan local/data/arm/20261124_011609 \
    --policy random
```

It runs in the LeRobot environment, which needs `timm` added for the encoder
(`uv pip install --python ~/projects/lerobot/.venv/bin/python --no-deps timm==1.0.31`).

ViT-L features, predictor trained on 95 s of run `20261124_011609`. Mean absolute joint error to the goal,
start -> after 7 s:

| Policy | Episodes | All joints | elbow_flex | wrist_flex | wrist_roll | gripper | Got closer |
|---|---|---|---|---|---|---|---|
| Planning, seed 0 | 3 | 7.4 -> 2.9 deg | 7.4 -> 2.3 | 12.4 -> 1.1 | 6.0 -> 4.6 | 3.6 -> 3.8 | 3 of 3 |
| Planning, seed 1 | 6 | 6.2 -> 2.9 deg | 8.8 -> 1.6 | 5.8 -> 1.3 | 5.3 -> 5.9 | 4.8 -> 2.6 | 6 of 6 |
| Random actions, seed 1 | 6 | 6.5 -> 6.5 deg | 6.0 -> 6.9 | 7.7 -> 6.8 | 8.3 -> 7.5 | 3.8 -> 4.7 | 4 of 6 |

- Planning from images works on the real arm: error falls in every episode, mostly within the first 3 s,
  then holds at about 2.8 deg. Random actions of the same size leave it unchanged.
- Elbow and wrist flex converge to 1 to 2 deg. Wrist roll does not improve and the gripper is mixed: these
  are the joints whose effect on the image is small, so the feature distance barely rewards getting them
  right. Feature distance to the goal ends at about 0.19 against a noise floor of 0.08.
- The scene check (live features against training frames at the same pose) read 0.22 against 0.20 between
  training pairs, so the camera had not moved. The planner stops itself if that check exceeds 1.5x.
- Limits: 9 planning episodes, goals drawn from the same +/-10 deg box as the training data, same scene
  and lighting as training, four joints in one plane.

## Simulated SO-101 (MuJoCo)

The same arm in simulation, to get hours of data with all six joints moving at no risk to the hardware,
and later to add a quadrotor (MuJoCo Menagerie's `skydio_x2` and `bitcraze_crazyflie_2` are already
fetched next to the arm model). Runs in the LeRobot env on aorus and renders headless on the GPU (EGL).

Setup on aorus (done once):

```bash
uv pip install --python ~/projects/lerobot/.venv/bin/python mujoco     # 4.0.1
mkdir -p ~/projects/sim_assets && cd ~/projects/sim_assets
git clone --depth 1 --filter=blob:none --sparse https://github.com/TheRobotStudio/SO-ARM100.git
git -C SO-ARM100 sparse-checkout set Simulation/SO101
git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git
git -C mujoco_menagerie sparse-checkout set skydio_x2 bitcraze_crazyflie_2
```

- `sim_arm.py`: `SimArm` loads the maker's MJCF unchanged and adds a table, a lamp and a side camera
  framed like the real C270 (640x360, base on the right). The arm is recoloured black like the printed
  one. Joint angles use the same zero-at-range-middle convention and signs as `arm_collect.py`'s degrees,
  except the gripper, which is the jaw angle in degrees here.
- Servo stiffness is set to the maker's measured STS3215 value (kp 17.8 N m/rad, from
  `joints_properties.xml`); the MJCF's actuators ship at 998, which would make the arm unrealistically
  stiff. With 17.8 a loaded joint lags its goal by about 1 deg, close to the real arm's sag.
- `sim_collect.py`: the same random walk as `arm_collect.py`, writing the same `frames/` + `log.npz` +
  `meta.json` layout to `local/data/sim_arm/<ts>`, so `arm_encode.py` and `arm_predictor.py` run on it
  unchanged. All six joints move, +/-20 deg around the ready pose, targets rejected if the gripper would
  come within 3 cm of the table. 367 fps including rendering and JPEG writing: an hour of data in under
  3 minutes.

```bash
~/projects/lerobot/.venv/bin/python -m local.sim_collect --seconds 120
.venv/bin/python -m local.arm_encode local/data/sim_arm/<ts> --models vjepa2_1_vit_base_384 vjepa2_1_vit_large_384
```

Interactive viewer (`sim_view.py`): the same scene in MuJoCo's window, starting on the dataset camera's
side view. `--demo` runs the random walk; without it, the "Control" panel's sliders set each servo goal.
To open it on the desktop's own screen from the laptop (someone logged in there):

```bash
ssh aorus 'cd projects/vjepa2 && DISPLAY=:0 XAUTHORITY=/run/user/1000/gdm/Xauthority \
  nohup ~/projects/lerobot/.venv/bin/python -m local.sim_view --demo > /tmp/sim_view.log 2>&1 &'
```

First run (`20261124_132605`, 120 s simulated, 6 joints, R2 / MAE in deg, blocked 5-fold CV in time):

| Features | shoulder_pan | shoulder_lift | elbow_flex | wrist_flex | wrist_roll | gripper |
|---|---|---|---|---|---|---|
| joint range | 39.9 | 40.3 | 39.5 | 39.0 | 37.8 | 19.9 |
| raw pixels 32x32 | 0.76 / 5.1 | 0.76 / 5.6 | -1.67 / 6.2 | -0.32 / 8.9 | 0.25 / 6.8 | 0.13 / 4.6 |
| 2.1 ViT-B grid4x4 | 0.99 / 0.7 | 0.98 / 1.2 | 0.95 / 1.5 | 0.63 / 4.4 | 0.50 / 4.9 | 0.72 / 2.4 |
| 2.1 ViT-L grid4x4 | 0.99 / 0.7 | 0.99 / 1.1 | 0.96 / 1.3 | 0.64 / 4.0 | 0.63 / 4.4 | 0.70 / 2.5 |

- With six joints moving at once, raw pixels fail on everything past the shoulder, while encoder features
  still read the shoulder and elbow almost exactly. Compare the real run, where four joints moved and raw
  pixels got elbow and wrist flex to R2 0.99.
- Wrist flex, wrist roll and gripper are the weak joints: small parts, all black on a dark arm, and their
  effect on the image depends on every joint above them. More data, a larger crop around the gripper, or
  a contrasting gripper colour are the obvious levers; the simulator makes each a one-line test.
### One hour of simulated data: does more data help?

`hour_seed1`: 3600 s simulated (108,000 frames, 3,455 targets, all six joints, +/-20 deg) in 7.6 min,
ViT-L encoding 17 min. For long runs `arm_encode.py` probes on at most 12,000 evenly spaced frames and
`arm_predictor.py` keeps features on the GPU in fp16 (see their `--probe-max`, `--max-test`).

```bash
~/projects/lerobot/.venv/bin/python -m local.sim_collect --seconds 3600 --seed 1 --out local/data/sim_arm/hour_seed1
.venv/bin/python -m local.arm_encode local/data/sim_arm/hour_seed1 --models vjepa2_1_vit_large_384
.venv/bin/python -m local.arm_predictor local/data/sim_arm/hour_seed1 --model vjepa2_1_vit_large_384 --steps 6000 \
    --train-seconds 600        # omit for all of the training part
```

Probe on the hour (R2 / MAE deg; compare the 2-minute table above):

| Features | shoulder_pan | shoulder_lift | elbow_flex | wrist_flex | wrist_roll | gripper |
|---|---|---|---|---|---|---|
| raw pixels 32x32 | 0.99 / 0.4 | 0.99 / 0.5 | 0.94 / 0.7 | 0.65 / 3.3 | -0.02 / 6.9 | -0.21 / 5.5 |
| 2.1 ViT-L pooled | 1.00 / 0.5 | 1.00 / 0.6 | 0.99 / 0.7 | 0.97 / 1.6 | 0.91 / 3.1 | 0.79 / 2.1 |
| 2.1 ViT-L grid4x4 | 1.00 / 0.2 | 1.00 / 0.2 | 1.00 / 0.2 | 1.00 / 0.7 | 0.96 / 1.9 | 0.91 / 1.3 |

Predictor (ViT-L grid, 0.5 s step, 6,000 steps each), trained on the first N seconds; every row is scored
on the same last 12 minutes (4,000 test starts):

| Training data | Decoded joints, mean | True action top-1 of 64 | Chosen action off by | 2.5 s rollout |
|---|---|---|---|---|
| copy (no change) | 1.92 deg | - | - | 4.69 deg |
| 2 min | 2.14 deg | 14.8% | 1.21 deg | 4.04 deg |
| 10 min | 1.20 deg | 50.5% | 0.57 deg | 2.21 deg |
| 48 min | **0.97 deg** | **79.5%** | **0.19 deg** | **1.61 deg** |
| floor (decode the real future) | 0.85 deg | - | - | - |

- With six joints moving, 2 minutes of data is not enough: the predictor is worse than predicting no
  change (2.14 vs 1.92 deg) and overfits (its training loss is the lowest of the three). The real-arm run
  got away with 2 minutes because only four joints moved, in one plane.
- 10 minutes already beats copy by a wide margin; the full 48 minutes gets within 0.12 deg of the floor,
  ranks the true action first 80% of the time (chance 1.6%), and a 2.5 s open-loop rollout stays at
  1.6 deg where copy drifts to 4.7.
- Per joint at 48 min (deg, predicted / floor): shoulder_pan 0.27 / 0.20, shoulder_lift 0.32 / 0.23,
  elbow 0.38 / 0.27, wrist_flex 0.96 / 0.78, wrist_roll 2.39 / 2.26, gripper 1.51 / 1.33. What remains is
  mostly the features' own limit on the wrist roll and gripper, not the predictor.
- More data also fixed the probe: wrist_flex R2 0.64 -> 1.00, gripper 0.70 -> 0.91.

### Closed-loop image-goal reaching in simulation

`sim_plan.py` runs the real-arm planning loop on the simulated arm: the CEM planner and the frame encoder
are shared with `arm_plan.py` (`cem_plan`, `encode_frames`; the extracted CEM gives bit-identical actions
to the old inline version). Each episode puts the arm at a random goal pose in the training box (all six
joints, +/-20 deg) and records the goal image, then at a start pose at least 6 deg away on average, and
gives it 20 steps of 0.5 s. Episodes depend only on `--seed` (100, unseen in training), so every row below
faces the same 20 episodes.

```bash
~/projects/lerobot/.venv/bin/python -m local.sim_plan local/data/sim_arm/hour_seed1 [--pred train600s] [--policy random]
```

Add `--view` to watch it in MuJoCo's viewer at real-time speed. The goal pose is drawn as a see-through
green ghost arm, and the top-left text shows the episode, step and tip distance. The ghost lives only in the
viewer's extra scene (`user_scn`); the planner's camera renders its own scene, so it never sees the ghost
and the results are those of a plain run. Closing the window stops the run and reports the finished
episodes. To show it on the desktop's screen from the laptop, launch it over SSH with
`DISPLAY=:0 XAUTHORITY=/run/user/1000/gdm/Xauthority`, as for `sim_view.py`.

| Policy | Joint error, mean | Gripper tip | Ends within 2 cm | Got closer |
|---|---|---|---|---|
| Planning, predictor on 48 min | 11.1 -> **3.2 deg** | 9.7 -> **2.1 cm** | 11 of 20 | 20 of 20 |
| Planning, predictor on 10 min | 11.1 -> 3.3 deg | 9.7 -> 2.0 cm | 10 of 20 | 20 of 20 |
| Planning, predictor on 2 min | 11.1 -> 4.6 deg | 9.7 -> 2.5 cm | 6 of 20 | 20 of 20 |
| Random actions, same size | 11.1 -> 12.7 deg | 9.7 -> 13.2 cm | 0 of 20 | 7 of 20 |

Per joint, 48-min predictor (deg): shoulder_pan 9.8 -> 0.6, shoulder_lift 12.3 -> 1.4, elbow 14.6 -> 2.2,
wrist_flex 9.3 -> 1.9, wrist_roll 14.7 -> 9.7, gripper 5.9 -> 3.2. Planning takes about 90 ms per step.

- Image-goal planning works with all six joints: every episode gets closer, and the gripper tip goes from
  9.7 cm to about 2 cm within 3 s (6 steps), then holds. Random actions of the same size drift away.
- Shoulder, elbow and wrist flex converge to 0.6 to 2 deg. Wrist roll barely improves and the gripper
  halves: same pattern as on the real arm, and the same cause. Those joints change the image little, so
  the feature distance hardly rewards getting them right (the probe's floor on them is the highest too).
- Predictor quality matters, up to a point: the 2-minute predictor still plans (it is worse than copy
  at one-step prediction, yet ranking actions only needs their order right) but ends 40% further away;
  10 and 48 minutes plan equally well, although 48 predicts better. Planning, at this horizon and goal
  distance, is limited by the energy (feature L1 to the goal image), not the predictor.
- Next levers, each cheap to test here: a goal energy that weights the gripper region (a finer token grid
  or a crop), a longer horizon, and goals outside the training box.

- Sim images are not real images: features and predictors trained here will not transfer to the C270
  view as they are. Use the simulator to develop methods and answer scaling questions, then confirm on the
  real arm.

## Simulated Crazyflie 2 (MuJoCo)

`sim_drone.py` flies MuJoCo Menagerie's `bitcraze_crazyflie_2` (27 g, commit eea4c74) in a 6 x 6 m room
with checkered floor and walls, coloured pillars and boxes, so an onboard camera has something to see.

```bash
~/projects/lerobot/.venv/bin/python -m local.sim_drone [--video]     # flight test: numbers, views.jpg, flight.mp4
ssh aorus 'cd projects/vjepa2 && DISPLAY=:0 XAUTHORITY=/run/user/1000/gdm/Xauthority nohup \
    ~/projects/lerobot/.venv/bin/python -m local.sim_drone --view --loop > /tmp/sim_drone_view.log 2>&1 &'
```

Changes to the Menagerie model, made when the scene is built (the asset files are untouched):

- **Rotors instead of body torques.** Menagerie drives the body with one thrust and three torque
  actuators whose torque tops out at 1e-5 N m: 0.4 rad/s^2 of roll acceleration, about 1/1000 of the real
  drone. They are replaced by four rotors at the tips of the 92 mm (motor to motor) X frame, 0.15 N each
  (thrust-to-weight 2.3), with a drag torque of 0.006 N m per N of thrust, alternating in sign, and a
  30 ms first-order spin-up lag. A rotor's command is its thrust in newtons.
- **Onboard camera** `fpv`: 2 cm ahead of the centre, looking forward, pitched 15 deg down, 70 deg vertical
  field of view. `track` (Menagerie's chase camera, 1 m behind, follows position not heading) and a fixed
  `corner` camera are also available.

Control runs every physics step (500 Hz): a position PD loop with velocity feed-forward and a 30 deg lean
limit gives the desired thrust vector, a geometric attitude loop (Lee, Leok and McClamroch 2010) gives body
torques, and the mixer solves the four rotor thrusts. A learned planner will later set this controller's
targets, not the rotor thrusts: a quadrotor does not stay upright by itself, so a fast loop must run
underneath whatever the world model decides.

Test flight (27 s of simulated flight in 2 s of wall time, rendering two cameras at 30 fps): take off to 1 m,
fly a 1 m square at 0.5 m/s, turn 90 deg, take a 0.05 N sideways shove for 0.1 s, land.

| Leg | Tracking error, mean / max | Max tilt | Rotor thrusts |
|---|---|---|---|
| Take off and hover | 2.4 / 9.4 cm | 0 deg | 0.049-0.085 N |
| Each side of the square | 5.4 / 10.6 cm | 14 deg | 0.057-0.076 N |
| Turn 90 deg | 0.4 / 3.4 cm | 1 deg | 0.060-0.072 N |
| Shove | 1.5 / 5.5 cm | 7 deg | 0.062-0.071 N |
| Land | 2.2 / 7.6 cm | 0 deg | 0.049-0.083 N |

Hover holds position to 0.3 mm RMS, and the hover thrust per rotor (0.0662 N) equals a quarter of the
weight. The turn ends at 90.0 deg, and the drone settles back to its 1.2 cm resting height. The 5 cm error
while flying the square is lag behind the moving target (PD only, no integral term or acceleration
feed-forward), and the 14 deg lean comes at the start and end of each leg, where the target's velocity
steps between 0 and 0.5 m/s. Rotors never get near 0 or the 0.15 N limit, so there is headroom for faster
flight.

## Suggested next steps for the drone world-model plan

1. Wire `load_model("vjepa2_ac_vit_giant")` into an image-goal MPC loop (the energy
   landscape notebook in `notebooks/` is the reference) and measure planning latency per
   step on MPS. ViT-g is ~1B params, ~2 GB in bf16, so it fits.
2. Fine-tune / post-train on the laptop only with ViT-S/B encoders and short clips; keep
   the ViT-L and ViT-g runs for the CUDA box.
3. When the GPU arrives nothing in `local/` needs to change: `pick_device()` prefers CUDA
   and the trainer patches keep the upstream CUDA/NCCL path.
