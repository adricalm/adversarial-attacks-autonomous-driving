# Handoff

Short orientation for setting up this workspace. Detailed commands live in [`README.md`](README.md) and [`notes26/`](notes26/).

## Repos (code)

| Repo | Role | Clone into |
|------|------|------------|
| [adricalm/adversarial-attacks-autonomous-driving](https://github.com/adricalm/adversarial-attacks-autonomous-driving) | Glue repo (scripts, Autoware overrides, stereo mod, patch pipeline) | `~/summer26` (this repo) |
| [adricalm/DSGN_custom](https://github.com/adricalm/DSGN_custom) | DSGN train / inference | `~/summer26/external/DSGN_custom` |
| [adricalm/dsgn_offline](https://github.com/adricalm/dsgn_offline) | ROS 2 node that replays KITTI detections into Autoware | `~/summer26/src/dsgn_offline` |

After cloning the glue repo:

```bash
cd ~/summer26
git clone git@github.com:adricalm/DSGN_custom.git external/DSGN_custom
git clone git@github.com:adricalm/dsgn_offline.git src/dsgn_offline
```

Use branch **`main`** on the glue repo and on `dsgn_offline`. Use **`master`** on `DSGN_custom`. Ignore branch `dsgn2` on the glue repo.

Both forks have `origin` = your/adria fork and should keep `upstream` = [DF-Autoware-AWSIM](https://github.com/orgs/DF-Autoware-AWSIM/repositories) if you need Arka’s updates.

## Data link (not in git)

**Shared drive / folder:** _TODO: drive link to data here_

Copy the archive contents so the tree matches the table below (paths relative to `~/summer26`).

| Bundle / top-level folder | Put it here | Needed for |
|---------------------------|-------------|------------|
| `awsim/` (v1.6.1 zip or `extracted/` + optional `modded/`) | `data/awsim/` | Simulator |
| `nishishinjuku_autoware_map/` (include `.pcd` files) | `data/maps/nishishinjuku_autoware_map/` | Autoware localization |
| `ml_models/` | `data/autoware_data/ml_models/` | Autoware perception models |
| `dsgn/datasets/` | `dsgn/datasets/` | DSGN + patch work |
| `dsgn/checkpoints/` (at least `kitti/dsgn_12g_b/finetune_48.tar` + config) | `dsgn/checkpoints/` | DSGN inference / patch optimize |
| `dsgn/detections/` | `dsgn/detections/` | Patch localize / attack stats (optional if you re-infer) |

Rough sizes on the lab machine: AWSIM ~2.5G, maps ~0.2G, ml_models ~3.6G, datasets ~18G, checkpoints ~0.1G, detections ~0.05G.

You do **not** need to copy Docker containers. Pull the Autoware image when needed:

```bash
docker pull ghcr.io/autowarefoundation/autoware:universe-cuda-humble
```

Skip copying `logs/`, AWSIM `core.*` dumps, and old disposable outputs.

### Quick presence check

```bash
export ROOT="${ROOT:-$HOME/summer26}"
test -d "$ROOT/data/maps/nishishinjuku_autoware_map"
test -d "$ROOT/data/autoware_data/ml_models"
test -x "$ROOT/data/awsim/extracted/awsim_labs_v1.6.1/awsim_labs.x86_64"
test -d "$ROOT/external/DSGN_custom"
test -d "$ROOT/src/dsgn_offline"
test -f "$ROOT/dsgn/checkpoints/kitti/dsgn_12g_b/finetune_48.tar"
```

## Lab assumptions

| Item | Value |
|------|-------|
| Host | Lab Ubuntu machine with NVIDIA GPU + Docker GPU access |
| Project root | `~/summer26` (or set `ROOT`) |
| Docker image | `ghcr.io/autowarefoundation/autoware:universe-cuda-humble` |
| AWSIM | Labs **v1.6.1** under `data/awsim/extracted/` |
| ROS domain | `ROS_DOMAIN_ID=26` (hard-coded in several scripts) |
| GUI | xrdp / X11 for the AWSIM window |

## What to run first

1. **Stack:** [`notes26/autoware-awsim-startup.md`](notes26/autoware-awsim-startup.md) — Autoware + AWSIM + drive a saved route.
2. **DSGN inference:** `scripts/dsgn/dsgn_setup_venv.sh` once, then `scripts/dsgn/dsgn_run_inference.sh`. Prefer checkpoint **`finetune_48`** (PyTorch 2.6). Treat Arka’s `finetune_60` as legacy / unreliable on this host.
3. **Patches:** [`notes26/PATCH_OPTIMIZATION.md`](notes26/PATCH_OPTIMIZATION.md).

Optional later: stereo mod + recording ([`notes26/AWSIM_STEREO_CAMERA.md`](notes26/AWSIM_STEREO_CAMERA.md)), offline Autoware replay ([`notes26/DSGN_OFFLINE_RUNBOOK.md`](notes26/DSGN_OFFLINE_RUNBOOK.md)). Note: `dsgn_offline` currently defaults its detection folder to `resource/testing_offline_no_finetune_patched_optimized` — override the ROS parameter if you want a different dump.

## Known caveats (do not re-learn the hard way)

- AWSIM must run **inside** the Autoware Docker image, not on the host.
- Always scrub `ROS_DISTRO` / ament paths for AWSIM (the launch scripts do this).
- Official KITTI **`finetune_48`** is the supported DSGN checkpoint here. Arka **`finetune_60`** was trained on PT 1.3 and does not re-infer faithfully on PT 2.6 / L40S — use Arka’s precomputed detection dumps if you need that baseline.
- Joint Autoware + stereo (modded) AWSIM under heavy GPU load was not fully validated; recording with modded AWSIM alone works.
- Ask before stopping other users’ Docker containers on the shared server.
