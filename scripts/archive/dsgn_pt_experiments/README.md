# Archived DSGN PyTorch experiments (host)

These scripts were moved out of `scripts/` on 2026-07-08 so the active toolbox only contains
what works on the lab L40S (PyTorch 2.6) or Docker on an old GPU.

**Do not run these on the L40S for day-to-day work.**

## What to use instead

| Goal | Use |
|------|-----|
| Run inference on L40S (wrong detections, but compiles) | `scripts/dsgn_setup_venv.sh` + `scripts/dsgn_run_inference.sh` |
| Faithful PT 1.3 inference | `scripts/dsgn_build_docker.sh` + `scripts/dsgn_run_inference_docker.sh` on V100/Titan/Pascal |
| Autoware experiments without re-inference | Replay Arka baseline — `notes/DSGN_OFFLINE_RUNBOOK.md` |

## Archived files

| Script | PyTorch | Why archived |
|--------|---------|--------------|
| `dsgn_setup_pt110.sh` | 1.10 + cu113 | Failed experiment — 34 false dets on frame 000010 vs baseline 0 |
| `dsgn_run_inference_pt110.sh` | 1.10 | Companion to setup above |
| `dsgn_setup_pt13.sh` | 1.3 CPU | Host PT 1.3 cannot run on L40S (CUDA ops + sm_89). One-time use: creates `finetune_60_legacy.tar` if missing |
| `dsgn_run_inference_pt13.sh` | 1.3 | Host inference — non-functional on L40S |

## One-time: legacy checkpoint conversion

`finetune_60_legacy.tar` is required for Docker PT 1.3 inference (PyTorch 1.3 cannot read zip-serialized checkpoints).

**Status on this machine:** `finetune_60_legacy.tar` is **not present** yet under `dsgn/checkpoints/arka/dsgn_12g_awsim_remote_downsample/`.

To create it (needs PT 2.6 venv from `scripts/dsgn_setup_venv.sh`):

```bash
bash ~/summer26/scripts/archive/dsgn_pt_experiments/dsgn_setup_pt13.sh
```

The setup script converts `finetune_60.tar` → `finetune_60_legacy.tar` at the end using the PT 2.6 venv. You do **not** need to run PT 1.3 inference on the L40S afterward.

## Running archived scripts (reference only)

```bash
# PT 1.10 — deprecated, wrong detections
bash ~/summer26/scripts/archive/dsgn_pt_experiments/dsgn_setup_pt110.sh
bash ~/summer26/scripts/archive/dsgn_pt_experiments/dsgn_run_inference_pt110.sh

# PT 1.3 host — only useful on old GPU or for legacy ckpt conversion
bash ~/summer26/scripts/archive/dsgn_pt_experiments/dsgn_setup_pt13.sh
bash ~/summer26/scripts/archive/dsgn_pt_experiments/dsgn_run_inference_pt13.sh
```
