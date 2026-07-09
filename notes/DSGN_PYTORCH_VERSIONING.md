# DSGN PyTorch versioning

Arka's `finetune_60.tar` was trained with **PyTorch 1.3.0** (`external/DSGN_custom/requirement.txt`).  
Running inference with PyTorch 2.6 produces garbage (46 false cars on frame 000010 vs 0 in the Arka baseline).

## Recommended environments (active scripts)

| Env | Script | PyTorch | GPU on L40S | Fidelity |
|-----|--------|---------|-------------|----------|
| **Broken (dev only)** | `scripts/dsgn_setup_venv.sh` | 2.6 + cu124 | Yes | Wrong detections — rebuild from `scripts/requirements_dsgn_pt26_frozen.txt` |
| **Docker (old GPU)** | `scripts/dsgn_build_docker.sh` | 1.3 + CUDA 10.1 | No on L40S/Ada | Correct on V100/Titan |

**Active inference on L40S:** `scripts/dsgn_run_inference.sh` (PT 2.6 — wrong detections).

**Faithful inference:** Docker on an old GPU (`scripts/dsgn_run_inference_docker.sh`), or replay Arka's precomputed outputs (`notes/DSGN_OFFLINE_RUNBOOK.md`).

## Archived host experiments

PT 1.3 and PT 1.10 **host** setup/inference scripts are in `scripts/archive/dsgn_pt_experiments/` — see that README for why. Do not use on the L40S for day-to-day work.

| Env | Archived script | Result on L40S |
|-----|-----------------|----------------|
| PT 1.3 host | `dsgn_setup_pt13.sh` / `dsgn_run_inference_pt13.sh` | Cannot run inference (CUDA-only ops; sm_89 unsupported). Setup can still create `finetune_60_legacy.tar` once. |
| PT 1.10 | `dsgn_setup_pt110.sh` / `dsgn_run_inference_pt110.sh` | Runs but wrong — 34 dets on 000010 vs baseline 0 |

## Faithful inference (Docker on old GPU)

PyTorch 1.3 + CUDA 10.1 does not support L40S (Ada/sm_89). Use a machine with **V100 / Titan / Pascal**:

```bash
# host — needs sudo
bash ~/summer26/scripts/dsgn_build_docker.sh
USE_GPU=1 bash ~/summer26/scripts/dsgn_run_inference_docker.sh
```

Docker expects `dsgn/checkpoints/arka/dsgn_12g_awsim_remote_downsample/finetune_60_legacy.tar`. If missing, run the archived setup once (see `scripts/archive/dsgn_pt_experiments/README.md`).

PT 2.6 csrc patches live in `scripts/patches/dsgn_csrc_pt26.patch` (applied by `dsgn_setup_venv.sh`).

## L40S host limitations

1. **Cannot run faithful PT 1.3 GPU inference** — architecture too new for CUDA 10.1 kernels.
2. **Cannot run CPU inference** — `build_cost_volume` and other custom ops are CUDA-only.
3. **PT 1.10 host — archived failed experiment.** Still gives false positives (34 cars on 000010 vs baseline 0).

## Code fixes applied (all PT versions)

- `tools/test_no_eval.py` — CPU `.cuda()` shim, explicit device handling, `--devices cpu` support.
- `dsgn/models/stereonet.py` — pass `align_corners=self.cfg.align_corners` to all `F.grid_sample` calls (PyTorch ≥1.3 default changed). Minor improvement on PT 2.6 (46→39 dets on 000010) but not sufficient alone.

## Validation checklist

After faithful re-inference:

| Frame | Baseline (`awsim_output_offline`) | Expect |
|-------|-----------------------------------|--------|
| 000010 | 0 detections | ~0 |
| 000099 | 2 detections | ~2 |
| 000105 | 2 detections | ~2 (patched image should differ) |

```bash
wc -l src/dsgn_offline/resource/awsim_output_offline/000010.txt
wc -l dsgn/detections/adria/<TAG>/000010.txt
```

## Merge + Autoware (unchanged)

```bash
python3 scripts/merge_dsgn_outputs.py \
  --baseline src/dsgn_offline/resource/awsim_output_offline \
  --patched  dsgn/detections/adria/<TAG> \
  --output   src/dsgn_offline/resource/awsim_output_patched_merged \
  --frames   100-135
```
