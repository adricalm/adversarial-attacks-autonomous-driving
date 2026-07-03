# DSGN PyTorch versioning

Arka's `finetune_60.tar` was trained with **PyTorch 1.3.0** (`external/DSGN_custom/requirement.txt`).  
Running inference with PyTorch 2.6 produces garbage (46 false cars on frame 000010 vs 0 in the Arka baseline).

## Recommended environments

| Env | Script | PyTorch | GPU on L40S | Fidelity |
|-----|--------|---------|-------------|----------|
| **Arka-compatible** | `scripts/dsgn_setup_pt13.sh` | 1.3.0 CPU | No (CUDA ops required) | Required for correct detections |
| **Broken (dev only)** | `scripts/dsgn_setup_venv.sh` | 2.6 + cu124 | Yes | Wrong detections |
| **Docker (old GPU)** | `scripts/dsgn_build_docker.sh` | 1.3 + CUDA 10.1 | No on L40S/Ada | Correct on V100/Titan |
| **Experimental** | `scripts/dsgn_setup_pt110.sh` | 1.13 + cu116 | Yes | Still wrong on 000010 (34 dets) |

## One-time setup (Arka-compatible)

```bash
bash ~/summer26/scripts/dsgn_setup_pt13.sh
```

This installs Miniconda under `~/summer26/.conda/miniconda3`, creates env `dsgn-pt13` (Python 3.7), installs PyTorch 1.3.0+cpu, builds DSGN against **original** csrc (not the PT 2.6 API patches), and creates `finetune_60_legacy.tar` (PyTorch 1.3 cannot read zip-serialized checkpoints).

PT 2.6 csrc patches are saved in `scripts/patches/dsgn_csrc_pt26.patch` and re-applied after the PT 1.3 build so the PT 2.6 venv keeps working.

## Run inference (PT 1.3)

```bash
# Full patched dataset
bash ~/summer26/scripts/dsgn_run_inference_pt13.sh

# Validate single frame (clean images)
DATA_PATH=~/summer26/data/arka/awsim/testing_offline \
  SPLIT_FILE=~/summer26/data/arka/awsim/test_offline_frame10.txt \
  TAG=_pt13_test \
  bash ~/summer26/scripts/dsgn_run_inference_pt13.sh
```

**GPU:** PyTorch 1.3 + CUDA 10.1 does not support L40S (Ada/sm_89). CUDA 1.3 wheels are also no longer hosted (403). Use a machine with **V100 / Titan / Pascal** and Docker:

```bash
# host — needs sudo
bash ~/summer26/scripts/dsgn_build_docker.sh
USE_GPU=1 bash ~/summer26/scripts/dsgn_run_inference_docker.sh
```

## L40S host limitations

1. **Cannot run faithful PT 1.3 GPU inference** — architecture too new for CUDA 10.1 kernels.
2. **Cannot run CPU inference** — `build_cost_volume` and other custom ops are CUDA-only.
3. **PT 1.13 on L40S** still gives false positives (34 cars on 000010 vs baseline 0).

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
wc -l models/arka/.../awsim_output_2<TAG>/000010.txt
```

## Merge + Autoware (unchanged)

```bash
python3 scripts/merge_dsgn_outputs.py \
  --baseline src/dsgn_offline/resource/awsim_output_offline \
  --patched  models/arka/dsgn_12g_awsim_remote_downsample/awsim_output_2<TAG> \
  --output   src/dsgn_offline/resource/awsim_output_patched_merged \
  --frames   100-135
```
