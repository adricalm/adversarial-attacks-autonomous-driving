# Experiment: det_head adapt on KITTI-converted AWSIM labels (Jul 2026)

**Outcome:** qualitative results on debug frames were **not great** (kept for record before full-model finetune).

Related: [`DSGN_AWSIM_FINDINGS.md`](DSGN_AWSIM_FINDINGS.md).

---

## Goal

Start from official KITTI `finetune_48` (coherent on AWSIM after half-res config), adapt only the **detection head** on AWSIM labels converted to true KITTI convention.

## How it was trained

**Script (removed):** `scripts/dsgn_finetune_awsim.sh` with `MODE=det_head` — freeze stereo `feature_extraction,dres0,classif1`; train RPN / bbox heads only.  
**Current train entrypoint:** `scripts/dsgn_train.sh` (full-model only; no `MODE=`).

| Setting | Value |
|---------|--------|
| Init | `dsgn/checkpoints/kitti/dsgn_12g_b/finetune_48.tar` |
| Config | `external/DSGN_custom/configs/config_car_12g_awsim.py` (`input_size=[540,960]`, `output_size=[135,240]`) |
| Data | `dsgn/datasets/adria/training_kitti_labels` (only training tree on disk; Arka `training/` removed later) |
| Split | `dsgn/datasets/arka/dsgn_awsim/trainval.txt` (282 frames; still under arka) |
| Labels | AWSIM→KITTI via `scripts/dsgn_transform_label.py` |
| LR | `BASE_LR=2e-4`, `MIN_LR=2e-5`, `LR_SCALE=40`, `--no_warmup` |
| Schedule | 60 epochs, save every 5, batch 1, `FORCE_TARGETS=1` |

Authoritative recipe (as run):  
`dsgn/checkpoints/adria/kitti48_awsim_adapt_det_head_kitti_labels/finetune_recipe.txt`

Historical command (script no longer in tree):

```bash
MODE=det_head EPOCHS=60 SAVE_EVERY=5 LR_SCALE=40 FORCE_TARGETS=1 \
DATA_PATH=~/summer26/dsgn/datasets/adria/training_kitti_labels \
SPLIT_FILE=~/summer26/dsgn/datasets/arka/dsgn_awsim/trainval.txt \
LOADMODEL=~/summer26/dsgn/checkpoints/kitti/dsgn_12g_b/finetune_48.tar \
SAVEMODEL=~/summer26/dsgn/checkpoints/adria/kitti48_awsim_adapt_det_head_kitti_labels \
bash ~/summer26/scripts/dsgn_finetune_awsim.sh   # removed — see finetune_recipe.txt
```

## Artifacts

### Checkpoints

`dsgn/checkpoints/adria/kitti48_awsim_adapt_det_head_kitti_labels/`

- `finetune_{5,10,...,60}.tar`
- `finetune_recipe.txt`, `training.log`, `tensorboard/`

### Detections (debug split: 10, 99, 105, 150, 180, 200)

`dsgn/detections/adria/adapt_kitti_labels_ep{5..60}/`

Inference via `scripts/dsgn_run_inference.sh` on `testing_offline` + `test_offline_debug.txt`.  
Viz: `--box-convention kitti`.

## Next

Full-model finetune (Arka-style: train entire net) on the same converted labels — see findings note / train command in chat or README update.
