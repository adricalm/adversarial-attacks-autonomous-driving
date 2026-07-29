# DSGN on AWSIM — findings (Jul 16 2026)

Concise log of what we learned fixing stereo 3D detection on AWSIM.  
Related: [`DSGN_PYTORCH_VERSIONING.md`](DSGN_PYTORCH_VERSIONING.md), [`DSGN_OFFLINE_RUNBOOK.md`](DSGN_OFFLINE_RUNBOOK.md), [`official_vs_arka_dsgn_repo.md`](official_vs_arka_dsgn_repo.md).

---

## Timeline (short)

1. **Hypothesis: PyTorch version** — Official DSGN authors reported a large eval drop when moving past PT 1.3. We blamed PT 2.6 on the L40S for “nonsense” detections (Arka `finetune_60` and early AWSIM runs). That **is still true for Arka’s PT 1.3 AWSIM checkpoint** on this host, but it was **not** the main reason official KITTI `finetune_48` looked broken on AWSIM.

2. **Detour: DSGN++ (DSGN2)** — Tried for newer deps / GPU-friendly stack. Weak on AWSIM (KITTI-trained). On a KITTI subset it ran but missed many cars. Original DSGN looked better on the same subset (light check, not a full DSGN2 bake-off). Stayed on **`external/DSGN_custom`** for continuity with Autoware / `dsgn_offline`. Scripts under `scripts/dsgn2_*` are **kept for now**; removing them is future cleanup.

3. **Real AWSIM fix: geometry + config** — Once half-res config matched the loader, **official `finetune_48` produced coherent boxes on AWSIM without finetuning**. Remaining pain: **false positives**.

4. **Label convention** — Arka’s AWSIM `label_2` is **not** true KITTI. Early finetunes trained against wrong targets. We documented the remap, converted labels under adria, and started **`det_head` adapt from `finetune_48`**.

---

## Critical: half-resolution geometry

The AWSIM loader downsamples images by **0.5** and scales calibration the same way (`calib.scale(0.5)` in `KITTILoader_dataset3d*.py`). The cost volume / `grid_sample` path must use the **same pixel grid**.

In `external/DSGN_custom/configs/config_car_12g_awsim.py`:

| Setting | Was (wrong for 0.5× loader) | Now (correct) |
|---------|------------------------------|---------------|
| `input_size` | `[1080, 1920]` | **`[540, 960]`** |
| `output_size` | `[270, 480]` | **`[135, 240]`** |

`CV_X_MAX` / `CV_Y_MAX` are derived from `input_size` — they follow automatically.

Also useful on AWSIM:

- `RPN3D.NMS_THRESH = 0.25` (was 0.6-style / commented)
- At viz / eval time, raise score floor (`--min-score ~0.3`) — training defaults are low and show many FPs

**Depth GT** under `training/depth/` is **metres** (`depth_disp=True`). Resize the map spatially; do **not** multiply depth values by 0.5.

With the size fix, **KITTI `dsgn/checkpoints/kitti/dsgn_12g_b/finetune_48.tar` + PT 2.6 on L40S is a valid AWSIM inference path**. Viz those detections with `--box-convention kitti`.

---

## Label conventions (AWSIM / Arka vs KITTI)

Same 15-field `.txt` layout; **different meaning** for dims and `y` (and how `ry = 0` aligns length).

| | AWSIM / Arka plotter | KITTI / DSGN `Object3d` |
|--|----------------------|-------------------------|
| fields 8,9,10 | length, height, width | height, width, length |
| `y` | box **center** | box **bottom-center** |
| `ry ≈ 0` | length along camera **Z** | length along camera **X** |

**Convert AWSIM → KITTI** (validated; used for training):

```text
h, w, l = f9, f10, f8
y       = y_awsim + h/2
ry      = wrap(ry_awsim - π/2)
alpha   = wrap(ry - atan2(x, z))
# type, trunc, occ, 2D bbox, x, z unchanged
```

Script: `scripts/dsgn_transform_label.py`  
### Dataset layout on this machine (Jul 2026)

| Path | Role |
|------|------|
| `dsgn/datasets/adria/training_kitti_labels/` | **Only training set** — images/calib/depth/velodyne + KITTI-converted `label_2/` |
| `dsgn/datasets/arka/dsgn_awsim/testing/` | Arka test split (kept) |
| `dsgn/datasets/arka/dsgn_awsim/testing_offline/` | Offline / Autoware replay images (kept) |
| `dsgn/datasets/arka/dsgn_awsim/{train,val,trainval,test}*.txt` | Split lists (kept; `trainval.txt` still indexes training frame IDs) |

**Arka’s `training/` was deleted** to free disk (~20G). It is no longer on this host. All finetune `DATA_PATH` must point at `training_kitti_labels`. Do not assume `dsgn/datasets/arka/dsgn_awsim/training` exists.

Viz: `scripts/visualize_dsgn_detections.py --box-convention {awsim,kitti}`.

DSGN training **always** expects KITTI via `Object3d`. Raw Arka-format labels → wrong targets → aggressive finetune destroys geometry.

---

## Finetuning strategy (current)

| Approach | Status |
|----------|--------|
| Ship **`finetune_48`** + half-res config + score thresh | Still the safe default until a full adapt beats it |
| **`MODE=det_head`** on converted labels | **Tried** — see [`DSGN_ADAPT_DET_HEAD_KITTI_LABELS.md`](DSGN_ADAPT_DET_HEAD_KITTI_LABELS.md); debug viz **not great** |
| **Full-model** on converted labels (Arka-style) | Next: train entire net from `finetune_48` via `dsgn_train.sh` |
| Arka **`finetune_60`** on this host | Optional A/B via precomputed dumps; PT 2.6 re-inference unfaithful |

Always: converted `label_2` under `training_kitti_labels`, `FORCE_TARGETS=1` when labels change, viz with **`--box-convention kitti`**.

### Full-model train (host) — Arka-style schedule

Uses `scripts/dsgn_train.sh` (full net, lr≈1e-3, 60 ep). Same converted data as det_head:

```bash
FORCE_TARGETS=1 EPOCHS=60 \
DATA_PATH=~/summer26/dsgn/datasets/adria/training_kitti_labels \
SPLIT_FILE=~/summer26/dsgn/datasets/arka/dsgn_awsim/trainval.txt \
LOADMODEL=~/summer26/dsgn/checkpoints/kitti/dsgn_12g_b/finetune_48.tar \
SAVEMODEL=~/summer26/dsgn/checkpoints/adria/kitti48_awsim_full_kitti_labels \
bash ~/summer26/scripts/dsgn_train.sh
```

(`dsgn_finetune_awsim.sh` with `MODE=gentle|det_head` no longer exists; historical det_head recipe is in [`DSGN_ADAPT_DET_HEAD_KITTI_LABELS.md`](DSGN_ADAPT_DET_HEAD_KITTI_LABELS.md).)

---

## Takeaways

1. PT version ≠ the whole AWSIM story; **config/loader pixel alignment** was the breakthrough for `finetune_48`.  
2. **Label convention** explains bad AWSIM finetunes and “tower” viz bugs.  
3. Prefer **convert labels → gentle/det_head from 48** over trusting Arka `finetune_60` on L40S.  
4. Train from `adria/training_kitti_labels` only; Arka keeps `testing` / `testing_offline` / split lists (Arka `training/` removed for disk).
