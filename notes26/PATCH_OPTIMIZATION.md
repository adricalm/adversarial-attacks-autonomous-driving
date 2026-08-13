# Patch optimization walkthrough

Plain-language order of the scripts under `scripts/patch_optimization/`.  
Run from the glue-repo root (`~/summer26`). Use the DSGN venv for anything that loads the network:

```bash
# once
bash scripts/dsgn/dsgn_setup_venv.sh
VENV=external/DSGN_custom/.venv/bin/python
```

Supported detector checkpoint: **`dsgn/checkpoints/kitti/dsgn_12g_b/finetune_48.tar`** with the half-res AWSIM config beside it. See [`DSGN_AWSIM_FINDINGS.md`](DSGN_AWSIM_FINDINGS.md).

---

## Pipeline (what each step produces)

```text
recordings (KITTI layout)
    → prepare_recording_datasets.py   # optional; calib / splits after record_kitti_dataset.sh
    → DSGN inference                  # scripts/dsgn/dsgn_run_inference.sh → detections/*.txt
    → localize_patches.py             # rear-face boxes → patches_localized.csv
    → build_combined_dataset.py       # merge several recordings → patch_train/{dataset,train.csv,val.csv}
    → optimize_patch.py               # universal patch → patch_train/face050/patch_best.png
    → apply_face_patch.py             # paste patch onto test stereo → patched KITTI tree
    → eval_patch.py / attack_stats.py # offline metrics
    → (optional) dsgn_offline replay  # Autoware behavior — see DSGN_OFFLINE_RUNBOOK.md
```

---

## 1. Localize rear faces

Needs clean DSGN detections (KITTI `.txt`) and matching `calib/`.

```bash
python3 scripts/patch_optimization/localize_patches.py \
  --detections dsgn/detections/adria/train_recordings_clean/train_frontal1 \
  --calib dsgn/datasets/recordings/train_frontal1/calib \
  --output dsgn/datasets/recordings/train_frontal1/patches_localized.csv \
  --box-convention kitti --selection closest
```

Repeat per recording you care about. `--selection closest` = one row per frame (what apply/optimize expect for the lead car).

## 2. Build a combined train/val view

`optimize_patch.py` wants one image root + `train.csv` / `val.csv`.

```bash
python3 scripts/patch_optimization/build_combined_dataset.py \
  --out dsgn/datasets/adria/patch_train \
  --train-source train_frontal1 dsgn/datasets/recordings/train_frontal1 \
    dsgn/datasets/recordings/train_frontal1/patches_localized.csv \
  --val-source train_frontal5 dsgn/datasets/recordings/train_frontal5 \
    dsgn/datasets/recordings/train_frontal5/patches_localized.csv
```

Output: `patch_train/dataset/` (symlinked KITTI tree) plus `train.csv` / `val.csv`.

## 3. Optimize the universal patch

```bash
$VENV scripts/patch_optimization/optimize_patch.py \
  --images dsgn/datasets/adria/patch_train/dataset \
  --csv dsgn/datasets/adria/patch_train/train.csv \
  --val-csv dsgn/datasets/adria/patch_train/val.csv \
  --out dsgn/datasets/adria/patch_train/face050 \
  --area-frac 0.50 --epochs 20
```

Look for `patch_best.png` / `.pt` under the `--out` directory. Defaults assume the `finetune_48` checkpoint paths; override with `--cfg` / `--loadmodel` if needed.

## 4. Apply to a held-out recording

```bash
python3 scripts/patch_optimization/localize_patches.py \
  --detections dsgn/detections/adria/test_recordings_clean/test_frontal1 \
  --calib dsgn/datasets/recordings/test_frontal1/calib \
  --output dsgn/datasets/recordings/test_frontal1/patches_localized.csv \
  --box-convention kitti --selection closest

python3 scripts/patch_optimization/apply_face_patch.py \
  --source dsgn/datasets/recordings/test_frontal1 \
  --csv dsgn/datasets/recordings/test_frontal1/patches_localized.csv \
  --patch dsgn/datasets/adria/patch_train/face050/patch_best.png \
  --out dsgn/datasets/adria/patch_test/face050/test_frontal1 \
  --area-frac 0.50
```

Use the **same `--area-frac`** as optimization.

## 5. Evaluate offline

Against the optimizer’s train/val CSVs:

```bash
$VENV scripts/patch_optimization/eval_patch.py \
  --run dsgn/datasets/adria/patch_train/face050 \
  --images dsgn/datasets/adria/patch_train/dataset \
  --csv dsgn/datasets/adria/patch_train/train.csv \
  --val-csv dsgn/datasets/adria/patch_train/val.csv
```

Or compare full detection folders (clean vs patched re-inference) with `attack_stats.py` — see its docstring.

## 6. Optional: Autoware replay

Re-run DSGN on the patched images, copy `.txt` dumps into `src/dsgn_offline/resource/…`, then follow [`DSGN_OFFLINE_RUNBOOK.md`](DSGN_OFFLINE_RUNBOOK.md).

---

## Folder cheat sheet

| Path | Meaning |
|------|---------|
| `dsgn/datasets/recordings/*` | Raw stereo clips (KITTI layout) |
| `dsgn/detections/adria/*` | Precomputed clean / patched detection dumps |
| `dsgn/datasets/adria/patch_train/` | Combined optimize view + run outputs (`face050/`, …) |
| `dsgn/datasets/adria/patch_test/` | Patched test trees ready for re-inference |

Script flags and edge cases: read the docstring at the top of each file — that is the source of truth.
