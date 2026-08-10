"""Offline evaluation of a universal adversarial patch against DSGN.

Runs the best patch (or any supplied .pt/.png) through every val frame
(and optionally every train frame) with the same geometry used during
optimize_patch.py. Reports:

  - Per-frame: split, depth, clean_score (from CSV), patched_max_s,
    suppressed (patched < score_thresh), post-NMS det count.
  - Summary table by depth bin.
  - Suppression rate: val, train, overall.

Results are saved as a CSV beside the patch and printed to stdout.

Usage
-----
    python scripts/patch_optimization/eval_patch.py \\
        --run  dsgn/datasets/adria/2.training_patch_optimization/optimize_logit_face050 \\
        --images dsgn/datasets/adria/training_kitti_labels \\
        --csv   dsgn/datasets/adria/2.training_patch_optimization/patches_localized.csv \\
        --cfg   dsgn/checkpoints/kitti/dsgn_12g_b/save_config_awsim.py \\
        --loadmodel dsgn/checkpoints/kitti/dsgn_12g_b/finetune_48.tar

    # Evaluate a specific checkpoint instead of patch_best:
        --patch dsgn/.../patch_best_epoch010.pt

    # Include training frames (check for overfitting):
        --splits train val
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

# Reuse the geometry and model-loading code from the optimizer.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from optimize_patch import (  # noqa: E402
    RunOpts,
    calib_for_frame,
    car_logits_in_radius,
    compute_locations_bev,
    count_nms_matched_cars,
    filter_visible_frames,
    load_cfg,
    load_csv,
    load_model,
    load_z_init,
    prepare_stereo_pair,
    read_image_01,
    split_frames,
)

SCORE_THRESH = 0.33
DEPTH_BINS = [0, 5, 10, 15, 20, 30, float("inf")]
DEPTH_LABELS = ["<5m", "5-10m", "10-15m", "15-20m", "20-30m", ">30m"]


def eval_frame(
    spec,
    patch_z: torch.Tensor,
    model,
    cfg,
    locations_bev: torch.Tensor,
    images_root: Path,
    device: torch.device,
    opts: RunOpts,
) -> dict:
    left_path = images_root / "image_2" / f"{spec.frame}.png"
    right_path = images_root / "image_3" / f"{spec.frame}.png"
    calib_path = images_root / "calib" / f"{spec.frame}.txt"

    left = read_image_01(left_path)
    right = read_image_01(right_path)
    calib, calib_r, f_u, baseline = calib_for_frame(calib_path)

    p = torch.sigmoid(patch_z)

    with torch.no_grad():
        img_l, img_r, image_size = prepare_stereo_pair(
            left, right, p, spec, f_u, baseline, device,
            shape=opts.shape, area_frac=opts.area_frac,
            resample=opts.resample, quantize=opts.quantize,
        )

        calibs_fu = torch.tensor([float(calib.f_u)], device=device, dtype=torch.float32)
        calibs_baseline = torch.tensor([float(baseline)], device=device, dtype=torch.float32)
        calibs_proj = torch.tensor(
            np.asarray(calib.P, dtype=np.float32)[None, ...], device=device
        )
        calibs_proj_r = torch.tensor(
            np.asarray(calib_r.P, dtype=np.float32)[None, ...], device=device
        )

        outputs = model(img_l, img_r, calibs_fu, calibs_baseline,
                        calibs_proj, calibs_Proj_R=calibs_proj_r)

        logits_all = car_logits_in_radius(
            outputs["bbox_cls"], locations_bev,
            spec.loc_x, spec.loc_z, opts.match_radius,
            num_classes=int(cfg.num_classes),
            num_angles=int(cfg.num_angles),
        )
        patched_max_s = float(logits_all.sigmoid().max().item()) if logits_all.numel() else 0.0

        n_nms, nms_max = count_nms_matched_cars(
            outputs, cfg, image_size, calib.P, spec.loc_x, spec.loc_z, opts.match_radius
        )

    return dict(
        patched_max_s=patched_max_s,
        suppressed=int(patched_max_s < SCORE_THRESH),
        nms_n=n_nms,
        nms_max=nms_max,
    )


def depth_bin(d: float) -> int:
    for i in range(len(DEPTH_BINS) - 1):
        if d < DEPTH_BINS[i + 1]:
            return i
    return len(DEPTH_BINS) - 2


def print_summary(rows: list[dict], title: str):
    subset = [r for r in rows if r.get("split") in title.split("/")]
    if not subset:
        subset = rows
    total = len(subset)
    supp = sum(r["suppressed"] for r in subset)
    print(f"\n{'─'*62}")
    print(f"  {title}  ({total} frames)")
    print(f"{'─'*62}")
    print(f"  Suppressed (score < {SCORE_THRESH}): {supp}/{total}  ({supp/max(total,1):.0%})")
    nms_gone = sum(1 for r in subset if r["nms_n"] == 0)
    print(f"  NMS detections gone:               {nms_gone}/{total}  ({nms_gone/max(total,1):.0%})")
    patched_scores = [r["patched_max_s"] for r in subset]
    print(f"  patched max_s   mean={np.mean(patched_scores):.4f}  "
          f"median={np.median(patched_scores):.4f}  "
          f"worst={max(patched_scores):.4f}")
    clean_scores = [r["clean_score"] for r in subset]
    print(f"  clean   max_s   mean={np.mean(clean_scores):.4f}  "
          f"median={np.median(clean_scores):.4f}")
    print()
    print(f"  {'Depth bin':<10} {'N':>4} {'Suppressed':>12} {'Rate':>6} "
          f"{'Clean':>7} {'Patched':>9} {'NMS gone':>9}")
    print(f"  {'─'*10} {'─'*4} {'─'*12} {'─'*6} {'─'*7} {'─'*9} {'─'*9}")
    for b, label in enumerate(DEPTH_LABELS):
        brows = [r for r in subset if r["depth_bin"] == b]
        if not brows:
            continue
        n = len(brows)
        s = sum(r["suppressed"] for r in brows)
        ng = sum(1 for r in brows if r["nms_n"] == 0)
        mc = np.mean([r["clean_score"] for r in brows])
        mp = np.mean([r["patched_max_s"] for r in brows])
        print(f"  {label:<10} {n:>4} {s:>12} {s/n:>5.0%}  {mc:>7.4f} {mp:>9.4f} {ng:>9}")
    print(f"{'─'*62}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True,
                    help="optimize_patch.py output dir (patch_best.pt lives here)")
    ap.add_argument("--images", type=Path,
                    default=Path("dsgn/datasets/adria/training_kitti_labels"))
    ap.add_argument("--csv", type=Path,
                    default=Path("dsgn/datasets/adria/2.training_patch_optimization/patches_localized.csv"))
    ap.add_argument("--val-csv", type=Path, default=None,
                    help="Explicit held-out validation CSV, matching optimize_patch.py. "
                         "When set, --csv is entirely train and no frame split is made.")
    ap.add_argument("--cfg", type=Path,
                    default=Path("dsgn/checkpoints/kitti/dsgn_12g_b/save_config_awsim.py"))
    ap.add_argument("--loadmodel", type=Path,
                    default=Path("dsgn/checkpoints/kitti/dsgn_12g_b/finetune_48.tar"))
    ap.add_argument("--patch", type=Path, default=None,
                    help="Override patch file (default: <run>/patch_best.pt)")
    ap.add_argument("--splits", nargs="+", choices=["train", "val"], default=["val"],
                    help="Which splits to evaluate (default: val only)")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--split-mode", choices=["strided", "contiguous"], default="strided")
    ap.add_argument("--shape", choices=["face", "square"], default="face")
    ap.add_argument("--area-frac", type=float, default=0.50)
    ap.add_argument("--resample", default="bilinear")
    ap.add_argument("--quantize", action="store_true", default=True)
    ap.add_argument("--match-radius", type=float, default=2.0)
    ap.add_argument("--min-visible-frac", type=float, default=0.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--patch-size", type=int, default=64)
    args = ap.parse_args()

    patch_path = args.patch or (args.run / "patch_best.pt")
    if not patch_path.is_file():
        sys.exit(f"patch not found: {patch_path}")

    device = torch.device(args.device)
    cfg = load_cfg(args.cfg)
    model = load_model(cfg, args.loadmodel, device)
    locations_bev = compute_locations_bev(
        cfg.Z_MIN, cfg.Z_MAX, cfg.VOXEL_Z_SIZE,
        cfg.X_MIN, cfg.X_MAX, cfg.VOXEL_X_SIZE, device,
    )

    resumed = load_z_init(patch_path, args.patch_size, device)
    patch_z = resumed.z.to(device)
    print(f"patch: {patch_path}  (saved epoch={resumed.epoch}, "
          f"best_val_max_s={resumed.best_val_max_s})")

    opts = RunOpts(
        loss="logit", temperature=0.2, logit_temperature=1.0, logit_clamp=None,
        match_radius=args.match_radius, score_thresh=SCORE_THRESH, max_matches=3,
        shape=args.shape, area_frac=args.area_frac,
        resample=args.resample, quantize=args.quantize,
    )

    # Build the clean score lookup from CSV(s), computed at localization time.
    all_frames = load_csv(args.csv)
    clean_score_map = {s.frame: 0.0 for s in all_frames}
    score_csvs = [args.csv]
    if args.val_csv is not None:
        score_csvs.append(args.val_csv)
    for score_csv in score_csvs:
        with score_csv.open() as f:
            for row in csv.DictReader(f):
                frame = f"{int(row['frame']):06d}"
                clean_score_map[frame] = float(row.get("score") or 0.0)

    # Reproduce either the explicit event split or the legacy frame split.
    visible_frames, dropped = filter_visible_frames(
        all_frames, args.images, args.shape, args.area_frac, args.min_visible_frac
    )
    print(f"train CSV: dropped {len(dropped)}/{len(all_frames)} off-image frames")

    if args.val_csv is not None:
        train_frames = visible_frames
        all_val_frames = load_csv(args.val_csv)
        val_frames, val_dropped = filter_visible_frames(
            all_val_frames, args.images, args.shape, args.area_frac, args.min_visible_frac
        )
        print(
            f"val CSV: dropped {len(val_dropped)}/{len(all_val_frames)} off-image frames"
        )
        overlap = {frame.frame for frame in train_frames} & {
            frame.frame for frame in val_frames
        }
        if overlap:
            sys.exit(
                f"train and validation CSVs overlap on {len(overlap)} frame IDs "
                f"(e.g. {sorted(overlap)[:5]})"
            )
    else:
        train_frames, val_frames = split_frames(
            visible_frames, args.val_frac, mode=args.split_mode
        )

    frames_to_eval: list = []
    if "val" in args.splits:
        frames_to_eval += [(s, "val") for s in val_frames]
    if "train" in args.splits:
        frames_to_eval += [(s, "train") for s in train_frames]

    print(f"evaluating {len(frames_to_eval)} frames "
          f"({'val' if 'val' in args.splits else ''}"
          f"{'+train' if 'train' in args.splits else ''})\n")

    results: list[dict] = []
    for i, (spec, split) in enumerate(frames_to_eval, 1):
        r = eval_frame(spec, patch_z, model, cfg, locations_bev,
                       args.images, device, opts)
        r["frame"] = spec.frame
        r["split"] = split
        r["depth_m"] = spec.depth_m
        r["ego_dist"] = getattr(spec, "ego_dist", spec.depth_m)
        r["clean_score"] = clean_score_map.get(spec.frame, 0.0)
        r["depth_bin"] = depth_bin(spec.depth_m)
        mark = "SUPP" if r["suppressed"] else "    "
        print(f"  [{i:>3}/{len(frames_to_eval)}] {spec.frame} {split:5s} "
              f"depth={spec.depth_m:5.1f}m  "
              f"clean={r['clean_score']:.3f}  patched={r['patched_max_s']:.3f}  "
              f"nms={r['nms_n']}  {mark}")
        results.append(r)

    # Print summary tables.
    if "val" in args.splits:
        print_summary([r for r in results if r["split"] == "val"], "val")
    if "train" in args.splits:
        print_summary([r for r in results if r["split"] == "train"], "train")
    if "val" in args.splits and "train" in args.splits:
        print_summary(results, "val+train (all visible)")

    # Save per-frame CSV.
    out_csv = args.run / "eval_results.csv"
    fieldnames = ["frame", "split", "depth_m", "ego_dist", "depth_bin",
                  "clean_score", "patched_max_s", "suppressed", "nms_n", "nms_max"]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in fieldnames})
    print(f"\nper-frame results → {out_csv}")


if __name__ == "__main__":
    main()
