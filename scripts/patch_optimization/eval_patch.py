"""Offline patch evaluation vs DSGN. See notes26/PATCH_OPTIMIZATION.md."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from patch_geometry import filter_visible_frames, load_face_csv
from optimize_patch import (
    MIN_VISIBLE_FRAC,
    compute_locations_bev,
    load_cfg,
    load_model,
    load_z_init,
    run_frame,
)

SCORE_THRESH = 0.33
DEPTH_BINS = [0, 5, 10, 15, 20, 30, float("inf")]
DEPTH_LABELS = ["<5m", "5-10m", "10-15m", "15-20m", "20-30m", ">30m"]


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
                    default=Path("dsgn/datasets/adria/patch_train/dataset"))
    ap.add_argument("--csv", type=Path,
                    default=Path("dsgn/datasets/adria/patch_train/train.csv"))
    ap.add_argument("--val-csv", type=Path,
                    default=Path("dsgn/datasets/adria/patch_train/val.csv"),
                    help="Explicit held-out validation CSV, matching optimize_patch.py. "
                         "When set, --csv is entirely train and no frame split is made.")
    ap.add_argument("--cfg", type=Path,
                    default=Path("dsgn/checkpoints/kitti/dsgn_12g_b/save_config_awsim.py"))
    ap.add_argument("--loadmodel", type=Path,
                    default=Path("dsgn/checkpoints/kitti/dsgn_12g_b/finetune_48.tar"))
    ap.add_argument("--patch", type=Path, default=None,
                    help="Override patch file (default: <run>/patch_best.pt)")
    ap.add_argument("--splits", nargs="+", choices=["train", "val"], default=["val"])
    ap.add_argument("--area-frac", type=float, default=0.50)
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

    all_frames = load_face_csv(args.csv)
    train_frames, dropped = filter_visible_frames(
        all_frames, args.images, args.area_frac, MIN_VISIBLE_FRAC
    )
    print(f"train CSV: dropped {len(dropped)}/{len(all_frames)} off-image frames")

    all_val_frames = load_face_csv(args.val_csv)
    val_frames, val_dropped = filter_visible_frames(
        all_val_frames, args.images, args.area_frac, MIN_VISIBLE_FRAC
    )
    print(f"val CSV: dropped {len(val_dropped)}/{len(all_val_frames)} off-image frames")
    overlap = {frame.frame for frame in train_frames} & {frame.frame for frame in val_frames}
    if overlap:
        sys.exit(
            f"train and validation CSVs overlap on {len(overlap)} frame IDs "
            f"(e.g. {sorted(overlap)[:5]})"
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
        with torch.no_grad():
            out = run_frame(
                model, cfg, locations_bev, patch_z, spec,
                args.images, device, args.area_frac, nms=True,
            )
        r = dict(
            frame=spec.frame,
            split=split,
            depth_m=spec.depth_m,
            ego_dist=spec.ego_dist,
            clean_score=spec.score,
            depth_bin=depth_bin(spec.depth_m),
            patched_max_s=out.max_s,
            suppressed=int(out.max_s < SCORE_THRESH),
            nms_n=out.n_nms,
            nms_max=out.nms_max,
        )
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
