#!/usr/bin/env python3
"""Apply a universal adversarial patch to a KITTI-layout stereo dataset
using the same face geometry as optimize_patch.py (--shape face --area-frac N).

For each frame that has an entry in the localized CSV, the script:
  - Computes the patch rectangle from the rear-face AABB at the chosen area fraction
  - Pastes the patch PNG on left image_2 and (disparity-shifted) right image_3
  - Copies frames without a CSV entry unchanged
  - Copies calib/ from source

Usage
-----
    # 1. Localize patches on the test detections (if not done yet):
    python scripts/patch_optimization/localize_patches.py \\
        --detections dsgn/detections/adria/testing_offline_no_finetune \\
        --calib dsgn/datasets/arka/dsgn_awsim/testing_offline/calib \\
        --output dsgn/datasets/adria/testing_offline_patches_localized.csv \\
        --box-convention kitti --selection closest

    # 2. Apply the optimized patch:
    python scripts/apply_face_patch.py \\
        --source  dsgn/datasets/arka/dsgn_awsim/testing_offline \\
        --csv     dsgn/datasets/adria/testing_offline_patches_localized.csv \\
        --patch   dsgn/datasets/adria/2.training_patch_optimization/optimize_logit_face050/patch_best.png \\
        --out     dsgn/datasets/adria/testing_offline_logit_face050 \\
        --area-frac 0.50
"""
from __future__ import annotations

import argparse
import csv
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

FULL_W, FULL_H = 1920, 1080


@dataclass
class FrameSpec:
    frame: str
    face_x0: float
    face_y0: float
    face_x1: float
    face_y1: float
    depth_m: float
    loc_x: float
    loc_z: float


def load_csv(path: Path) -> dict[str, FrameSpec]:
    specs: dict[str, FrameSpec] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            frame = f"{int(row['frame']):06d}"
            x0 = float(row.get("x0") or 0)
            y0 = float(row.get("y0") or 0)
            x1 = float(row.get("x1") or 0)
            y1 = float(row.get("y1") or 0)
            if x1 <= x0 or y1 <= y0:
                continue  # malformed face box
            specs[frame] = FrameSpec(
                frame=frame,
                face_x0=x0, face_y0=y0, face_x1=x1, face_y1=y1,
                depth_m=float(row["depth_m"]),
                loc_x=float(row.get("loc_x") or 0),
                loc_z=float(row.get("loc_z") or 0),
            )
    return specs


def read_calib(calib_path: Path) -> tuple[float, float]:
    vals: dict[str, list[float]] = {}
    with calib_path.open() as f:
        for line in f:
            if ":" not in line:
                continue
            k, _, rest = line.partition(":")
            vals[k.strip()] = [float(x) for x in rest.split()]
    p2, p3 = vals["P2"], vals["P3"]
    f_u = p2[0]
    baseline = abs(p2[3] - p3[3]) / f_u
    return f_u, baseline


def patch_rect(spec: FrameSpec, area_frac: float) -> tuple[int, int, int, int]:
    """Face-geometry patch rectangle: (x0, y0, w, h) in full-res pixels."""
    fw = spec.face_x1 - spec.face_x0
    fh = spec.face_y1 - spec.face_y0
    k = math.sqrt(max(area_frac, 1e-6))
    w = max(1, round(fw * k))
    h = max(1, round(fh * k))
    cx = 0.5 * (spec.face_x0 + spec.face_x1)
    cy = 0.5 * (spec.face_y0 + spec.face_y1)
    return round(cx - w / 2), round(cy - h / 2), w, h


def visible_area(x0: int, y0: int, w: int, h: int) -> int:
    ix0, iy0 = max(0, x0), max(0, y0)
    ix1, iy1 = min(FULL_W, x0 + w), min(FULL_H, y0 + h)
    return max(0, ix1 - ix0) * max(0, iy1 - iy0)


def paste(img: Image.Image, patch: Image.Image, x0: int, y0: int) -> Image.Image:
    """Paste patch at (x0,y0), clipped to canvas boundaries."""
    iw, ih = img.size
    pw, ph = patch.size
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(iw, x0 + pw), min(ih, y0 + ph)
    if cx0 >= cx1 or cy0 >= cy1:
        return img
    crop = patch.crop((cx0 - x0, cy0 - y0, cx1 - x0, cy1 - y0))
    out = img.copy()
    out.paste(crop, (cx0, cy0))
    return out


def apply_frame(
    spec: FrameSpec,
    src_left: Path,
    src_right: Path,
    calib_path: Path,
    out_left: Path,
    out_right: Path,
    patch_img: Image.Image,
    area_frac: float,
    resample,
) -> bool:
    """Returns True if patch was visible in at least one image, False if skipped."""
    x0, y0, w, h = patch_rect(spec, area_frac)
    if visible_area(x0, y0, w, h) == 0:
        return False  # entirely off-image on the left

    f_u, baseline = read_calib(calib_path)
    disp = f_u * baseline / max(spec.depth_m, 0.01)
    rx0 = round(x0 - disp)

    if visible_area(rx0, y0, w, h) == 0 and visible_area(x0, y0, w, h) == 0:
        return False

    scaled = patch_img.resize((w, h), resample)

    left_img  = Image.open(src_left).convert("RGB")
    right_img = Image.open(src_right).convert("RGB")

    paste(left_img,  scaled, x0,  y0).save(out_left)
    paste(right_img, scaled, rx0, y0).save(out_right)
    return True


def list_frame_ids(image_dir: Path) -> list[str]:
    return sorted(p.stem for p in image_dir.glob("*.png"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source",    required=True, type=Path,
                    help="KITTI-layout source dataset (testing_offline)")
    ap.add_argument("--csv",       required=True, type=Path,
                    help="Localized patch CSV (from localize_patches.py)")
    ap.add_argument("--patch",     required=True, type=Path,
                    help="Patch PNG to apply (patch_best.png)")
    ap.add_argument("--out",       required=True, type=Path,
                    help="Output patched dataset directory")
    ap.add_argument("--area-frac", type=float, default=0.50,
                    help="Fraction of rear-face area to cover (default 0.50)")
    ap.add_argument("--resample",  choices=["bilinear","lanczos","bicubic"],
                    default="bilinear",
                    help="Resampling filter for patch resize (default bilinear, "
                         "matches optimizer deployment)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    if not args.source.is_dir():
        sys.exit(f"source not found: {args.source}")
    if not args.patch.is_file():
        sys.exit(f"patch not found: {args.patch}")

    resample_map = {
        "bilinear": Image.BILINEAR,
        "lanczos":  Image.LANCZOS,
        "bicubic":  Image.BICUBIC,
    }
    resample = resample_map[args.resample]

    patch_img = Image.open(args.patch).convert("RGB")
    specs = load_csv(args.csv)
    print(f"patch:  {args.patch}  ({patch_img.size[0]}×{patch_img.size[1]}px)")
    print(f"csv:    {len(specs)} localized frames")
    print(f"source: {args.source}")
    print(f"out:    {args.out}")
    print(f"area-frac={args.area_frac}  resample={args.resample}")

    src_img2   = args.source / "image_2"
    src_img3   = args.source / "image_3"
    src_calib  = args.source / "calib"
    out_img2   = args.out / "image_2"
    out_img3   = args.out / "image_3"
    out_calib  = args.out / "calib"

    for d in (out_img2, out_img3):
        d.mkdir(parents=True, exist_ok=True)

    if out_calib.exists():
        shutil.rmtree(out_calib)
    shutil.copytree(src_calib, out_calib)

    frames = list_frame_ids(src_img2)
    n_patched = n_skipped = n_copied = 0

    for frame in frames:
        left_in   = src_img2  / f"{frame}.png"
        right_in  = src_img3  / f"{frame}.png"
        calib_f   = src_calib / f"{frame}.txt"
        left_out  = out_img2  / f"{frame}.png"
        right_out = out_img3  / f"{frame}.png"

        if frame in specs:
            ok = apply_frame(
                specs[frame], left_in, right_in, calib_f,
                left_out, right_out, patch_img, args.area_frac, resample,
            )
            if ok:
                n_patched += 1
                if args.verbose:
                    s = specs[frame]
                    x0, y0, w, h = patch_rect(s, args.area_frac)
                    print(f"  PATCH  {frame}  depth={s.depth_m:.1f}m  rect=({x0},{y0},{w}×{h})")
            else:
                shutil.copy2(left_in,  left_out)
                shutil.copy2(right_in, right_out)
                n_skipped += 1
                if args.verbose:
                    print(f"  SKIP   {frame}  (patch rect off-image)")
        else:
            shutil.copy2(left_in,  left_out)
            shutil.copy2(right_in, right_out)
            n_copied += 1

    print(f"\ndone → {args.out}")
    print(f"  patched:  {n_patched}")
    print(f"  skipped (off-image): {n_skipped}")
    print(f"  copied unchanged:    {n_copied}")


if __name__ == "__main__":
    main()
