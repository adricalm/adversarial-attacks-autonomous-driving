#!/usr/bin/env python3
"""Apply an optimized patch to a KITTI-layout stereo dataset. See notes26/PATCH_OPTIMIZATION.md."""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from PIL import Image

from patch_geometry import FacePlacement, paste_pil, stereo_rects, visible_area_px, load_face_csv


def apply_frame(
    spec: FacePlacement,
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
    placed = stereo_rects(spec, area_frac, calib_path)
    if placed is None:
        return False
    (x0, y0, w, h), (rx0, ry0, _, _), _ = placed
    if visible_area_px((x0, y0, w, h)) == 0 and visible_area_px((rx0, ry0, w, h)) == 0:
        return False

    scaled = patch_img.resize((w, h), resample)
    left_img = Image.open(src_left).convert("RGB")
    right_img = Image.open(src_right).convert("RGB")
    paste_pil(left_img, scaled, x0, y0).save(out_left)
    paste_pil(right_img, scaled, rx0, ry0).save(out_right)
    return True


def list_frame_ids(image_dir: Path) -> list[str]:
    return sorted(p.stem for p in image_dir.glob("*.png"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source",    required=True, type=Path,
                    help="KITTI-layout source dataset")
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
    ap.add_argument("--frame-min", type=int, default=None,
                    help="Only process frames with id >= this (inclusive)")
    ap.add_argument("--frame-max", type=int, default=None,
                    help="Only process frames with id <= this (inclusive)")
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
    specs = {face.frame: face for face in load_face_csv(args.csv)}
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
        frame_id = int(frame)
        if args.frame_min is not None and frame_id < args.frame_min:
            continue
        if args.frame_max is not None and frame_id > args.frame_max:
            continue

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
                    placed = stereo_rects(s, args.area_frac, calib_f)
                    assert placed is not None
                    x0, y0, w, h = placed[0]
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
