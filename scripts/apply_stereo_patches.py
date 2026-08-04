#!/usr/bin/env python3
"""Apply square B&W noise patches to a KITTI-layout stereo dataset.

Left image (image_2): patch center comes from the config CSV (you set these).
Right image (image_3): same noise texture, center shifted by stereo disparity
  disparity_px = f_u * baseline / depth_m   (from calib + depth per frame)

Typical workflow (frame-by-frame)
---------------------------------
1. Preview one frame; each run updates the CSV and the compare image (~instant):

     python3 scripts/apply_stereo_patches.py preview \\
       --frame 105 --center-x 921 --center-y 420 --size 80 --depth 12.4

   Open dsgn/datasets/adria/preview/000105_compare.png. Tweak coords and re-run;
   the CSV row for that frame is overwritten each time.

2. Repeat for each frame, then build the full dataset once:

     python3 scripts/apply_stereo_patches.py apply \\
       --source dsgn/datasets/arka/dsgn_awsim/testing_offline \\
       --output dsgn/datasets/adria/testing_offline_patched \\
       --config dsgn/datasets/adria/testing_offline_patched/patches_100_200.csv \\
       --frames 100-200

   Static (non-noise) patch texture:

     python3 scripts/apply_stereo_patches.py apply \\
       --source dsgn/datasets/arka/dsgn_awsim/testing_offline \\
       --output dsgn/datasets/adria/testing_offline_patched_optimized \\
       --config dsgn/datasets/adria/testing_offline_patched/patches_100_200.csv \\
       --patch-image multimedia/ChatGPT-patch.png \\
       --copy-calib

3. Run DSGN inference on --output, then copy awsim_output_* into dsgn_offline/resource/.

Config CSV columns (header required):
  frame,center_x,center_y,size,depth_m,seed
  - frame:     six-digit id, e.g. 000105
  - center_x:  patch center u (pixels, origin top-left)
  - center_y:  patch center v
  - size:      square side in pixels (optional; default from init-config / --default-size)
  - depth_m:   assumed depth at patch center for right-camera shift (optional)
  - seed:      RNG seed for noise (optional; default = frame number)
"""
from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
FRAME_RE = re.compile(r"^(\d{6})$")


@dataclass
class PatchSpec:
    frame: str
    center_x: float
    center_y: float
    size: int
    depth_m: float | None
    seed: int


def parse_frame_range(spec: str) -> list[str]:
    """Parse '100-200' or '100,105,110' into ['000100', ...]."""
    frames: list[str] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
            if end < start:
                raise ValueError(f"invalid frame range: {part}")
            for i in range(start, end + 1):
                frames.append(f"{i:06d}")
        else:
            frames.append(f"{int(part):06d}")
    return frames


def read_calib(calib_path: Path) -> tuple[float, float]:
    """Return (f_u, baseline_m) from a KITTI calib file."""
    values: dict[str, list[float]] = {}
    with calib_path.open() as f:
        for line in f:
            if ":" not in line:
                continue
            key, rest = line.split(":", 1)
            values[key.strip()] = [float(x) for x in rest.split()]
    p2 = values["P2"]
    p3 = values["P3"]
    f_u = p2[0]
    baseline = abs(p2[3] - p3[3]) / f_u
    return f_u, baseline


def disparity_px(f_u: float, baseline_m: float, depth_m: float) -> float:
    if depth_m <= 0:
        raise ValueError(f"depth must be positive, got {depth_m}")
    return f_u * baseline_m / depth_m


def sample_depth_m(
    depth_dir: Path | None,
    frame: str,
    center_x: float,
    center_y: float,
    fallback: float,
) -> float:
    if depth_dir is None or not HAS_NUMPY:
        return fallback
    u, v = int(round(center_x)), int(round(center_y))
    for name in (f"{frame}.npy", f"{frame}_r.npy"):
        path = depth_dir / name
        if not path.is_file():
            continue
        depth = np.load(path)
        if depth.ndim != 2:
            continue
        if 0 <= v < depth.shape[0] and 0 <= u < depth.shape[1]:
            val = float(depth[v, u])
            if val > 0.5:
                return val
    return fallback


def make_noise_patch(size: int, seed: int) -> Image.Image:
    rng = random.Random(seed)
    gray = Image.new("L", (size, size))
    gray.putdata([rng.randint(0, 255) for _ in range(size * size)])
    return gray.convert("RGB")


RESAMPLE_FILTERS = {
    "lanczos": Image.Resampling.LANCZOS,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "nearest": Image.Resampling.NEAREST,
    "area": Image.Resampling.BOX,
}


def make_patch(
    size: int,
    seed: int,
    static_patch: Image.Image | None = None,
    resample: str = "lanczos",
) -> Image.Image:
    """Square patch of side `size`: static image resized, or B&W noise.

    `resample` must match what the optimizer rendered with, otherwise the
    deployed patch is not the one that was trained -- an optimized patch is
    saturated high-frequency content, which is exactly where filters disagree
    most. 'bilinear' pairs with optimize_patch's antialiased bilinear.
    """
    if static_patch is None:
        return make_noise_patch(size, seed)
    return static_patch.resize((size, size), RESAMPLE_FILTERS[resample]).convert("RGB")


def paste_patch(base: Image.Image, patch: Image.Image, center_x: float, center_y: float) -> Image.Image:
    """Paste patch centered at (center_x, center_y); clip at image bounds."""
    out = base.copy()
    pw, ph = patch.size
    half_w, half_h = pw // 2, ph // 2
    x0 = int(round(center_x)) - half_w
    y0 = int(round(center_y)) - half_h
    x1, y1 = x0 + pw, y0 + ph

    # Intersection with image [0, W) x [0, H)
    ix0 = max(0, x0)
    iy0 = max(0, y0)
    ix1 = min(out.width, x1)
    iy1 = min(out.height, y1)
    if ix0 >= ix1 or iy0 >= iy1:
        return out

    px0 = ix0 - x0
    py0 = iy0 - y0
    px1 = px0 + (ix1 - ix0)
    py1 = py0 + (iy1 - iy0)
    crop = patch.crop((px0, py0, px1, py1))
    out.paste(crop, (ix0, iy0))
    return out


def load_patch_config(
    config_path: Path,
    default_size: int,
    default_depth: float,
) -> dict[str, PatchSpec]:
    specs: dict[str, PatchSpec] = {}
    with config_path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"frame", "center_x", "center_y"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"config must include columns: {sorted(required)}")
        for row in reader:
            frame = row["frame"].strip()
            if not FRAME_RE.match(frame):
                raise ValueError(f"invalid frame id: {frame!r}")
            size_s = (row.get("size") or "").strip()
            depth_s = (row.get("depth_m") or "").strip()
            seed_s = (row.get("seed") or "").strip()
            specs[frame] = PatchSpec(
                frame=frame,
                center_x=float(row["center_x"]),
                center_y=float(row["center_y"]),
                size=int(size_s) if size_s else default_size,
                depth_m=float(depth_s) if depth_s else None,
                seed=int(seed_s) if seed_s else int(frame),
            )
    return specs


def write_patch_config(
    path: Path,
    frames: list[str],
    default_center: tuple[int, int],
    default_size: int,
    default_depth: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "center_x", "center_y", "size", "depth_m", "seed"])
        cx, cy = default_center
        for frame in frames:
            writer.writerow([frame, cx, cy, default_size, default_depth, int(frame)])


def list_frame_ids(image_dir: Path) -> list[str]:
    return sorted(p.stem for p in image_dir.glob("*.png") if FRAME_RE.match(p.stem))


def normalize_frame(frame: str) -> str:
    frame = frame.strip()
    if FRAME_RE.match(frame):
        return frame
    return f"{int(frame):06d}"


def resolve_depth_m(
    source: Path,
    frame: str,
    center_x: float,
    center_y: float,
    depth_m: float | None,
    default_depth: float,
    use_depth_maps: bool,
) -> float:
    if depth_m is not None:
        return depth_m
    depth_dir = source / "depth" if use_depth_maps else None
    if depth_dir and not depth_dir.is_dir():
        depth_dir = None
    if depth_dir and not HAS_NUMPY:
        depth_dir = None
    return sample_depth_m(depth_dir, frame, center_x, center_y, default_depth)


def patch_stereo_pair(
    left_img: Image.Image,
    right_img: Image.Image,
    calib_path: Path,
    center_x: float,
    center_y: float,
    size: int,
    depth_m: float,
    seed: int,
    static_patch: Image.Image | None = None,
    resample: str = "lanczos",
) -> tuple[Image.Image, Image.Image, float, float]:
    f_u, baseline = read_calib(calib_path)
    disp = disparity_px(f_u, baseline, depth_m)
    right_cx = center_x - disp
    right_cy = center_y
    patch = make_patch(size, seed, static_patch, resample=resample)
    left_out = paste_patch(left_img, patch, center_x, center_y)
    right_out = paste_patch(right_img, patch, right_cx, right_cy)
    return left_out, right_out, right_cx, disp


def draw_patch_box(img: Image.Image, center_x: float, center_y: float, size: int) -> Image.Image:
    """Outline patch bounds on a copy (helps tune placement)."""
    out = img.copy()
    draw = ImageDraw.Draw(out)
    half = size // 2
    x0 = int(round(center_x)) - half
    y0 = int(round(center_y)) - half
    x1 = x0 + size
    y1 = y0 + size
    draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=2)
    draw.line(
        [center_x - 8, center_y, center_x + 8, center_y],
        fill=(255, 255, 0),
        width=1,
    )
    draw.line(
        [center_x, center_y - 8, center_x, center_y + 8],
        fill=(255, 255, 0),
        width=1,
    )
    return out


def make_compare_sheet(
    left_pat: Image.Image,
    right_pat: Image.Image,
    title: str,
) -> Image.Image:
    """Side-by-side: left patched | right patched."""
    w, h = left_pat.size
    label_h = 28
    sheet = Image.new("RGB", (w * 2, h + label_h * 2), (32, 32, 32))
    sheet.paste(left_pat, (0, label_h))
    sheet.paste(right_pat, (w, label_h))
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 6), "left patched", fill=(220, 220, 220))
    draw.text((w + 8, 6), "right patched", fill=(220, 220, 220))
    draw.text((8, h + label_h + 6), title, fill=(180, 255, 180))
    return sheet


def upsert_csv_row(
    config_path: Path,
    frame: str,
    center_x: float,
    center_y: float,
    size: int,
    depth_m: float,
    seed: int,
) -> None:
    fieldnames = ["frame", "center_x", "center_y", "size", "depth_m", "seed"]
    rows: list[dict[str, str]] = []
    if config_path.is_file():
        with config_path.open(newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                fieldnames = list(reader.fieldnames)
            rows = list(reader)
    new_row = {
        "frame": frame,
        "center_x": str(int(center_x) if center_x == int(center_x) else center_x),
        "center_y": str(int(center_y) if center_y == int(center_y) else center_y),
        "size": str(size),
        "depth_m": str(depth_m),
        "seed": str(seed),
    }
    replaced = False
    for i, row in enumerate(rows):
        if row.get("frame", "").strip() == frame:
            rows[i] = new_row
            replaced = True
            break
    if not replaced:
        rows.append(new_row)
    rows.sort(key=lambda r: r["frame"])
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def cmd_init_config(args: argparse.Namespace) -> int:
    frames = parse_frame_range(args.frames)
    write_patch_config(
        Path(args.output),
        frames,
        tuple(args.default_center),
        args.default_size,
        args.default_depth,
    )
    print(f"Wrote {len(frames)} rows to {args.output}")
    print("Edit center_x, center_y per frame (left image coordinates), then run apply.")
    return 0


def cmd_apply(args: argparse.Namespace) -> int:
    source = Path(args.source)
    output = Path(args.output)
    config_path = Path(args.config)
    specs = load_patch_config(config_path, args.default_size, args.default_depth)

    src_left = source / "image_2"
    src_right = source / "image_3"
    src_calib = source / "calib"
    for d, name in ((src_left, "image_2"), (src_right, "image_3"), (src_calib, "calib")):
        if not d.is_dir():
            print(f"error: missing {name}/ under {source}", file=sys.stderr)
            return 1

    out_left = output / "image_2"
    out_right = output / "image_3"
    out_calib = output / "calib"
    out_left.mkdir(parents=True, exist_ok=True)
    out_right.mkdir(parents=True, exist_ok=True)
    if args.copy_calib or not out_calib.exists() or not any(out_calib.iterdir()):
        if out_calib.exists():
            shutil.rmtree(out_calib)
        shutil.copytree(src_calib, out_calib)

    static_patch: Image.Image | None = None
    if getattr(args, "patch_image", None):
        patch_path = Path(args.patch_image)
        if not patch_path.is_file():
            print(f"error: patch image not found: {patch_path}", file=sys.stderr)
            return 1
        static_patch = Image.open(patch_path).convert("RGB")
        print(f"Using static patch: {patch_path}")

    all_frames = list_frame_ids(src_left)
    if args.frames:
        patch_only = set(parse_frame_range(args.frames))
        # Still copy every frame; only listed frames are patched if in specs.
    else:
        patch_only = set(specs.keys())

    patched = 0
    copied = 0
    for frame in all_frames:
        left_in = src_left / f"{frame}.png"
        right_in = src_right / f"{frame}.png"
        left_out = out_left / f"{frame}.png"
        right_out = out_right / f"{frame}.png"

        if frame in specs and frame in patch_only:
            spec = specs[frame]
            calib_path = src_calib / f"{frame}.txt"
            depth_m = resolve_depth_m(
                source,
                frame,
                spec.center_x,
                spec.center_y,
                spec.depth_m,
                args.default_depth,
                args.use_depth_maps,
            )
            left_img = Image.open(left_in).convert("RGB")
            right_img = Image.open(right_in).convert("RGB")
            left_img, right_img, right_cx, disp = patch_stereo_pair(
                left_img,
                right_img,
                calib_path,
                spec.center_x,
                spec.center_y,
                spec.size,
                depth_m,
                spec.seed,
                static_patch,
                resample=args.patch_resample,
            )
            left_img.save(left_out)
            right_img.save(right_out)
            patched += 1
            if args.verbose:
                print(
                    f"{frame}: left=({spec.center_x:.1f},{spec.center_y:.1f}) "
                    f"right=({right_cx:.1f},{spec.center_y:.1f}) "
                    f"size={spec.size} depth={depth_m:.2f}m disp={disp:.1f}px"
                )
        else:
            shutil.copy2(left_in, left_out)
            shutil.copy2(right_in, right_out)
            copied += 1

    print(f"Done → {output}")
    print(f"  patched: {patched} stereo pairs")
    print(f"  copied unchanged: {copied} stereo pairs")
    return 0


def cmd_preview(args: argparse.Namespace) -> int:
    source = Path(args.source)
    frame = normalize_frame(args.frame)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    left_path = source / "image_2" / f"{frame}.png"
    right_path = source / "image_3" / f"{frame}.png"
    calib_path = source / "calib" / f"{frame}.txt"
    for path, label in (
        (left_path, "image_2"),
        (right_path, "image_3"),
        (calib_path, "calib"),
    ):
        if not path.is_file():
            print(f"error: missing {label}/{frame} under {source}", file=sys.stderr)
            return 1

    center_x, center_y = args.center_x, args.center_y
    size = args.size
    seed = args.seed if args.seed is not None else int(frame)
    depth_m = resolve_depth_m(
        source, frame, center_x, center_y, args.depth, args.default_depth, args.use_depth_maps
    )

    static_patch: Image.Image | None = None
    if getattr(args, "patch_image", None):
        patch_path = Path(args.patch_image)
        if not patch_path.is_file():
            print(f"error: patch image not found: {patch_path}", file=sys.stderr)
            return 1
        static_patch = Image.open(patch_path).convert("RGB")

    left_orig = Image.open(left_path).convert("RGB")
    right_orig = Image.open(right_path).convert("RGB")
    left_pat, right_pat, right_cx, disp = patch_stereo_pair(
        left_orig,
        right_orig,
        calib_path,
        center_x,
        center_y,
        size,
        depth_m,
        seed,
        static_patch,
        resample=args.patch_resample,
    )

    title = (
        f"{frame}  left=({center_x:.0f},{center_y:.0f})  "
        f"right=({right_cx:.0f},{center_y:.0f})  "
        f"size={size}  depth={depth_m:.1f}m  disp={disp:.1f}px"
    )
    compare = make_compare_sheet(left_pat, right_pat, title)
    compare_path = out_dir / f"{frame}_compare.png"
    compare.save(compare_path)

    if args.show_box:
        box_path = out_dir / f"{frame}_left_box.png"
        draw_patch_box(left_orig, center_x, center_y, size).save(box_path)
        print(f"  box guide → {box_path}")

    csv_line = f"{frame},{center_x:g},{center_y:g},{size},{depth_m:g},{seed}"
    print(f"frame {frame}")
    print(f"  left center : ({center_x:.1f}, {center_y:.1f})")
    print(f"  right center: ({right_cx:.1f}, {center_y:.1f})  disparity={disp:.1f}px")
    print(f"  depth={depth_m:.2f}m  size={size}  seed={seed}")
    print(f"  compare     → {compare_path}")
    if args.no_save_csv:
        print(f"  csv row     → {csv_line}  (not saved; drop --no-save-csv to write CSV)")
    else:
        config_path = Path(args.config)
        upsert_csv_row(config_path, frame, center_x, center_y, size, depth_m, seed)
        print(f"  csv saved   → {config_path}")
        print(f"  csv row     → {csv_line}")

    if args.open:
        opener = "xdg-open" if sys.platform.startswith("linux") else "open"
        try:
            subprocess.run([opener, str(compare_path)], check=False)
        except FileNotFoundError:
            print(f"  (could not run {opener}; open {compare_path} manually)")
    return 0


def cmd_save_row(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    frame = normalize_frame(args.frame)
    seed = args.seed if args.seed is not None else int(frame)
    depth_m = args.depth if args.depth is not None else args.default_depth
    upsert_csv_row(
        config_path, frame, args.center_x, args.center_y, args.size, depth_m, seed
    )
    print(f"Saved {frame} → {config_path}")
    print(
        f"  {frame},{args.center_x:g},{args.center_y:g},"
        f"{args.size},{depth_m:g},{seed}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    init_p = sub.add_parser("init-config", help="create CSV template for patch placements")
    init_p.add_argument("--frames", required=True, help="e.g. 100-200 or 100,105,110")
    init_p.add_argument("--output", required=True, help="output CSV path")
    init_p.add_argument(
        "--default-center",
        type=int,
        nargs=2,
        default=[IMAGE_WIDTH // 2, IMAGE_HEIGHT // 2],
        metavar=("X", "Y"),
        help=f"placeholder center (default: {IMAGE_WIDTH // 2} {IMAGE_HEIGHT // 2})",
    )
    init_p.add_argument("--default-size", type=int, default=80, help="square patch side in px")
    init_p.add_argument("--default-depth", type=float, default=15.0, help="depth for disparity (m)")
    init_p.set_defaults(func=cmd_init_config)

    apply_p = sub.add_parser("apply", help="build patched dataset from config")
    apply_p.add_argument("--source", required=True, help="KITTI-layout source (testing_offline)")
    apply_p.add_argument("--output", required=True, help="patched dataset output directory")
    apply_p.add_argument("--config", required=True, help="patch CSV from init-config")
    apply_p.add_argument(
        "--frames",
        help="only apply patches to these frames (still copy all); default: all rows in CSV",
    )
    apply_p.add_argument("--default-size", type=int, default=80)
    apply_p.add_argument("--default-depth", type=float, default=15.0)
    apply_p.add_argument(
        "--use-depth-maps",
        action="store_true",
        help="sample depth_m from source/depth/*.npy when CSV depth_m is empty",
    )
    apply_p.add_argument("--copy-calib", action="store_true", help="refresh calib/ from source")
    apply_p.add_argument(
        "--patch-image",
        default=None,
        help="static patch PNG/JPG (resized per-frame to CSV size); default: random B&W noise",
    )
    apply_p.add_argument(
        "--patch-resample",
        choices=sorted(RESAMPLE_FILTERS),
        default="bilinear",
        help="filter used to resize --patch-image. Default matches the "
        "optimizer's antialiased bilinear to within 1/255; the earlier "
        "'lanczos' behaviour differed from training by up to 150/255",
    )
    apply_p.add_argument("-v", "--verbose", action="store_true")
    apply_p.set_defaults(func=cmd_apply)

    preview_p = sub.add_parser(
        "preview",
        help="patch one frame instantly and write preview images (fast iteration)",
    )
    preview_p.add_argument(
        "--source",
        default="dsgn/datasets/arka/dsgn_awsim/testing_offline",
        help="KITTI-layout source dataset",
    )
    preview_p.add_argument("--frame", required=True, help="e.g. 105 or 000105")
    preview_p.add_argument("--center-x", type=float, required=True, metavar="X")
    preview_p.add_argument("--center-y", type=float, required=True, metavar="Y")
    preview_p.add_argument("--size", type=int, default=80)
    preview_p.add_argument(
        "--depth",
        type=float,
        default=None,
        help="depth at patch (m); default: --default-depth or depth map",
    )
    preview_p.add_argument("--default-depth", type=float, default=15.0)
    preview_p.add_argument("--seed", type=int, default=None)
    preview_p.add_argument(
        "--config",
        default="dsgn/datasets/adria/testing_offline_patched/patches_100_200.csv",
        help="patch CSV updated on each preview (default: testing_offline_patched/patches_100_200.csv)",
    )
    preview_p.add_argument(
        "--no-save-csv",
        action="store_true",
        help="preview images only; do not write the CSV row",
    )
    preview_p.add_argument(
        "--output",
        default="dsgn/datasets/adria/preview",
        help="where to write preview PNGs",
    )
    preview_p.add_argument(
        "--show-box",
        action="store_true",
        help="also save left_orig with red box at patch center (no noise)",
    )
    preview_p.add_argument(
        "--use-depth-maps",
        action="store_true",
        help="sample depth from source/depth/*.npy when --depth omitted",
    )
    preview_p.add_argument(
        "--patch-image",
        default=None,
        help="static patch PNG/JPG (resized to --size); default: random B&W noise",
    )
    preview_p.add_argument(
        "--patch-resample",
        choices=sorted(RESAMPLE_FILTERS),
        default="bilinear",
        help="filter used to resize --patch-image. Default matches the "
        "optimizer's antialiased bilinear to within 1/255; the earlier "
        "'lanczos' behaviour differed from training by up to 150/255",
    )
    preview_p.add_argument(
        "--open",
        action="store_true",
        help="open compare image with xdg-open after saving",
    )
    preview_p.set_defaults(func=cmd_preview)

    save_p = sub.add_parser(
        "save-row",
        help="write placement into the patch CSV without generating preview images",
    )
    save_p.add_argument("--config", required=True, help="patches CSV to update")
    save_p.add_argument("--frame", required=True)
    save_p.add_argument("--center-x", type=float, required=True)
    save_p.add_argument("--center-y", type=float, required=True)
    save_p.add_argument("--size", type=int, default=80)
    save_p.add_argument("--depth", type=float, default=None)
    save_p.add_argument("--default-depth", type=float, default=15.0)
    save_p.add_argument("--seed", type=int, default=None)
    save_p.set_defaults(func=cmd_save_row)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
