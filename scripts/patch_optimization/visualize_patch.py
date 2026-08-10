"""Visualize the current best patch pasted on real training/val images.

Usage:
    python scripts/patch_optimization/visualize_patch.py \
        --run  dsgn/datasets/adria/2.training_patch_optimization/optimize_logit_face050 \
        --images dsgn/datasets/adria/training_kitti_labels \
        --csv   dsgn/datasets/adria/2.training_patch_optimization/patches_localized.csv \
        --out   /tmp/patch_vis

Picks a spread of frames across depth bins, renders the patch on left+right,
draws the face bounding box in red and the pasted patch rect in green.
Saves one PNG per frame + a compact contact-sheet montage.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ── geometry constants (must match optimize_patch.py) ────────────────────────
FULL_W, FULL_H = 1920, 1080
DOWNSCALE = 0.5


def load_csv(path: Path):
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            rows.append(
                dict(
                    frame=f"{int(r['frame']):06d}",
                    depth_m=float(r["depth_m"]),
                    ego_dist=float(r.get("ego_dist") or r["depth_m"]),
                    face_x0=float(r.get("x0") or 0),
                    face_y0=float(r.get("y0") or 0),
                    face_x1=float(r.get("x1") or 0),
                    face_y1=float(r.get("y1") or 0),
                    loc_x=float(r["loc_x"]),
                    loc_z=float(r["loc_z"]),
                )
            )
    rows.sort(key=lambda x: x["frame"])
    return rows


def patch_rect_face(row: dict, area_frac: float):
    """(x0, y0, w, h) in full-res pixels, face geometry."""
    fw = row["face_x1"] - row["face_x0"]
    fh = row["face_y1"] - row["face_y0"]
    if fw <= 0 or fh <= 0:
        return None
    k = float(np.sqrt(max(area_frac, 1e-6)))
    w = max(1, int(round(fw * k)))
    h = max(1, int(round(fh * k)))
    cx = 0.5 * (row["face_x0"] + row["face_x1"])
    cy = 0.5 * (row["face_y0"] + row["face_y1"])
    return int(round(cx - w / 2.0)), int(round(cy - h / 2.0)), w, h


def load_calib(calib_path: Path):
    data = {}
    with calib_path.open() as f:
        for line in f:
            k, _, v = line.partition(":")
            data[k.strip()] = [float(x) for x in v.split()]
    P2 = np.array(data["P2"]).reshape(3, 4)
    P3 = np.array(data["P3"]).reshape(3, 4)
    f_u = P2[0, 0]
    baseline = abs(P2[0, 3] - P3[0, 3]) / f_u
    return f_u, baseline


def visible_frac(rect, img_w=FULL_W, img_h=FULL_H):
    x0, y0, w, h = rect
    ix0, iy0 = max(0, x0), max(0, y0)
    ix1, iy1 = min(img_w, x0 + w), min(img_h, y0 + h)
    return max(0, ix1 - ix0) * max(0, iy1 - iy0) / float(max(1, w * h))


def paste_patch(img_pil: Image.Image, patch_pil: Image.Image, x0: int, y0: int):
    """Paste patch at (x0,y0) clipped to canvas."""
    iw, ih = img_pil.size
    pw, ph = patch_pil.size
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(iw, x0 + pw), min(ih, y0 + ph)
    if cx0 >= cx1 or cy0 >= cy1:
        return img_pil.copy()
    region = patch_pil.crop((cx0 - x0, cy0 - y0, cx1 - x0, cy1 - y0))
    out = img_pil.copy()
    out.paste(region, (cx0, cy0))
    return out


def draw_rects(img_pil: Image.Image, face_rect, patch_rect, thickness=3):
    """Draw face box (red) and patch rect (lime)."""
    out = img_pil.copy()
    draw = ImageDraw.Draw(out)
    fx0, fy0, fx1, fy1 = (
        int(face_rect[0]),
        int(face_rect[1]),
        int(face_rect[2]),
        int(face_rect[3]),
    )
    draw.rectangle([fx0, fy0, fx1, fy1], outline=(255, 40, 40), width=thickness)
    if patch_rect is not None:
        px0, py0, pw, ph = patch_rect
        draw.rectangle(
            [px0, py0, px0 + pw, py0 + ph], outline=(60, 230, 60), width=thickness
        )
    return out


def add_label(img_pil: Image.Image, text: str, color=(255, 255, 80)):
    out = img_pil.copy()
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
    except Exception:
        font = ImageFont.load_default()
    draw.text((12, 10), text, fill=color, font=font)
    return out


def render_frame(row, images_root, patch_pil, area_frac=0.50, scale=0.4):
    """Render left + right side-by-side with patch + annotations."""
    frame = row["frame"]
    left_path = images_root / "image_2" / f"{frame}.png"
    right_path = images_root / "image_3" / f"{frame}.png"
    calib_path = images_root / "calib" / f"{frame}.txt"

    if not left_path.is_file():
        return None

    f_u, baseline = load_calib(calib_path)
    left_img = Image.open(left_path).convert("RGB")
    right_img = Image.open(right_path).convert("RGB")

    rect = patch_rect_face(row, area_frac)
    if rect is None:
        return None

    x0, y0, w, h = rect
    disp = f_u * baseline / row["depth_m"]
    right_x0 = int(round(x0 - disp))

    patch_resized = patch_pil.resize((w, h), Image.BILINEAR)

    # Paste patch onto images
    left_patched = paste_patch(left_img, patch_resized, x0, y0)
    right_patched = paste_patch(right_img, patch_resized, right_x0, y0)

    # Draw boxes: red = full face AABB, green = 50%-area patch rect
    face_box = (row["face_x0"], row["face_y0"], row["face_x1"], row["face_y1"])

    left_ann = draw_rects(left_patched, face_box, rect)
    right_ann = draw_rects(right_patched, face_box, (right_x0, y0, w, h))

    depth = row["depth_m"]
    vis_l = visible_frac(rect)
    vis_r = visible_frac((right_x0, y0, w, h))
    left_ann = add_label(
        left_ann,
        f"L  frame={frame}  depth={depth:.1f}m  patch={w}×{h}px  vis={vis_l:.0%}",
    )
    right_ann = add_label(
        right_ann,
        f"R  disp={disp:.1f}px  right_x0={right_x0}  vis={vis_r:.0%}",
    )

    # Scale down, then stitch L|R
    new_w = int(left_ann.width * scale)
    new_h = int(left_ann.height * scale)
    left_s = left_ann.resize((new_w, new_h), Image.LANCZOS)
    right_s = right_ann.resize((new_w, new_h), Image.LANCZOS)
    combined = Image.new("RGB", (new_w * 2, new_h))
    combined.paste(left_s, (0, 0))
    combined.paste(right_s, (new_w, 0))
    return combined


def pick_frames(rows, n=12):
    """Pick n frames spread across depth bins, all with visible patch."""
    visible = [r for r in rows if patch_rect_face(r, 0.50) is not None
               and visible_frac(patch_rect_face(r, 0.50)) > 0
               and visible_frac(
                   (_right_x0_for(r, patch_rect_face(r, 0.50)), 0,
                    patch_rect_face(r, 0.50)[2], patch_rect_face(r, 0.50)[3])
               ) > 0]
    if not visible:
        return rows[:n]
    visible.sort(key=lambda r: r["depth_m"])
    step = max(1, len(visible) // n)
    return visible[::step][:n]


def _right_x0_for(row, rect):
    calib_path = Path(
        "dsgn/datasets/adria/training_kitti_labels/calib"
    ) / f"{row['frame']}.txt"
    try:
        f_u, baseline = load_calib(calib_path)
    except Exception:
        return rect[0]
    disp = f_u * baseline / row["depth_m"]
    return int(round(rect[0] - disp))


def make_montage(frames_imgs, cols=2):
    """Stack rows of rendered frame images."""
    if not frames_imgs:
        return None
    w, h = frames_imgs[0].size
    rows = (len(frames_imgs) + cols - 1) // cols
    montage = Image.new("RGB", (w * cols, h * rows), (30, 30, 30))
    for i, img in enumerate(frames_imgs):
        r, c = divmod(i, cols)
        montage.paste(img, (c * w, r * h))
    return montage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True,
                    help="Output dir of optimize_patch.py (contains patch_best.png)")
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None,
                    help="Where to save PNGs (default: --run/vis/)")
    ap.add_argument("--patch", type=Path, default=None,
                    help="Override patch PNG (default: <run>/patch_best.png)")
    ap.add_argument("--n-frames", type=int, default=12,
                    help="How many frames to visualize")
    ap.add_argument("--area-frac", type=float, default=0.50)
    ap.add_argument("--scale", type=float, default=0.40,
                    help="Downsample each 1920×1080 image by this factor before stitching")
    ap.add_argument("--frames", nargs="*", default=None,
                    help="Specific frame IDs to render (e.g. 000019 000037)")
    args = ap.parse_args()

    patch_path = args.patch or (args.run / "patch_best.png")
    if not patch_path.is_file():
        sys.exit(f"patch not found: {patch_path}")

    out_dir = args.out or (args.run / "vis")
    out_dir.mkdir(parents=True, exist_ok=True)

    patch_pil = Image.open(patch_path).convert("RGB")
    rows = load_csv(args.csv)

    if args.frames:
        frame_set = set(f"{int(x):06d}" for x in args.frames)
        selected = [r for r in rows if r["frame"] in frame_set]
    else:
        selected = pick_frames(rows, args.n_frames)

    print(f"patch: {patch_path}  ({patch_pil.size[0]}×{patch_pil.size[1]}px)")
    print(f"rendering {len(selected)} frames → {out_dir}")

    rendered = []
    for row in selected:
        img = render_frame(row, args.images, patch_pil, args.area_frac, args.scale)
        if img is None:
            print(f"  skip {row['frame']} (no image or rect)")
            continue
        out_path = out_dir / f"{row['frame']}.png"
        img.save(out_path)
        print(f"  {row['frame']}  depth={row['depth_m']:.1f}m  → {out_path.name}")
        rendered.append(img)

    if rendered:
        montage = make_montage(rendered, cols=2)
        if montage:
            mp = out_dir / "montage.png"
            montage.save(mp)
            print(f"\nmontage → {mp}")


if __name__ == "__main__":
    main()
