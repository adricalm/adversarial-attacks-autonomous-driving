"""Visualize optimized patch on sample train/val frames."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "patch_optimization"))
from patch_geometry import (
    FacePlacement,
    load_face_csv,
    paste_pil,
    stereo_rects,
    visible_frac,
)


def draw_rects(img_pil: Image.Image, face: FacePlacement, patch_rect, thickness=3):
    """Draw face box (red) and patch rect (lime)."""
    out = img_pil.copy()
    draw = ImageDraw.Draw(out)
    draw.rectangle(
        [int(face.face_x0), int(face.face_y0), int(face.face_x1), int(face.face_y1)],
        outline=(255, 40, 40),
        width=thickness,
    )
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


def render_frame(
    face: FacePlacement,
    images_root: Path,
    patch_pil: Image.Image,
    area_frac: float,
    scale: float,
):
    """Render left + right side-by-side with patch + annotations."""
    left_path = images_root / "image_2" / f"{face.frame}.png"
    right_path = images_root / "image_3" / f"{face.frame}.png"
    calib_path = images_root / "calib" / f"{face.frame}.txt"
    if not left_path.is_file() or not calib_path.is_file():
        return None

    placed = stereo_rects(face, area_frac, calib_path)
    if placed is None:
        return None
    (x0, y0, w, h), (rx0, ry0, _, _), disp = placed

    left_img = Image.open(left_path).convert("RGB")
    right_img = Image.open(right_path).convert("RGB")
    patch_resized = patch_pil.resize((w, h), Image.BILINEAR)
    left_patched = paste_pil(left_img, patch_resized, x0, y0)
    right_patched = paste_pil(right_img, patch_resized, rx0, ry0)

    left_rect = (x0, y0, w, h)
    right_rect = (rx0, ry0, w, h)
    left_ann = add_label(
        draw_rects(left_patched, face, left_rect),
        f"L  frame={face.frame}  depth={face.depth_m:.1f}m  "
        f"patch={w}×{h}px  vis={visible_frac(left_rect):.0%}",
    )
    right_ann = add_label(
        draw_rects(right_patched, face, right_rect),
        f"R  disp={disp:.1f}px  right_x0={rx0}  vis={visible_frac(right_rect):.0%}",
    )

    new_w = int(left_ann.width * scale)
    new_h = int(left_ann.height * scale)
    left_s = left_ann.resize((new_w, new_h), Image.LANCZOS)
    right_s = right_ann.resize((new_w, new_h), Image.LANCZOS)
    combined = Image.new("RGB", (new_w * 2, new_h))
    combined.paste(left_s, (0, 0))
    combined.paste(right_s, (new_w, 0))
    return combined


def pick_frames(
    rows: list[FacePlacement], images_root: Path, n: int, area_frac: float
) -> list[FacePlacement]:
    """Pick n frames spread across depth, all with a visible stereo patch."""
    visible: list[FacePlacement] = []
    for face in rows:
        calib_path = images_root / "calib" / f"{face.frame}.txt"
        if not calib_path.is_file():
            continue
        placed = stereo_rects(face, area_frac, calib_path)
        if placed is None:
            continue
        left_rect, right_rect, _ = placed
        if visible_frac(left_rect) > 0 and visible_frac(right_rect) > 0:
            visible.append(face)
    if not visible:
        return rows[:n]
    visible.sort(key=lambda item: item.depth_m)
    step = max(1, len(visible) // n)
    return visible[::step][:n]


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
    rows = load_face_csv(args.csv)

    if args.frames:
        frame_set = {f"{int(x):06d}" for x in args.frames}
        selected = [face for face in rows if face.frame in frame_set]
    else:
        selected = pick_frames(rows, args.images, args.n_frames, args.area_frac)

    print(f"patch: {patch_path}  ({patch_pil.size[0]}×{patch_pil.size[1]}px)")
    print(f"rendering {len(selected)} frames → {out_dir}")

    rendered = []
    for face in selected:
        img = render_frame(face, args.images, patch_pil, args.area_frac, args.scale)
        if img is None:
            print(f"  skip {face.frame} (no image or rect)")
            continue
        out_path = out_dir / f"{face.frame}.png"
        img.save(out_path)
        print(f"  {face.frame}  depth={face.depth_m:.1f}m  → {out_path.name}")
        rendered.append(img)

    if rendered:
        montage = make_montage(rendered, cols=2)
        if montage:
            mp = out_dir / "montage.png"
            montage.save(mp)
            print(f"\nmontage → {mp}")


if __name__ == "__main__":
    main()
