"""Shared rear-face patch geometry for localize → optimize → apply → visualize.

All of those steps must agree on: image size, how --area-frac maps a face AABB
to a paste rectangle, and the stereo disparity used on the right image.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

FULL_W, FULL_H = 1920, 1080


@dataclass
class FacePlacement:
    frame: str
    depth_m: float
    face_x0: float
    face_y0: float
    face_x1: float
    face_y1: float
    loc_x: float = 0.0
    loc_z: float = 0.0
    score: float = 0.0
    ego_dist: float = 0.0

    @property
    def face_w(self) -> float:
        return self.face_x1 - self.face_x0

    @property
    def face_h(self) -> float:
        return self.face_y1 - self.face_y0


def load_face_csv(path: Path) -> list[FacePlacement]:
    """Load localize_patches.py rows. Skips frames with a degenerate face box."""
    rows: list[FacePlacement] = []
    with path.open() as handle:
        for row in csv.DictReader(handle):
            x0 = float(row.get("x0") or 0)
            y0 = float(row.get("y0") or 0)
            x1 = float(row.get("x1") or 0)
            y1 = float(row.get("y1") or 0)
            if x1 <= x0 or y1 <= y0:
                continue
            loc_x = float(row["loc_x"]) if row.get("loc_x") else 0.0
            loc_z = float(row["loc_z"]) if row.get("loc_z") else 0.0
            depth_m = float(row["depth_m"])
            score_s = row.get("score") or ""
            ego_s = row.get("ego_dist") or ""
            rows.append(
                FacePlacement(
                    frame=f"{int(row['frame']):06d}",
                    depth_m=depth_m,
                    face_x0=x0,
                    face_y0=y0,
                    face_x1=x1,
                    face_y1=y1,
                    loc_x=loc_x,
                    loc_z=loc_z,
                    score=float(score_s) if score_s else 0.0,
                    ego_dist=float(ego_s) if ego_s else math.hypot(loc_x, loc_z),
                )
            )
    rows.sort(key=lambda item: item.frame)
    return rows


def patch_rect(
    face: FacePlacement, area_frac: float
) -> tuple[int, int, int, int] | None:
    """Paste rectangle (x0, y0, w, h) covering `area_frac` of the face AABB."""
    if face.face_w <= 0 or face.face_h <= 0:
        return None
    k = math.sqrt(max(area_frac, 1e-6))
    w = max(1, int(round(face.face_w * k)))
    h = max(1, int(round(face.face_h * k)))
    cx = 0.5 * (face.face_x0 + face.face_x1)
    cy = 0.5 * (face.face_y0 + face.face_y1)
    return int(round(cx - w / 2.0)), int(round(cy - h / 2.0)), w, h


def disparity_px(f_u: float, baseline_m: float, depth_m: float) -> float:
    return f_u * baseline_m / max(depth_m, 0.01)


def right_x0(x0: int, disp: float) -> int:
    return int(round(x0 - disp))


def visible_area_px(
    rect: tuple[int, int, int, int], img_w: int = FULL_W, img_h: int = FULL_H
) -> int:
    x0, y0, w, h = rect
    ix0, iy0 = max(0, x0), max(0, y0)
    ix1, iy1 = min(img_w, x0 + w), min(img_h, y0 + h)
    return max(0, ix1 - ix0) * max(0, iy1 - iy0)


def visible_frac(
    rect: tuple[int, int, int, int], img_w: int = FULL_W, img_h: int = FULL_H
) -> float:
    _, _, w, h = rect
    return visible_area_px(rect, img_w, img_h) / float(max(1, w * h))


def read_stereo_calib(calib_path: Path) -> tuple[float, float]:
    """Return (f_u, baseline_m) from a KITTI calib file."""
    values: dict[str, list[float]] = {}
    with calib_path.open() as handle:
        for line in handle:
            if ":" not in line:
                continue
            key, _, rest = line.partition(":")
            values[key.strip()] = [float(x) for x in rest.split()]
    p2, p3 = values["P2"], values["P3"]
    f_u = p2[0]
    baseline = abs(p2[3] - p3[3]) / f_u
    return f_u, baseline


def paste_pil(img: Image.Image, patch: Image.Image, x0: int, y0: int) -> Image.Image:
    """Paste `patch` at (x0, y0), clipped to the canvas."""
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


def stereo_rects(
    face: FacePlacement, area_frac: float, calib_path: Path
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int], float] | None:
    """Left rect, right rect, and disparity. None if the face box is degenerate."""
    rect = patch_rect(face, area_frac)
    if rect is None:
        return None
    f_u, baseline = read_stereo_calib(calib_path)
    disp = disparity_px(f_u, baseline, face.depth_m)
    x0, y0, w, h = rect
    return rect, (right_x0(x0, disp), y0, w, h), disp


def filter_visible_frames(
    frames: list[FacePlacement],
    images_root: Path,
    area_frac: float,
    min_visible_frac: float,
) -> tuple[list[FacePlacement], list[tuple[str, float, float]]]:
    """Drop frames whose patch does not land on both stereo images."""
    kept: list[FacePlacement] = []
    dropped: list[tuple[str, float, float]] = []
    thresh = max(0.0, min_visible_frac)
    for face in frames:
        calib_path = images_root / "calib" / f"{face.frame}.txt"
        placed = stereo_rects(face, area_frac, calib_path)
        if placed is None:
            dropped.append((face.frame, 0.0, 0.0))
            continue
        left_rect, right_rect, _ = placed
        left_frac, right_frac = visible_frac(left_rect), visible_frac(right_rect)
        if min(left_frac, right_frac) <= thresh:
            dropped.append((face.frame, left_frac, right_frac))
        else:
            kept.append(face)
    return kept, dropped
