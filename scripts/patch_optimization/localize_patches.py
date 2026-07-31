#!/usr/bin/env python3
"""Localize physically plausible rear-face patches from DSGN KITTI detections.

For each detection, project the 3D box, pick the vertical face closest to the
ego camera (smallest mean depth), and write a square patch (center + size)
that fits inside that face's 2D projection.

CSV is compatible with scripts/apply_stereo_patches.py:
  frame,center_x,center_y,size,depth_m,seed
Extra columns (ignored by apply): det_idx,score,face_w,face_h,x0,y0,x1,y1,loc_x,loc_z,ego_dist

Closest selection uses bird's-eye distance hypot(loc_x, loc_z), not rear-face
depth alone — so a near lead car wins over a far neighbor.

Note: apply_stereo_patches currently keeps one patch per frame (last CSV row
wins). Default --selection closest emits one row/frame so apply works as-is.
Use --selection all for multi-det CSVs (optimizer / future multi-patch apply).

Examples
--------
  # Closest car only → apply-ready CSV
  python3 scripts/patch_optimization/localize_patches.py \\
    --detections src/dsgn_offline/resource/awsim_output_offline_no_finetune \\
    --calib dsgn/datasets/arka/dsgn_awsim/testing_offline/calib \\
    --output dsgn/datasets/adria/1.patch_optimization/patches_localized.csv \\
    --box-convention kitti --selection closest

  # Preview overlays on a few frames
  python3 scripts/patch_optimization/localize_patches.py \\
    --detections src/dsgn_offline/resource/awsim_output_offline_no_finetune \\
    --calib dsgn/datasets/arka/dsgn_awsim/testing_offline/calib \\
    --images dsgn/datasets/arka/dsgn_awsim/testing_offline/image_2 \\
    --output dsgn/datasets/adria/1.patch_optimization/patches_localized.csv \\
    --preview-dir dsgn/datasets/adria/1.patch_optimization/preview \\
    --frames 100,150,200
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import cv2

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# Vertical faces as corner-index quads (order matches visualize_dsgn_detections).
# KITTI object frame: x=length, y=up(-), z=width. AWSIM: z=length, y=height, x=width.
FACES_KITTI = (
    (0, 1, 5, 4),  # +X length
    (3, 2, 6, 7),  # -X length
    (0, 3, 7, 4),  # +Z width
    (1, 2, 6, 5),  # -Z width
)
FACES_AWSIM = (
    (0, 1, 2, 3),  # +Z length
    (4, 5, 6, 7),  # -Z length
    (0, 1, 5, 4),  # +X width
    (3, 2, 6, 7),  # -X width
)

IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080


@dataclass
class Detection:
    type: str
    bbox: tuple[float, float, float, float]
    dimensions: tuple[float, float, float]
    location: tuple[float, float, float]
    rotation_y: float
    score: float | None


@dataclass
class PatchRow:
    frame: str
    det_idx: int
    center_x: float
    center_y: float
    size: int
    depth_m: float
    seed: int
    score: float | None
    face_w: float
    face_h: float
    x0: float
    y0: float
    x1: float
    y1: float
    loc_x: float
    loc_z: float

    @property
    def ego_dist(self) -> float:
        """Bird's-eye distance from ego camera to box center."""
        return float(np.hypot(self.loc_x, self.loc_z))


def read_kitti_detection(path: Path) -> list[Detection]:
    dets: list[Detection] = []
    if not path.is_file() or path.stat().st_size == 0:
        return dets
    with path.open() as f:
        for line in f:
            fields = line.strip().split()
            if len(fields) < 15:
                continue
            dets.append(
                Detection(
                    type=fields[0],
                    bbox=tuple(float(x) for x in fields[4:8]),  # type: ignore[arg-type]
                    dimensions=tuple(float(x) for x in fields[8:11]),  # type: ignore[arg-type]
                    location=tuple(float(x) for x in fields[11:14]),  # type: ignore[arg-type]
                    rotation_y=float(fields[14]),
                    score=float(fields[15]) if len(fields) > 15 else None,
                )
            )
    return dets


def read_kitti_calib(path: Path) -> np.ndarray:
    with path.open() as f:
        for line in f:
            if line.startswith("P2:"):
                return np.array([float(x) for x in line.strip().split()[1:]]).reshape(3, 4)
    raise ValueError(f"P2 matrix not found in {path}")


def compute_box_3d_kitti(dim, loc, ry) -> np.ndarray:
    h, w, l = dim
    x, y, z = loc
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    corners = np.array([x_corners, y_corners, z_corners], dtype=np.float64)
    r = np.array(
        [
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)],
        ]
    )
    corners_3d = r @ corners
    corners_3d[0, :] += x
    corners_3d[1, :] += y
    corners_3d[2, :] += z
    return corners_3d


def compute_box_3d_awsim(dim, loc, ry) -> np.ndarray:
    a, b, c = dim
    x, y, z = loc
    x_corners = [c / 2, c / 2, -c / 2, -c / 2, c / 2, c / 2, -c / 2, -c / 2]
    y_corners = [b / 2, -b / 2, -b / 2, b / 2, b / 2, -b / 2, -b / 2, b / 2]
    z_corners = [a / 2, a / 2, a / 2, a / 2, -a / 2, -a / 2, -a / 2, -a / 2]
    corners = np.array([x_corners, y_corners, z_corners], dtype=np.float64)
    r = np.array(
        [
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)],
        ]
    )
    corners_3d = r @ corners
    corners_3d[0, :] += x
    corners_3d[1, :] += y
    corners_3d[2, :] += z
    return corners_3d


def compute_box_3d(dim, loc, ry, convention: str) -> np.ndarray:
    if convention == "kitti":
        return compute_box_3d_kitti(dim, loc, ry)
    if convention == "awsim":
        return compute_box_3d_awsim(dim, loc, ry)
    raise ValueError(f"unknown box convention: {convention}")


def project_to_image(pts_3d: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Project 3xN camera points → 2xN image pixels. Drops points with z<=0."""
    n = pts_3d.shape[1]
    pts_h = np.vstack((pts_3d, np.ones((1, n))))
    pts_2d = p @ pts_h
    z = pts_2d[2]
    valid = z > 1e-6
    out = np.full((2, n), np.nan, dtype=np.float64)
    out[0, valid] = pts_2d[0, valid] / z[valid]
    out[1, valid] = pts_2d[1, valid] / z[valid]
    return out


def parse_frame_list(spec: str | None, det_dir: Path) -> list[str]:
    if spec is None:
        return sorted(p.stem for p in det_dir.glob("*.txt"))
    frames: list[str] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part and part.replace("-", "").isdigit():
            a, b = part.split("-", 1)
            for i in range(int(a), int(b) + 1):
                frames.append(f"{i:06d}")
        else:
            frames.append(f"{int(part):06d}")
    return frames


def localize_rear_patch(
    det: Detection,
    p2: np.ndarray,
    convention: str,
    size_frac: float,
    min_size: int,
    max_size: int | None,
    image_w: int,
    image_h: int,
) -> tuple[float, float, int, float, float, float, float, float, float] | None:
    """Return (cx, cy, size, depth, face_w, face_h, x0, y0, x1, y1) or None."""
    corners_3d = compute_box_3d(det.dimensions, det.location, det.rotation_y, convention)
    faces = FACES_KITTI if convention == "kitti" else FACES_AWSIM

    best_idx = None
    best_depth = float("inf")
    for fi, idxs in enumerate(faces):
        depths = corners_3d[2, list(idxs)]
        if np.any(depths <= 0.1):
            continue
        mean_z = float(np.mean(depths))
        if mean_z < best_depth:
            best_depth = mean_z
            best_idx = fi

    if best_idx is None:
        return None

    idxs = list(faces[best_idx])
    face_3d = corners_3d[:, idxs]
    face_2d = project_to_image(face_3d, p2)
    if np.any(~np.isfinite(face_2d)):
        return None

    u_min, v_min = float(face_2d[0].min()), float(face_2d[1].min())
    u_max, v_max = float(face_2d[0].max()), float(face_2d[1].max())
    face_w = u_max - u_min
    face_h = v_max - v_min
    if face_w < 2 or face_h < 2:
        return None

    cx = 0.5 * (u_min + u_max)
    cy = 0.5 * (v_min + v_max)
    # Square that fits inside the face AABB.
    size = int(round(min(face_w, face_h) * size_frac))
    size = max(min_size, size)
    if max_size is not None:
        size = min(size, max_size)

    half = size / 2.0
    # Keep center inside image so apply's clip still leaves a visible patch.
    cx = float(np.clip(cx, half, image_w - half))
    cy = float(np.clip(cy, half, image_h - half))

    depth = float(np.mean(face_3d[2]))
    return cx, cy, size, depth, face_w, face_h, u_min, v_min, u_max, v_max


def select_detections(
    dets: list[Detection],
    min_score: float | None,
    min_loc_z: float | None,
    only_cars: bool,
) -> list[tuple[int, Detection]]:
    """Pre-filter by class/score/box-center depth. Closest pick happens later."""
    indexed: list[tuple[int, Detection]] = []
    for i, det in enumerate(dets):
        if only_cars and det.type.lower() != "car":
            continue
        if min_score is not None and (det.score is None or det.score < min_score):
            continue
        # Filter on box-center z (not rear-face depth): a lead car at z≈3.4 m
        # can have rear-face depth ≈2 m, which is still a valid attack target.
        if min_loc_z is not None and det.location[2] < min_loc_z:
            continue
        indexed.append((i, det))
    return indexed


def pick_rows(
    candidates: list[PatchRow],
    selection: str,
    max_depth: float | None,
    max_center_y: float | None,
) -> list[PatchRow]:
    kept: list[PatchRow] = []
    for r in candidates:
        if max_depth is not None and r.depth_m > max_depth:
            continue
        # Drop ego hood / A-pillar FPs whose patch center sits too low in the image.
        if max_center_y is not None and r.center_y > max_center_y:
            continue
        kept.append(r)
    if selection == "all":
        return kept
    if selection == "closest":
        if not kept:
            return []
        return [min(kept, key=lambda r: r.ego_dist)]
    raise ValueError(f"unknown selection: {selection}")


def write_csv(path: Path, rows: list[PatchRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "frame",
        "center_x",
        "center_y",
        "size",
        "depth_m",
        "seed",
        "det_idx",
        "score",
        "face_w",
        "face_h",
        "x0",
        "y0",
        "x1",
        "y1",
        "loc_x",
        "loc_z",
        "ego_dist",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "frame": r.frame,
                    "center_x": f"{r.center_x:.2f}",
                    "center_y": f"{r.center_y:.2f}",
                    "size": r.size,
                    "depth_m": f"{r.depth_m:.4f}",
                    "seed": r.seed,
                    "det_idx": r.det_idx,
                    "score": "" if r.score is None else f"{r.score:.6f}",
                    "face_w": f"{r.face_w:.2f}",
                    "face_h": f"{r.face_h:.2f}",
                    "x0": f"{r.x0:.2f}",
                    "y0": f"{r.y0:.2f}",
                    "x1": f"{r.x1:.2f}",
                    "y1": f"{r.y1:.2f}",
                    "loc_x": f"{r.loc_x:.4f}",
                    "loc_z": f"{r.loc_z:.4f}",
                    "ego_dist": f"{r.ego_dist:.4f}",
                }
            )


def draw_preview(
    image_path: Path,
    rows: list[PatchRow],
    dets: list[Detection],
    p2: np.ndarray,
    convention: str,
    out_path: Path,
) -> None:
    if not HAS_CV2:
        raise RuntimeError("cv2 required for --preview-dir")
    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"failed to read {image_path}")

    for det in dets:
        if det.type.lower() != "car":
            continue
        corners = compute_box_3d(det.dimensions, det.location, det.rotation_y, convention)
        pts = project_to_image(corners, p2)
        if np.any(~np.isfinite(pts)):
            continue
        qs = pts.astype(np.int32).T
        for k in range(4):
            i, j = k, (k + 1) % 4
            cv2.line(img, tuple(qs[i]), tuple(qs[j]), (0, 200, 0), 1)
            cv2.line(img, tuple(qs[i + 4]), tuple(qs[j + 4]), (0, 200, 0), 1)
            cv2.line(img, tuple(qs[i]), tuple(qs[i + 4]), (0, 200, 0), 1)

    for r in rows:
        half = r.size // 2
        x0 = int(round(r.center_x)) - half
        y0 = int(round(r.center_y)) - half
        x1, y1 = x0 + r.size, y0 + r.size
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 2)
        cv2.drawMarker(
            img,
            (int(round(r.center_x)), int(round(r.center_y))),
            (0, 255, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=12,
            thickness=1,
        )
        # Face AABB used for sizing.
        cv2.rectangle(
            img,
            (int(round(r.x0)), int(round(r.y0))),
            (int(round(r.x1)), int(round(r.y1))),
            (255, 128, 0),
            1,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--detections",
        type=Path,
        default=Path("src/dsgn_offline/resource/awsim_output_offline_no_finetune"),
        help="Folder of KITTI-format detection .txt files",
    )
    p.add_argument(
        "--calib",
        type=Path,
        default=Path("dsgn/datasets/arka/dsgn_awsim/testing_offline/calib"),
        help="Folder of KITTI calib .txt (needed to project 3D faces)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("dsgn/datasets/adria/1.patch_optimization/patches_localized.csv"),
        help="Output CSV path",
    )
    p.add_argument(
        "--box-convention",
        choices=("kitti", "awsim"),
        default="kitti",
        help="no_finetune / official KITTI ckpt → kitti; Arka AWSIM offline → awsim",
    )
    p.add_argument(
        "--selection",
        choices=("closest", "all"),
        default="closest",
        help="closest: one patch/frame (apply-ready); all: one row per detection",
    )
    p.add_argument("--min-score", type=float, default=None)
    p.add_argument(
        "--min-loc-z",
        type=float,
        default=2.0,
        help="ignore detections whose box-center z is below this (m)",
    )
    p.add_argument(
        "--max-depth",
        type=float,
        default=None,
        help="optional max rear-face depth (m)",
    )
    p.add_argument(
        "--max-center-y-frac",
        type=float,
        default=None,
        help="optional: drop patches with center_y above this fraction of image height "
        "(can reject truncated near cars; leave unset for closest-to-ego)",
    )
    p.add_argument("--all-classes", action="store_true")
    p.add_argument("--size-frac", type=float, default=0.5, help="square side = frac * min(face_w, face_h)")
    p.add_argument("--min-size", type=int, default=16)
    p.add_argument("--max-size", type=int, default=400, help="cap square side (near cars can be huge)")
    p.add_argument("--image-width", type=int, default=IMAGE_WIDTH)
    p.add_argument("--image-height", type=int, default=IMAGE_HEIGHT)
    p.add_argument("--frames", default=None, help="e.g. 100-150 or 100,150,200")
    p.add_argument(
        "--images",
        type=Path,
        default=None,
        help="image_2/ for optional preview overlays",
    )
    p.add_argument(
        "--preview-dir",
        type=Path,
        default=None,
        help="If set (and --images given), write per-frame preview PNGs",
    )
    args = p.parse_args()

    det_dir: Path = args.detections
    calib_dir: Path = args.calib
    if not det_dir.is_dir():
        print(f"error: detections dir not found: {det_dir}", file=sys.stderr)
        return 1
    if not calib_dir.is_dir():
        print(f"error: calib dir not found: {calib_dir}", file=sys.stderr)
        return 1

    frames = parse_frame_list(args.frames, det_dir)
    only_cars = not args.all_classes
    rows: list[PatchRow] = []
    skipped = 0
    empty = 0

    for frame in frames:
        det_path = det_dir / f"{frame}.txt"
        calib_path = calib_dir / f"{frame}.txt"
        dets = read_kitti_detection(det_path)
        if not dets:
            empty += 1
            continue
        if not calib_path.is_file():
            print(f"warning: missing calib for {frame}, skip", file=sys.stderr)
            skipped += 1
            continue

        p2 = read_kitti_calib(calib_path)
        candidates_det = select_detections(
            dets, args.min_score, args.min_loc_z, only_cars
        )
        candidates: list[PatchRow] = []
        for det_idx, det in candidates_det:
            loc = localize_rear_patch(
                det,
                p2,
                args.box_convention,
                args.size_frac,
                args.min_size,
                args.max_size,
                args.image_width,
                args.image_height,
            )
            if loc is None:
                skipped += 1
                continue
            cx, cy, size, depth, face_w, face_h, x0, y0, x1, y1 = loc
            candidates.append(
                PatchRow(
                    frame=frame,
                    det_idx=det_idx,
                    center_x=cx,
                    center_y=cy,
                    size=size,
                    depth_m=depth,
                    seed=int(frame) * 1000 + det_idx,
                    score=det.score,
                    face_w=face_w,
                    face_h=face_h,
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    loc_x=det.location[0],
                    loc_z=det.location[2],
                )
            )
        max_center_y = (
            None
            if args.max_center_y_frac is None
            else args.max_center_y_frac * args.image_height
        )
        frame_rows = pick_rows(
            candidates, args.selection, args.max_depth, max_center_y
        )
        rows.extend(frame_rows)

        if args.preview_dir is not None:
            if args.images is None:
                print("error: --preview-dir requires --images", file=sys.stderr)
                return 1
            img_path = args.images / f"{frame}.png"
            if img_path.is_file() and frame_rows:
                draw_preview(
                    img_path,
                    frame_rows,
                    dets,
                    p2,
                    args.box_convention,
                    args.preview_dir / f"{frame}_patch_loc.png",
                )

    write_csv(args.output, rows)
    print(f"Wrote {len(rows)} patches → {args.output}")
    print(
        f"  frames scanned={len(frames)} empty={empty} "
        f"selection={args.selection} convention={args.box_convention}"
    )
    if args.selection == "all":
        print(
            "  note: apply_stereo_patches keeps the last row per frame; "
            "use --selection closest for apply, or extend apply for multi-patch."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
