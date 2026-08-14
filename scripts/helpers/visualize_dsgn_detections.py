#!/usr/bin/env python3
"""Draw DSGN / KITTI-format 3D boxes on AWSIM left images and save PNGs.

Two box conventions (see --box-convention):

  kitti  — standard KITTI label_2 / official DSGN KITTI checkpoints
           fields 8,9,10 = height, width, length
           location = bottom-center of the box (y down in camera)

  awsim  — Arka's AWSIM plotter (tools/plot_BB3D_awsim.py)
           same three numbers, but mapped differently to axes:
             field8 → length (object forward), field9 → height, field10 → width
           location = box center (± half-extents)
           Use for: awsim_output_offline, Arka AWSIM checkpoints, AWSIM label_2

Usage:
  # Official KITTI finetune_48 detections
  python scripts/helpers/visualize_dsgn_detections.py \\
    --box-convention kitti \\
    --images .../image_2 --calib .../calib --detections .../finetune_48_... \\
    --frames 000105 --output /tmp/viz --label kitti

  # Arka offline / AWSIM GT
  python scripts/helpers/visualize_dsgn_detections.py \\
    --box-convention awsim --full-res-bbox \\
    --images .../image_2 --calib .../calib \\
    --detections src/dsgn_offline/resource/awsim_output_offline \\
    --frames 000200 --output /tmp/viz --label awsim
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def read_kitti_detection(filename: Path) -> list[dict]:
    detections = []
    if not filename.is_file() or filename.stat().st_size == 0:
        return detections
    with filename.open() as f:
        for line in f:
            fields = line.strip().split()
            if len(fields) < 15:
                continue
            detections.append(
                {
                    "type": fields[0],
                    "bbox": [float(x) for x in fields[4:8]],
                    "dimensions": [float(x) for x in fields[8:11]],
                    "location": [float(x) for x in fields[11:14]],
                    "rotation_y": float(fields[14]),
                    "score": float(fields[15]) if len(fields) > 15 else None,
                }
            )
    return detections


def read_kitti_calib(calib_file: Path) -> np.ndarray:
    with calib_file.open() as f:
        for line in f:
            if line.startswith("P2:"):
                return np.array([float(x) for x in line.strip().split()[1:]]).reshape(3, 4)
    raise ValueError(f"P2 matrix not found in {calib_file}")


def compute_box_3d_kitti(dim, loc, ry) -> np.ndarray:
    """KITTI: dim=(h,w,l), loc=bottom-center, camera y points down."""
    h, w, l = dim
    x, y, z = loc
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    corners = np.array([x_corners, y_corners, z_corners])
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
    """Arka AWSIM plotter: same file fields, different axis mapping + center loc.

    Matches external/DSGN_custom/tools/plot_BB3D_awsim.py exactly:
      stored (a,b,c) = fields 8,9,10
      object X (lateral)  = ±c/2
      object Y (vertical) = ±b/2   ← so field9 is height
      object Z (length)   = ±a/2   ← so field8 is length
      location = center of the box
    """
    a, b, c = dim  # file fields 8,9,10 (not KITTI h,w,l meanings)
    x, y, z = loc
    x_corners = [c / 2, c / 2, -c / 2, -c / 2, c / 2, c / 2, -c / 2, -c / 2]
    y_corners = [b / 2, -b / 2, -b / 2, b / 2, b / 2, -b / 2, -b / 2, b / 2]
    z_corners = [a / 2, a / 2, a / 2, a / 2, -a / 2, -a / 2, -a / 2, -a / 2]
    corners = np.array([x_corners, y_corners, z_corners])
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
    n = pts_3d.shape[1]
    pts_3d_hom = np.vstack((pts_3d, np.ones((1, n))))
    pts_2d = p @ pts_3d_hom
    pts_2d[:2] /= pts_2d[2]
    return pts_2d[:2]


def draw_projected_box3d(img, qs, color=(0, 255, 0), thickness=2):
    qs = qs.astype(np.int32).T
    for k in range(4):
        i, j = k, (k + 1) % 4
        cv2.line(img, tuple(qs[i]), tuple(qs[j]), color, thickness)
        cv2.line(img, tuple(qs[i + 4]), tuple(qs[j + 4]), color, thickness)
        cv2.line(img, tuple(qs[i]), tuple(qs[i + 4]), color, thickness)
    return img


INFER_HW = (540, 960)  # half-res DSGN inference; use --full-res-bbox for full-res labels


def annotate_frame(
    img,
    detections: list[dict],
    p2: np.ndarray,
    min_score: float | None,
    only_cars: bool,
    bbox_full_res: bool = False,
    box_convention: str = "kitti",
) -> np.ndarray:
    out = img.copy()
    img_h, img_w = out.shape[:2]
    if bbox_full_res:
        sx = sy = 1.0
    else:
        sx = img_w / INFER_HW[1]
        sy = img_h / INFER_HW[0]
    kept = 0
    for det in detections:
        if only_cars and det["type"].lower() != "car":
            continue
        score = det["score"]
        if min_score is not None and (score is None or score < min_score):
            continue

        x1, y1, x2, y2 = det["bbox"]
        x1, x2 = int(round(x1 * sx)), int(round(x2 * sx))
        y1, y2 = int(round(y1 * sy)), int(round(y2 * sy))
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 1)

        corners_3d = compute_box_3d(
            det["dimensions"], det["location"], det["rotation_y"], box_convention
        )
        corners_2d = project_to_image(corners_3d, p2)
        draw_projected_box3d(out, corners_2d, color=(0, 255, 0), thickness=2)

        label = det["type"]
        if score is not None:
            label = f"{label} {score:.2f}"
        tx = int(round(float(corners_2d[0].mean())))
        ty = int(round(float(corners_2d[1].min()))) - 8
        cv2.putText(
            out,
            label,
            (max(tx, 0), max(ty, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        kept += 1

    cv2.putText(
        out,
        f"detections shown: {kept} [{box_convention}]",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Visualize DSGN detections (KITTI or Arka-AWSIM box convention)"
    )
    parser.add_argument("--images", required=True, help="Path to image_2/")
    parser.add_argument("--calib", required=True, help="Path to calib/")
    parser.add_argument("--detections", required=True, help="Folder of KITTI-format .txt")
    parser.add_argument("--frames", default=None, help="Comma-separated frame ids")
    parser.add_argument("--output", required=True, help="Output directory for PNGs")
    parser.add_argument("--label", default="_", help="Prefix for output filenames")
    parser.add_argument("--min-score", type=float, default=None, help="Optional score filter")
    parser.add_argument("--all-classes", action="store_true", help="Draw non-car classes too")
    parser.add_argument(
        "--full-res-bbox",
        action="store_true",
        help="2D bboxes already in image pixel space (training label_2 / some offline dumps)",
    )
    parser.add_argument(
        "--box-convention",
        choices=("kitti", "awsim"),
        default="kitti",
        help="kitti: official KITTI / finetune_48. awsim: Arka plot_BB3D_awsim / offline / AWSIM GT",
    )
    args = parser.parse_args()

    image_dir = Path(args.images)
    calib_dir = Path(args.calib)
    det_dir = Path(args.detections)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.frames is None:
        frames = sorted(p.stem for p in det_dir.glob("*.txt"))
    else:
        frames = [f.strip() for f in args.frames.split(",") if f.strip()]
    only_cars = not args.all_classes

    for frame in frames:
        img_path = image_dir / f"{frame}.png"
        calib_path = calib_dir / f"{frame}.txt"
        det_path = det_dir / f"{frame}.txt"
        if not img_path.is_file():
            raise SystemExit(f"missing image: {img_path}")

        img = cv2.imread(str(img_path))
        if img is None:
            raise SystemExit(f"failed to read image: {img_path}")

        p2 = read_kitti_calib(calib_path)
        detections = read_kitti_detection(det_path)
        annotated = annotate_frame(
            img,
            detections,
            p2,
            args.min_score,
            only_cars,
            bbox_full_res=args.full_res_bbox,
            box_convention=args.box_convention,
        )

        out_path = out_dir / f"{frame}_{args.label}.png"
        cv2.imwrite(str(out_path), annotated)
        print(f"wrote {out_path} ({len(detections)} lines, convention={args.box_convention})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
