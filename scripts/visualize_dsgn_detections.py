#!/usr/bin/env python3
"""Draw DSGN KITTI detections on AWSIM left images and save PNGs.

Based on external/DSGN_custom/tools/plot_BB3D_awsim.py (Arka), but writes files
instead of cv2.imshow so it works over SSH.

Usage:
  python scripts/visualize_dsgn_detections.py \\
    --images dsgn/datasets/arka/dsgn_awsim/testing_offline/image_2 \\
    --calib dsgn/datasets/arka/dsgn_awsim/testing_offline/calib \\
    --detections dsgn/detections/adria/finetune_60_val \\
    --frames 000010,000099,000105 \\
    --label finetune_60 \\
    --output dsgn/datasets/adria/dsgn_awsim/detection_previews
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


def compute_box_3d(dim, loc, ry) -> np.ndarray:
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


# DSGN AWSIM loader downsamples 1920×1080 → 960×540; written 2D bboxes are in
# that half-res pixel space. Images + P2 calib used for viz are full-res.
INFER_HW = (540, 960)  # (H, W) matching KITTILoader_* downscale_factor=0.5


def annotate_frame(
    img,
    detections: list[dict],
    p2: np.ndarray,
    min_score: float | None,
    only_cars: bool,
) -> np.ndarray:
    out = img.copy()
    img_h, img_w = out.shape[:2]
    sx = img_w / INFER_HW[1]
    sy = img_h / INFER_HW[0]
    kept = 0
    for det in detections:
        if only_cars and det["type"].lower() != "car":
            continue
        score = det["score"]
        if min_score is not None and (score is None or score < min_score):
            continue

        # 2D bbox is in inference (half-res) coords; scale to display image.
        x1, y1, x2, y2 = det["bbox"]
        x1, x2 = int(round(x1 * sx)), int(round(x2 * sx))
        y1, y2 = int(round(y1 * sy)), int(round(y2 * sy))
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 1)

        # 3D→2D uses full-res P2 on the full-res image — already correct.
        corners_3d = compute_box_3d(det["dimensions"], det["location"], det["rotation_y"])
        corners_2d = project_to_image(corners_3d, p2)
        draw_projected_box3d(out, corners_2d, color=(0, 255, 0), thickness=2)

        # Anchor text to the projected 3D box (not the 2D rect) — the stored 2D
        # bbox can still disagree slightly with the 3D projection after scaling.
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
        f"detections shown: {kept}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize DSGN KITTI detections on AWSIM images")
    parser.add_argument("--images", required=True, help="Path to image_2/")
    parser.add_argument("--calib", required=True, help="Path to calib/")
    parser.add_argument("--detections", required=True, help="Path to awsim_output_* folder")
    parser.add_argument("--frames", default=None, help="Comma-separated frame ids, e.g. 000010,000099")
    parser.add_argument("--output", required=True, help="Output directory for PNGs")
    parser.add_argument("--label", default="_", help="Prefix for output filenames")
    parser.add_argument("--min-score", type=float, default=None, help="Optional score filter")
    parser.add_argument("--all-classes", action="store_true", help="Draw non-car classes too")
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
        annotated = annotate_frame(img, detections, p2, args.min_score, only_cars)

        out_path = out_dir / f"{frame}_{args.label}.png"
        cv2.imwrite(str(out_path), annotated)
        print(f"wrote {out_path} ({len(detections)} lines in txt)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
