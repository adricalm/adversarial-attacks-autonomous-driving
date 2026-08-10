#!/usr/bin/env python3
"""Prepare recorded stereo subsets for DSGN inference.

The AWSIM stereo rig is fixed, so every paired frame receives a copy of one
known-good KITTI calibration file. A split file containing exactly the paired
left/right frame IDs is also written in each dataset root.

Velodyne data is deliberately ignored.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def frame_ids(path: Path) -> set[str]:
    return {p.stem for p in path.glob("*.png")}


def prepare_dataset(dataset: Path, template: Path, overwrite: bool) -> tuple[int, int, int]:
    left_dir = dataset / "image_2"
    right_dir = dataset / "image_3"
    if not left_dir.is_dir() or not right_dir.is_dir():
        raise FileNotFoundError(f"{dataset}: expected image_2/ and image_3/")

    left = frame_ids(left_dir)
    right = frame_ids(right_dir)
    paired = sorted(left & right)
    if not paired:
        raise RuntimeError(f"{dataset}: no paired stereo PNGs")

    missing_left = len(right - left)
    missing_right = len(left - right)
    calib_dir = dataset / "calib"
    calib_dir.mkdir(parents=True, exist_ok=True)

    for frame in paired:
        dst = calib_dir / f"{frame}.txt"
        if overwrite or not dst.exists():
            shutil.copy2(template, dst)

    split_path = dataset / "frames.txt"
    split_path.write_text("".join(f"{frame}\n" for frame in paired))
    return len(paired), missing_left, missing_right


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--template-calib",
        type=Path,
        required=True,
        help="Known-good KITTI calibration file for this fixed stereo rig",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        action="append",
        required=True,
        help="Recorded KITTI-style dataset root; repeat for each subset",
    )
    parser.add_argument(
        "--overwrite-calib",
        action="store_true",
        help="Replace calibration files that already exist",
    )
    args = parser.parse_args()

    if not args.template_calib.is_file():
        print(f"error: calibration template not found: {args.template_calib}", file=sys.stderr)
        return 1

    total = 0
    for dataset in args.dataset:
        try:
            paired, missing_left, missing_right = prepare_dataset(
                dataset, args.template_calib, args.overwrite_calib
            )
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        total += paired
        print(
            f"{dataset}: {paired} paired frames; "
            f"missing_left={missing_left} missing_right={missing_right}; "
            f"wrote calib/ and frames.txt"
        )

    print(f"prepared {len(args.dataset)} datasets ({total} paired frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
