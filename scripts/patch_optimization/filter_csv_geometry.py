#!/usr/bin/env python3
"""Filter a localized-patch CSV by target geometry (depth and lateral angle).

Motivation
----------
Depth alone hides a confound in the AWSIM training set: its near cars are
mostly *oblique* (median 27 deg off-axis) while its far cars are head-on, so
"near" and "oblique" are entangled. The offline test set is the opposite --
its near cars are head-on (median 10 deg), which is the AEB geometry that
matters. Selecting on lateral angle as well as depth lets a run target one
geometry instead of averaging over both.

Lateral angle is atan(|loc_x| / loc_z): 0 deg is directly ahead of the ego,
90 deg is straight out to the side.

All columns are preserved, so the output drops into optimize_patch.py /
eval_patch.py unchanged.

Example
-------
  # The AEB-critical slice: close and genuinely ahead.
  python scripts/patch_optimization/filter_csv_geometry.py \\
    --csv dsgn/datasets/adria/testing_offline_patches_localized.csv \\
    --out dsgn/datasets/adria/testing_offline_headon_near.csv \\
    --max-depth 15 --max-angle 15
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def lateral_angle_deg(loc_x: float, loc_z: float) -> float:
    """Angle off the ego's forward axis, in degrees."""
    return math.degrees(math.atan2(abs(loc_x), max(loc_z, 0.1)))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--csv", type=Path, required=True, help="Input localized CSV")
    p.add_argument("--out", type=Path, required=True, help="Output filtered CSV")
    p.add_argument("--min-depth", type=float, default=0.0, help="Keep depth_m >= this")
    p.add_argument("--max-depth", type=float, default=float("inf"), help="Keep depth_m <= this")
    p.add_argument("--min-angle", type=float, default=0.0, help="Keep lateral angle >= this (deg)")
    p.add_argument(
        "--max-angle",
        type=float,
        default=180.0,
        help="Keep lateral angle <= this (deg). 15 selects near-head-on targets.",
    )
    p.add_argument(
        "--min-frame",
        type=int,
        default=None,
        help="Keep frame id >= this. These CSVs come from continuous video, so "
        "splitting by frame range holds out a whole driving event -- unlike a "
        "strided split, which puts near-identical adjacent frames on both sides.",
    )
    p.add_argument("--max-frame", type=int, default=None, help="Keep frame id <= this")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    with args.csv.open() as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not fieldnames:
        raise RuntimeError(f"no header in {args.csv}")

    kept = []
    for r in rows:
        depth = float(r["depth_m"])
        angle = lateral_angle_deg(float(r["loc_x"]), float(r["loc_z"]))
        frame = int(r["frame"])
        if not (args.min_depth <= depth <= args.max_depth):
            continue
        if not (args.min_angle <= angle <= args.max_angle):
            continue
        if args.min_frame is not None and frame < args.min_frame:
            continue
        if args.max_frame is not None and frame > args.max_frame:
            continue
        kept.append(r)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)

    depths = [float(r["depth_m"]) for r in kept]
    angles = [lateral_angle_deg(float(r["loc_x"]), float(r["loc_z"])) for r in kept]
    print(f"{args.csv} -> {args.out}")
    print(
        f"kept {len(kept)}/{len(rows)} rows  "
        f"(depth {args.min_depth}-{args.max_depth} m, angle {args.min_angle}-{args.max_angle} deg)"
    )
    if kept:
        depths_sorted = sorted(depths)
        print(
            f"  depth : min={min(depths):.1f} median={depths_sorted[len(depths)//2]:.1f} "
            f"max={max(depths):.1f} m"
        )
        angles_sorted = sorted(angles)
        print(
            f"  angle : min={min(angles):.1f} median={angles_sorted[len(angles)//2]:.1f} "
            f"max={max(angles):.1f} deg"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
