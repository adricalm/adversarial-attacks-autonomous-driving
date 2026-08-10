#!/usr/bin/env python3
"""Measure whether localized clean targets survive patched DSGN inference.

The clean localization CSV supplies the target vehicle's camera-frame BEV
position. For each frame, this script searches the patched KITTI detections for
a Car within `--match-radius`. Missing/empty detection files count as no target
detection and are reported separately.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def read_patched_cars(path: Path) -> list[tuple[float, float, float]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    cars: list[tuple[float, float, float]] = []
    with path.open() as handle:
        for line in handle:
            fields = line.split()
            if len(fields) < 15 or fields[0].lower() != "car":
                continue
            loc_x = float(fields[11])
            loc_z = float(fields[13])
            score = float(fields[15]) if len(fields) > 15 else 1.0
            cars.append((loc_x, loc_z, score))
    return cars


def depth_label(depth: float) -> str:
    if depth < 5:
        return "<5m"
    if depth < 10:
        return "5-10m"
    if depth < 15:
        return "10-15m"
    if depth < 20:
        return "15-20m"
    if depth < 30:
        return "20-30m"
    return ">=30m"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-csv", type=Path, required=True)
    parser.add_argument("--patched-detections", type=Path, required=True)
    parser.add_argument("--match-radius", type=float, default=2.0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows: list[dict[str, str | int | float]] = []
    with args.clean_csv.open(newline="") as handle:
        for clean in csv.DictReader(handle):
            frame = f"{int(clean['frame']):06d}"
            target_x = float(clean["loc_x"])
            target_z = float(clean["loc_z"])
            det_path = args.patched_detections / f"{frame}.txt"
            cars = read_patched_cars(det_path)
            matched = [
                (score, math.hypot(loc_x - target_x, loc_z - target_z))
                for loc_x, loc_z, score in cars
                if math.hypot(loc_x - target_x, loc_z - target_z) <= args.match_radius
            ]
            detected = bool(matched)
            rows.append(
                {
                    "frame": frame,
                    "depth_m": float(clean["depth_m"]),
                    "depth_bin": depth_label(float(clean["depth_m"])),
                    "clean_score": float(clean.get("score") or 0.0),
                    "target_detected": int(detected),
                    "target_gone": int(not detected),
                    "patched_score": max((score for score, _ in matched), default=0.0),
                    "closest_match_m": min((distance for _, distance in matched), default=-1.0),
                    "patched_car_count": len(cars),
                    "detection_file_missing": int(not det_path.is_file()),
                }
            )

    total = len(rows)
    gone = sum(int(row["target_gone"]) for row in rows)
    missing = sum(int(row["detection_file_missing"]) for row in rows)
    print(
        f"target gone: {gone}/{total} ({gone / max(total, 1):.1%}); "
        f"target remains: {total - gone}/{total}"
    )
    print(f"missing patched detection files (counted as no detections): {missing}")
    for label in ("<5m", "5-10m", "10-15m", "15-20m", "20-30m", ">=30m"):
        subset = [row for row in rows if row["depth_bin"] == label]
        if not subset:
            continue
        subset_gone = sum(int(row["target_gone"]) for row in subset)
        print(
            f"  {label:>7}: {subset_gone}/{len(subset)} "
            f"({subset_gone / len(subset):.1%}) gone"
        )

    out = args.out or (args.patched_detections / "suppression_results.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "frame",
        "depth_m",
        "depth_bin",
        "clean_score",
        "target_detected",
        "target_gone",
        "patched_score",
        "closest_match_m",
        "patched_car_count",
        "detection_file_missing",
    ]
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
