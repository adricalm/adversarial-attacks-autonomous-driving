#!/usr/bin/env python3
"""Write KITTI calib/ files for a recorded AWSIM dataset.

WHY THIS IS A SEPARATE STEP
The calibration of this rig is constant: the cameras never move relative to
each other, so every frame shares one calib file. Arka's 213-frame dataset
confirms it -- all 213 of his calib/*.txt are byte-identical
(md5 a121be1bc3f936546bfe5ab06fcc3b29). Recording it per frame would be a
thousand copies of the same 1576 bytes, so we stamp it out afterwards instead.

WHAT ACTUALLY MATTERS
DSGN's Calibration.fromfile parses P0..P3, R0_rect and Tr_velo_to_cam, so all
those lines must be present or loading fails. But optimize_patch.py only reads
P (via calib.P, calib.f_u) and derives the baseline from P2[0,3] - P3[0,3].
R0_rect and Tr_velo_to_cam are inert for the patch pipeline. That matters
because Arka's copies of those two are wrong for this rig -- R0_rect and
Tr_imu_to_velo are verbatim real-KITTI values, and his Tr_velo_to_cam is yawed
90 degrees from the true lidar->camera transform. None of it affects a
stereo-only pipeline, so the "arka" profile keeps his bytes exactly and stays
comparable with the existing finetuned checkpoints.

P2/P3 are correct and describe our rig:
  fx=960.0  fy=959.390808  cx=960.5  cy=540.5  at 1920x1080
  P2 Tx=+259.2 -> left  centre x = -Tx/fx = -0.27 m
  P3 Tx=-259.2 -> right centre x = +0.27 m      => baseline 0.54 m
which is exactly what StereoMod builds.

Usage:
  python3 write_calib.py dsgn/datasets/adria/my_run
  python3 write_calib.py dsgn/datasets/adria/my_run --check-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

# Byte-for-byte reproduction of Arka's calib file, including the two trailing
# blank lines. Verified against md5 ARKA_MD5 at runtime.
ARKA_CALIB = (
    "P0: 9.6000000e+02 0.000000000000e+00 9.605000000e+02 0.000000000000e+00"
    " 0.000000000000e+00 9.593900000000e+02 5.405000000000e+02 0.000000000000e+00"
    " 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00\n"
    "P1: 9.6000000e+02 0.000000000000e+00 9.605000000e+02 0.000000000000e+00"
    " 0.000000000000e+00 9.593900000000e+02 5.405000000000e+02 0.000000000000e+00"
    " 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00\n"
    "P2: 9.6000000e+02 0.000000000000e+00 9.605000000e+02 2.5920001220703125e+02"
    " 0.000000000000e+00 9.59390808105468e+02 5.405000000000e+02 0.000000000000e+00"
    " 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00\n"
    "P3: 9.6000000e+02 0.000000000000e+00 9.605000000e+02 -2.5920001220703125e+02"
    " 0.000000000000e+00 9.59390808105468e+02 5.405000000000e+02 0.000000000000e+00"
    " 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00\n"
    "R0_rect: 9.999128000000e-01 1.009263000000e-02 -8.511932000000e-03"
    " -1.012729000000e-02 9.999406000000e-01 -4.037671000000e-03"
    " 8.470675000000e-03 4.123522000000e-03 9.999556000000e-01\n"
    "Tr_velo_to_cam: -1.000000000e+00 0.00000000000e+00 -0.004000000000e+00"
    " 0.000000000000e+00 0.004000000000e+00 0.000000000000e+00 -1.000000000e+00"
    " -0.050000000e-01 0.00000000000e+00 -1.000000000e+00 0.000000000000e+00"
    " -0.100000000e+00\n"
    "Tr_imu_to_velo: 9.999976000000e-01 7.553071000000e-04 -2.035826000000e-03"
    " -8.086759000000e-01 -7.854027000000e-04 9.998898000000e-01 -1.482298000000e-02"
    " 3.195559000000e-01 2.024406000000e-03 1.482454000000e-02 9.998881000000e-01"
    " -7.997231000000e-01\n"
    "\n\n"
)

ARKA_MD5 = "a121be1bc3f936546bfe5ab06fcc3b29"

# What the sim must report for the arka calib to describe it truthfully.
EXPECTED = {"fx": 960.0, "fy": 959.3908081054688, "cx": 960.5, "cy": 540.5,
            "width": 1920, "height": 1080}
TOL = 1e-3


def check_self() -> None:
    got = hashlib.md5(ARKA_CALIB.encode()).hexdigest()
    if got != ARKA_MD5:
        raise SystemExit(f"internal error: calib md5 {got} != {ARKA_MD5}")


def check_camera_info(run_json: Path) -> list[str]:
    """Compare recorded camera_info against what the calib text claims."""
    problems = []
    if not run_json.is_file():
        return [f"no {run_json}; skipping camera_info cross-check"]
    meta = json.loads(run_json.read_text())
    cams = meta.get("camera_info") or {}
    if not cams:
        return ["run.json has no camera_info; skipping cross-check"]
    for side, ci in cams.items():
        k = ci["K"]
        got = {"fx": k[0], "fy": k[4], "cx": k[2], "cy": k[5],
               "width": ci["width"], "height": ci["height"]}
        for key, want in EXPECTED.items():
            if abs(got[key] - want) > TOL:
                problems.append(f"{side} camera {key}={got[key]} but calib assumes {want}")
    return problems


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", type=Path, help="Dataset root (containing image_2/)")
    p.add_argument("--check-only", action="store_true",
                   help="Validate against recorded camera_info, write nothing")
    p.add_argument("--force", action="store_true", help="Overwrite existing calib/")
    args = p.parse_args(argv)

    check_self()

    images = args.dataset / "image_2"
    if not images.is_dir():
        print(f"error: {images} not found", file=sys.stderr)
        return 1
    frames = sorted(f.stem for f in images.glob("*.png"))
    if not frames:
        print(f"error: no frames in {images}", file=sys.stderr)
        return 1

    problems = check_camera_info(args.dataset / "meta" / "run.json")
    for msg in problems:
        print(f"  note: {msg}")

    hard = [m for m in problems if "skipping" not in m]
    if hard:
        print("\nerror: recorded camera_info disagrees with the calib constants above.",
              file=sys.stderr)
        print("The rig changed; do not silently write a stale calib.", file=sys.stderr)
        return 1

    if args.check_only:
        print(f"ok: {len(frames)} frames, camera_info matches the arka calib")
        return 0

    out = args.dataset / "calib"
    if out.exists() and not args.force and any(out.iterdir()):
        print(f"error: {out} exists and is not empty (use --force)", file=sys.stderr)
        return 1
    out.mkdir(exist_ok=True)
    for frame in frames:
        (out / f"{frame}.txt").write_text(ARKA_CALIB)

    print(f"wrote {len(frames)} calib files to {out}")
    print(f"  md5 {ARKA_MD5} (identical to dsgn/datasets/arka/.../calib)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
