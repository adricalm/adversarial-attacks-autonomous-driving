#!/usr/bin/env python3
"""Convert AWSIM label_2 files to KITTI convention for DSGN finetuning.

AWSIM (Arka plotter / training label_2):
  fields 8,9,10 = length, height, width
  y = box center
  ry = 0 means length along camera Z

KITTI (Object3d / finetune_48 / DSGN training):
  fields 8,9,10 = height, width, length
  y = bottom-center
  ry = 0 means length along camera X

Validated remap (see label_convention_probe):
  h,w,l = f9, f10, f8
  y     = y_awsim + h/2
  ry    = wrap(ry_awsim - pi/2)
  alpha = wrap(ry - atan2(x, z))

Usage (host):
  python scripts/dsgn_transform_label.py
  python scripts/dsgn_transform_label.py \\
    --labels_path <awsim_format_label_2_dir> \\
    --save_path  ~/summer26/dsgn/datasets/adria/training_kitti_labels/label_2

Note (Jul 2026): Arka's training/ tree was removed from this host to free space.
  Defaults below may not exist anymore; pass --labels_path if you still have a source.
  Training DATA_PATH is always adria/training_kitti_labels (see notes/DSGN_AWSIM_FINDINGS.md).
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

FRAME_RE = re.compile(r"^\d{6}\.txt$")


def wrap_pi(angle: float) -> float:
    """Wrap angle to (-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def transform_label_line(line: str) -> str:
    """Convert one AWSIM label line to KITTI. Pass through blank/short lines."""
    raw = line.rstrip("\n")
    if not raw.strip():
        return raw

    parts = raw.split()
    if len(parts) < 15:
        raise ValueError(f"expected >=15 fields, got {len(parts)}: {raw!r}")

    score = parts[15] if len(parts) > 15 else None
    f8, f9, f10 = float(parts[8]), float(parts[9]), float(parts[10])
    x, y, z = float(parts[11]), float(parts[12]), float(parts[13])
    ry_awsim = float(parts[14])

    h, w, l = f9, f10, f8
    y_kitti = y + h / 2.0
    ry_kitti = wrap_pi(ry_awsim - math.pi / 2.0)
    alpha = wrap_pi(ry_kitti - math.atan2(x, z))

    out = [
        parts[0],
        parts[1],
        parts[2],
        f"{alpha:.4f}",
        parts[4],
        parts[5],
        parts[6],
        parts[7],
        f"{h:.6f}",
        f"{w:.6f}",
        f"{l:.6f}",
        f"{x:.6f}",
        f"{y_kitti:.7f}",
        f"{z:.6f}",
        f"{ry_kitti:.8f}",
    ]
    if score is not None:
        out.append(score)
    return " ".join(out)


def transform_label_file(src: Path) -> str:
    lines_out: list[str] = []
    for line in src.read_text().splitlines():
        if not line.strip():
            continue
        lines_out.append(transform_label_line(line))
    return "\n".join(lines_out) + ("\n" if lines_out else "")


def validate_labels_dir(labels_path: Path) -> list[Path]:
    if not labels_path.is_dir():
        raise FileNotFoundError(f"labels_path is not a directory: {labels_path}")
    if labels_path.name != "label_2":
        raise ValueError(
            f"expected folder named label_2, got {labels_path.name!r} ({labels_path})"
        )

    files = sorted(labels_path.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"no .txt files in {labels_path}")

    bad_names = [p.name for p in files if not FRAME_RE.match(p.name)]
    if bad_names:
        raise ValueError(
            f"{len(bad_names)} files do not match ######.txt "
            f"(e.g. {bad_names[:3]})"
        )

    # Spot-check a few non-empty files for column count.
    checked = 0
    for path in files:
        text = path.read_text()
        if not text.strip():
            continue
        for line in text.splitlines():
            if not line.strip():
                continue
            n = len(line.split())
            if n < 15:
                raise ValueError(f"{path.name}: line has {n} fields (<15)")
        checked += 1
        if checked >= 5:
            break

    return files


def parse_args() -> argparse.Namespace:
    root = Path.home() / "summer26"
    parser = argparse.ArgumentParser(
        description="Transform AWSIM label_2 → KITTI label_2 for DSGN finetune"
    )
    parser.add_argument(
        "--labels_path",
        type=Path,
        default=root / "dsgn/datasets/arka/dsgn_awsim/training/label_2",
        help="Source AWSIM-format label_2 (may be absent on this host if Arka training/ was deleted)",
    )
    parser.add_argument(
        "--save_path",
        type=Path,
        default=root / "dsgn/datasets/adria/training_kitti_labels/label_2",
        help="Destination KITTI label_2 directory",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty save_path",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    labels_path: Path = args.labels_path.expanduser().resolve()
    save_path: Path = args.save_path.expanduser().resolve()

    try:
        files = validate_labels_dir(labels_path)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if save_path.exists() and any(save_path.iterdir()) and not args.overwrite:
        print(
            f"error: save_path is not empty: {save_path}\n"
            f"  pass --overwrite to replace existing .txt files",
            file=sys.stderr,
        )
        return 1

    # Refuse to write into Arka's tree.
    if "datasets/arka" in str(save_path):
        print(f"error: refusing to write under arka datasets: {save_path}", file=sys.stderr)
        return 1

    save_path.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    n_empty = 0
    n_lines = 0
    for src in files:
        text = transform_label_file(src)
        (save_path / src.name).write_text(text)
        if not text.strip():
            n_empty += 1
        else:
            n_ok += 1
            n_lines += sum(1 for line in text.splitlines() if line.strip())

    print(f"source:      {labels_path}")
    print(f"destination: {save_path}")
    print(f"files:       {len(files)} ({n_ok} non-empty, {n_empty} empty)")
    print(f"object lines:{n_lines}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
