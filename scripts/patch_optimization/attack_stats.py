#!/usr/bin/env python3
"""Compare baseline vs patched DSGN detection folders (suppression stats)."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

DEFAULT_DATASETS = [
    "test_frontal1",
    "test_frontal2",
    "test_frontal3",
    "test_frontal4",
    "test_frontal5",
    "moving_frontal1",
    "moving_frontal2",
    "moving_frontal3",
]
DEFAULT_PATCHES = ["face020", "face035", "face050"]
CAR = "Car"


def max_car_score(txt: Path) -> float:
    if not txt.is_file():
        return 0.0
    best = 0.0
    for line in txt.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 16 and parts[0] == CAR:
            best = max(best, float(parts[15]))
    return best


def list_datasets(baseline: Path) -> list[str]:
    return sorted(
        p.name for p in baseline.iterdir()
        if p.is_dir() and any(p.glob("*.txt"))
    )


def collect_stats(
    baseline: Path,
    patched_root: Path,
    datasets: list[str],
    patches: list[str],
    score_thresh: float,
) -> tuple[list[dict[str, str | int | float | bool]], list[dict[str, str | int | float]]]:
    per_frame: list[dict[str, str | int | float | bool]] = []
    summary: list[dict[str, str | int | float]] = []

    for ds in datasets:
        base_dir = baseline / ds
        if not base_dir.is_dir():
            print(f"warning: baseline missing for {ds}: {base_dir}", file=sys.stderr)
            continue

        frames = sorted(base_dir.glob("*.txt"), key=lambda p: p.stem)
        if not frames:
            print(f"warning: no detection txt in {base_dir}", file=sys.stderr)
            continue

        for patch in patches:
            patched_dir = patched_root / patch / ds
            baseline_det = suppressed = 0

            for frame_path in frames:
                frame = frame_path.stem
                b_score = max_car_score(frame_path)
                p_score = max_car_score(patched_dir / f"{frame}.txt")
                detected = b_score >= score_thresh
                is_suppressed = detected and p_score < score_thresh

                per_frame.append(
                    {
                        "dataset": ds,
                        "patch": patch,
                        "frame": frame,
                        "baseline_score": round(b_score, 4),
                        "patched_score": round(p_score, 4),
                        "baseline_detected": detected,
                        "suppressed": is_suppressed,
                    }
                )

                if detected:
                    baseline_det += 1
                    if is_suppressed:
                        suppressed += 1

            rate = suppressed / baseline_det if baseline_det else 0.0
            summary.append(
                {
                    "dataset": ds,
                    "patch": patch,
                    "frames": len(frames),
                    "baseline_det": baseline_det,
                    "patched_suppressed": suppressed,
                    "suppression_rate": round(rate, 4),
                }
            )

    return per_frame, summary


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def print_detail_table(summary: list[dict[str, str | int | float]]) -> None:
    print("dataset\tpatch\tframes\tbaseline_det\tpatched_suppressed\tsuppression_rate")
    for row in summary:
        print(
            f"{row['dataset']}\t{row['patch']}\t{row['frames']}\t"
            f"{row['baseline_det']}\t{row['patched_suppressed']}\t"
            f"{row['suppression_rate']:.1%}"
        )


def print_pivot_table(
    summary: list[dict[str, str | int | float]],
    datasets: list[str],
    patches: list[str],
) -> None:
    rates: dict[tuple[str, str], float] = {
        (row["dataset"], row["patch"]): float(row["suppression_rate"])
        for row in summary
    }

    header = "dataset\t" + "\t".join(patches)
    print(header)
    for ds in datasets:
        cells = [ds]
        for patch in patches:
            rate = rates.get((ds, patch))
            cells.append(f"{rate:.1%}" if rate is not None else "—")
        print("\t".join(cells))


def main() -> int:
    root = Path.home() / "summer26"
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=root / "dsgn/detections/adria/test_recordings_clean",
        help="Baseline (clean-image) detections root; one subfolder per dataset",
    )
    parser.add_argument(
        "--patched",
        type=Path,
        default=root / "dsgn/detections/adria/test_recordings_patched",
        help="Patched detections root; layout <patched>/<patch>/<dataset>/",
    )
    parser.add_argument(
        "--patches",
        nargs="+",
        default=DEFAULT_PATCHES,
        help="Patch size subfolder names under --patched (default: face020 face035 face050)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Dataset subfolder names; default: auto-discover from --baseline",
    )
    parser.add_argument(
        "--score-thresh",
        type=float,
        default=0.33,
        help="Car score threshold for detected / suppressed (default: 0.33)",
    )
    parser.add_argument(
        "--per-frame-csv",
        type=Path,
        default=None,
        help="Optional path for per-frame detail CSV",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Optional path for per-dataset/patch summary CSV",
    )
    parser.add_argument(
        "--no-pivot",
        action="store_true",
        help="Skip the pivoted suppression-rate table",
    )
    args = parser.parse_args()

    if not args.baseline.is_dir():
        print(f"error: baseline dir not found: {args.baseline}", file=sys.stderr)
        return 1
    if not args.patched.is_dir():
        print(f"error: patched dir not found: {args.patched}", file=sys.stderr)
        return 1

    datasets = args.datasets or list_datasets(args.baseline)
    if not datasets:
        print(f"error: no datasets found under {args.baseline}", file=sys.stderr)
        return 1

    per_frame, summary = collect_stats(
        args.baseline,
        args.patched,
        datasets,
        args.patches,
        args.score_thresh,
    )

    if not summary:
        print("error: no stats collected", file=sys.stderr)
        return 1

    print_detail_table(summary)
    if not args.no_pivot:
        print()
        print_pivot_table(summary, datasets, args.patches)

    if args.per_frame_csv is not None:
        write_csv(
            args.per_frame_csv,
            per_frame,
            [
                "dataset",
                "patch",
                "frame",
                "baseline_score",
                "patched_score",
                "baseline_detected",
                "suppressed",
            ],
        )
        print(f"\nper-frame CSV → {args.per_frame_csv}")

    if args.summary_csv is not None:
        write_csv(
            args.summary_csv,
            summary,
            [
                "dataset",
                "patch",
                "frames",
                "baseline_det",
                "patched_suppressed",
                "suppression_rate",
            ],
        )
        print(f"summary CSV → {args.summary_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
