#!/usr/bin/env python3
"""Merge recordings into one patch_train view (images + train/val CSVs)."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Source:
    name: str
    dataset: Path
    csv_path: Path
    split: str


def parse_sources(raw: list[list[str]] | None, split: str) -> list[Source]:
    return [
        Source(name=name, dataset=Path(dataset), csv_path=Path(csv_path), split=split)
        for name, dataset, csv_path in (raw or [])
    ]


def load_rows(source: Source) -> tuple[list[dict[str, str]], list[str]]:
    if not source.dataset.is_dir():
        raise FileNotFoundError(f"{source.name}: dataset not found: {source.dataset}")
    if not source.csv_path.is_file():
        raise FileNotFoundError(f"{source.name}: CSV not found: {source.csv_path}")

    with source.csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "frame" not in reader.fieldnames:
            raise ValueError(f"{source.name}: CSV must contain a frame column")
        rows = list(reader)
        fieldnames = list(reader.fieldnames)

    seen: set[str] = set()
    for row in rows:
        frame = f"{int(row['frame']):06d}"
        if frame in seen:
            raise ValueError(f"{source.name}: duplicate CSV frame {frame}")
        seen.add(frame)
        for subdir, suffix in (("image_2", ".png"), ("image_3", ".png"), ("calib", ".txt")):
            path = source.dataset / subdir / f"{frame}{suffix}"
            if not path.is_file():
                raise FileNotFoundError(f"{source.name}: missing {path}")
    return rows, fieldnames


def materialize(src: Path, dst: Path, mode: str) -> None:
    if mode == "symlink":
        dst.symlink_to(src.resolve())
    elif mode == "hardlink":
        os.link(src, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"unknown link mode: {mode}")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--train-source",
        nargs=3,
        action="append",
        metavar=("NAME", "DATASET", "CSV"),
        required=True,
    )
    parser.add_argument(
        "--val-source",
        nargs=3,
        action="append",
        metavar=("NAME", "DATASET", "CSV"),
        required=True,
    )
    parser.add_argument(
        "--link-mode",
        choices=("symlink", "hardlink", "copy"),
        default="symlink",
        help="How to materialize the combined image tree (default: symlink)",
    )
    args = parser.parse_args()

    sources = parse_sources(args.train_source, "train") + parse_sources(
        args.val_source, "val"
    )
    names = [source.name for source in sources]
    if len(names) != len(set(names)):
        print("error: source names must be unique", file=sys.stderr)
        return 1

    generated = [
        args.out / "dataset",
        args.out / "train.csv",
        args.out / "val.csv",
    ]
    if any(path.exists() for path in generated):
        print(
            f"error: generated output already exists under {args.out}; "
            "choose a new --out or remove the previous generated dataset",
            file=sys.stderr,
        )
        return 1

    loaded: list[tuple[Source, list[dict[str, str]]]] = []
    fieldnames: list[str] = ["frame"]
    try:
        for source in sources:
            rows, source_fields = load_rows(source)
            loaded.append((source, rows))
            for field in source_fields:
                if field not in fieldnames:
                    fieldnames.append(field)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    dataset_root = args.out / "dataset"
    for subdir in ("image_2", "image_3", "calib"):
        (dataset_root / subdir).mkdir(parents=True, exist_ok=False)

    output_rows: dict[str, list[dict[str, str]]] = {"train": [], "val": []}
    next_id = 0
    for source, rows in loaded:
        for row in rows:
            source_frame = f"{int(row['frame']):06d}"
            combined_frame = f"{next_id:06d}"
            next_id += 1

            for subdir, suffix in (("image_2", ".png"), ("image_3", ".png"), ("calib", ".txt")):
                src = source.dataset / subdir / f"{source_frame}{suffix}"
                dst = dataset_root / subdir / f"{combined_frame}{suffix}"
                materialize(src, dst, args.link_mode)

            rewritten = dict(row)
            rewritten["frame"] = combined_frame
            output_rows[source.split].append(rewritten)

    write_csv(args.out / "train.csv", fieldnames, output_rows["train"])
    write_csv(args.out / "val.csv", fieldnames, output_rows["val"])

    total = len(output_rows["train"]) + len(output_rows["val"])
    print(f"built {dataset_root} using {args.link_mode}s")
    print(f"  train rows: {len(output_rows['train'])}")
    print(f"  val rows:   {len(output_rows['val'])}")
    print(f"  total:      {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
