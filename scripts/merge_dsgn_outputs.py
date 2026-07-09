#!/usr/bin/env python3
"""Merge DSGN detection outputs: baseline for most frames, new inference only on patched range.

Use this when full re-inference was run with a newer PyTorch than Arka's training
environment — unpatched frames will have spurious extra detections if you take the
new folder wholesale.

Example:
  python3 scripts/merge_dsgn_outputs.py \\
    --baseline src/dsgn_offline/resource/awsim_output_offline \\
    --patched  dsgn/detections/adria/patched_100_135 \\
    --output   src/dsgn_offline/resource/awsim_output_patched_merged \\
    --frames   100-135
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

FRAME_RE = re.compile(r"^(\d{6})\.txt$")


def parse_frame_range(spec: str) -> list[str]:
    frames: list[str] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            start, end = int(a), int(b)
            for i in range(start, end + 1):
                frames.append(f"{i:06d}")
        else:
            frames.append(f"{int(part):06d}")
    return frames


def list_frames(folder: Path) -> list[str]:
    return sorted(
        m.group(1) for p in folder.glob("*.txt") if (m := FRAME_RE.match(p.name))
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baseline", required=True, help="Arka precomputed awsim_output_offline/")
    p.add_argument("--patched", required=True, help="new inference output folder")
    p.add_argument("--output", required=True, help="merged output folder")
    p.add_argument("--frames", required=True, help="frames to take from --patched, e.g. 100-135")
    args = p.parse_args()

    baseline = Path(args.baseline)
    patched = Path(args.patched)
    output = Path(args.output)
    replace = set(parse_frame_range(args.frames))

    if not baseline.is_dir():
        raise SystemExit(f"baseline not found: {baseline}")
    if not patched.is_dir():
        raise SystemExit(f"patched not found: {patched}")

    output.mkdir(parents=True, exist_ok=True)
    all_frames = sorted(set(list_frames(baseline)) | set(list_frames(patched)))

    from_baseline = 0
    from_patched = 0
    for frame in all_frames:
        name = f"{frame}.txt"
        dst = output / name
        if frame in replace:
            src = patched / name
            if not src.is_file():
                print(f"warning: missing patched {name}, keeping baseline")
                src = baseline / name
                from_baseline += 1
            else:
                from_patched += 1
        else:
            src = baseline / name
            from_baseline += 1
        if not src.is_file():
            print(f"warning: missing {name}")
            continue
        shutil.copy2(src, dst)

    print(f"Merged {len(all_frames)} frames → {output}")
    print(f"  baseline: {from_baseline}")
    print(f"  patched:  {from_patched} (frames {args.frames})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
