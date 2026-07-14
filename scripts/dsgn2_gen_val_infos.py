#!/usr/bin/env python3
"""Generate kitti_infos_val.pkl for unlabeled AWSIM frames (inference only)."""
import pickle
import sys
from pathlib import Path

import yaml
from easydict import EasyDict

from pcdet.datasets.kitti.lidar_kitti_dataset import LiDARKittiDataset


def main() -> int:
    workdir = Path.cwd()
    cfg_path = workdir / "configs/lidar/dataset_configs/kitti_dataset.yaml"
    data_path = workdir / "data/kitti"
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else data_path / "kitti_infos_val.pkl"

    if not cfg_path.exists():
        print(f"error: config not found: {cfg_path}", file=sys.stderr)
        return 1
    if not data_path.exists():
        print(f"error: data root not found: {data_path}", file=sys.stderr)
        return 1

    dataset_cfg = EasyDict(yaml.load(open(cfg_path), Loader=yaml.FullLoader))
    dataset = LiDARKittiDataset(
        dataset_cfg=dataset_cfg,
        class_names=["Car", "Pedestrian", "Cyclist"],
        root_path=data_path,
        training=False,
    )
    dataset.set_split("val")
    infos = dataset.get_infos(num_workers=2, has_label=False, count_inside_pts=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(infos, f)
    print(f"Saved {out_path} ({len(infos)} scenes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
