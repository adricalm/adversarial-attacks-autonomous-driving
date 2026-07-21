# DSGN offline runbook

Copy-paste commands for replaying Arka's precomputed DSGN detections into Autoware.  
**You run everything** — this repo only provides scripts and the patched `dsgn_offline` node.

See also: [README DSGN overlay section](../README.md#dsgn-offline-overlay-optional).

---

## 0. Prerequisites

Before building or running `dsgn_offline`:

- [x] Autoware container running (`autoware_full_test`) with **canonical launch** from README (NDT bind-mount + `launch_rviz_adaptors:=true`)
- [x] **Extra volume mounts** on `docker run` (see Step 1)
- [x] AWSIM running, `/clock` has one publisher
- [ ] Localization initialized, route set, traffic-light bridge running
- [ ] Ego driving or ready to engage (`/autoware/state` → 5)

**Synchronization note:** `dsgn_offline` picks the nearest row in `path.txt` from current `/localization/pose_with_covariance`. Detections align only when ego `(x, y)` is close to Arka's recorded MGRS trajectory (~81377, 49917). Compare while driving:

```bash
# inside Docker
ros2 topic echo /localization/pose_with_covariance --once
head -3 /home/aw/ros2_ws/src/dsgn_offline/resource/path.txt
```

---

## 1. Restart Autoware with overlay mounts — **host**

Add these mounts to the canonical `docker run` (in addition to existing map / autoware_data / NDT mounts):

```bash
-v "$HOME/summer26/src:/home/aw/ros2_ws/src:ro" \
-v "$HOME/summer26/scripts:/home/aw/scripts:ro" \
-v "$HOME/summer26/data/bags:/home/aw/bags" \
```

Full example (stop old container first):

```bash
sudo docker run --rm -d \
  --name autoware_full_test \
  --device nvidia.com/gpu=all \
  --network host \
  -e HOME=/home/aw \
  -e ROS_DOMAIN_ID=26 \
  -v "$HOME/summer26/data/maps:/home/aw/maps:ro" \
  -v "$HOME/summer26/data/autoware_data:/home/aw/autoware_data" \
  -v "$HOME/summer26/data/autoware_data/ndt_scan_matcher.param.yaml:/opt/autoware/autoware_launch/share/autoware_launch/config/localization/ndt_scan_matcher/ndt_scan_matcher.param.yaml" \
  -v "$HOME/summer26/src:/home/aw/ros2_ws/src:ro" \
  -v "$HOME/summer26/scripts:/home/aw/scripts:ro" \
  -v "$HOME/summer26/data/bags:/home/aw/bags" \
  --entrypoint /bin/bash \
  ghcr.io/autowarefoundation/autoware:universe-cuda-humble \
  -lc '
    source /opt/ros/humble/setup.bash
    source /opt/autoware/setup.bash
    unset CYCLONEDDS_URI
    export ROS_DOMAIN_ID=26
    MAP=/home/aw/maps/nishishinjuku_autoware_map
    DATA=/home/aw/autoware_data/ml_models
    ros2 launch autoware_launch e2e_simulator.launch.xml \
      vehicle_model:=awsim_labs_vehicle \
      sensor_model:=awsim_labs_sensor_kit \
      map_path:="$MAP" \
      data_path:="$DATA" \
      launch_vehicle_interface:=true \
      rviz:=false \
      rviz_respawn:=false \
      launch_rviz_adaptors:=true
  '
```

**Why:** `dsgn_offline` must be built inside the image (needs `autoware_perception_msgs`). Mounting `src/` exposes the package; mounting `scripts/` exposes build/run helpers.

---

## 2. Build ROS 2 overlay — **inside Docker**

```bash
sudo docker exec -it autoware_full_test bash
```

```bash
bash /home/aw/scripts/dsgn_offline_build.sh
source /home/aw/ros2_ws/install/setup.bash
```

Manual equivalent:

```bash
source /opt/ros/humble/setup.bash
source /opt/autoware/setup.bash
unset CYCLONEDDS_URI
export ROS_DOMAIN_ID=26
cd /home/aw/ros2_ws
colcon build --symlink-install --packages-select dsgn_offline
source install/setup.bash
```

---

## 3. Run `dsgn_offline` — **inside Docker** (second shell or background)

Stack must be up with localization + route. Recovery helper if needed:

```bash
bash /home/aw/autoware_data/recover_after_awsim_restart.sh \
  /home/aw/autoware_data/route_candidates_straight.json
```

Run the node:

```bash
bash /home/aw/scripts/dsgn_offline_run.sh
```

Manual equivalent:

```bash
source /opt/ros/humble/setup.bash
source /opt/autoware/setup.bash
source /home/aw/ros2_ws/install/setup.bash
unset CYCLONEDDS_URI
export ROS_DOMAIN_ID=26

ros2 run dsgn_offline dsgn_offline \
  --ros-args \
  -p detection_folder:=/home/aw/ros2_ws/src/dsgn_offline/resource/awsim_output_offline \
  -p path_file:=/home/aw/ros2_ws/src/dsgn_offline/resource/path.txt
```

Watch node logs for `Published detection from: NNNNNN.txt` and `Detection file not found` warnings.

---

## 4. Verify detection → tracking chain — **inside Docker**

```bash
ros2 topic hz /perception/object_recognition/detection/objects
ros2 topic echo /perception/object_recognition/detection/objects --once
ros2 topic hz /perception/object_recognition/tracking/objects
ros2 topic echo /perception/object_recognition/tracking/objects --once
```


| Symptom                           | Likely cause                                                           |
| --------------------------------- | ---------------------------------------------------------------------- |
| No detection messages             | Ego pose not initialized; wrong `path_file`; no matching `NNNNNN.txt`  |
| Detection empty `objects: []`     | Frame index points to empty file (many early indices are empty)        |
| Detection OK, tracking empty      | Wrong `frame_id`, stale `stamp`, low `existence_probability`, or class |
| Two publishers on detection topic | Default LiDAR detector still running — may need launch tweak later     |


**Known node quirks** (in `dsgn_offline.py`):

- Yaw hardcoded to `0`
- Close objects get `+5.0` m on `x` in base_link
- All objects classified as Car (`label=1`)

---

## 5. Confirm planner reaction — **inside Docker** + RViz

RViz: enable perception object displays; check markers near ego.

```bash
ros2 topic echo /planning/scenario_planning/trajectory --once
ros2 topic echo /vehicle/status/velocity_status --once
bash /home/aw/autoware_data/quick_motion_check.sh
bash /home/aw/autoware_data/diagnose_stuck.sh
```

Stop the node (Ctrl+C) and confirm behavior returns toward baseline when objects disappear.

**A/B rosbag experiments:** [`notes/ROSBAG_AB_TESTING.md`](ROSBAG_AB_TESTING.md) — bags go to `~/summer26/data/bags/` (not under `autoware_data`).

---

## 6. Output-space attack (no ML) — **host** + **inside Docker**

Copy clean outputs before editing:

```bash
cp -r ~/summer26/src/dsgn_offline/resource/awsim_output_offline \
      ~/summer26/src/dsgn_offline/resource/awsim_output_attack_ghost
```

Edit KITTI-format lines in selected `NNNNNN.txt` files under `awsim_output_attack_ghost/`:

- **Suppress:** delete all lines in a frame file
- **Closer:** reduce depth field (column 14, `z` in camera frame)
- **Ghost:** copy a line from a frame with a car and paste into an empty frame on the ego path

Relaunch with alternate folder:

```bash
DETECTION_FOLDER=/home/aw/ros2_ws/src/dsgn_offline/resource/awsim_output_attack_ghost \
  bash /home/aw/scripts/dsgn_offline_run.sh
```

---

## 7. Image-space patch attack (later) — **host** (GPU / PyTorch)

Requires DSGN environment from `external/DSGN_custom/` (separate repo; see its README for `setup.py` + CUDA deps).

**Paths on this machine:**


| Item | Path |
| ---- | ---- |
| Clean dataset (Arka) | `~/summer26/dsgn/datasets/arka/dsgn_awsim/testing_offline/` |
| Patched dataset (adversarial) | `~/summer26/dsgn/datasets/adria/testing_offline_patched/` — built by `scripts/apply_stereo_patches.py` from clean images + `patches_100_200.csv` |
| Full offline split | `~/summer26/dsgn/datasets/arka/dsgn_awsim/test_offline.txt` (214 frames) |
| Quick validation split | `~/summer26/dsgn/datasets/arka/dsgn_awsim/test_offline_validate.txt` (frames 000010, 000099, 000105) |
| Single-frame validation | `~/summer26/dsgn/datasets/arka/dsgn_awsim/test_offline_frame10.txt` (frame 000010 only) |
| Patch config | `~/summer26/dsgn/datasets/adria/testing_offline_patched/patches_100_200.csv` |
| Config | `~/summer26/external/DSGN_custom/configs/config_car_12g_awsim.py` |
| Checkpoint | `~/summer26/dsgn/checkpoints/arka/dsgn_12g_awsim_remote_downsample/finetune_60.tar` |

`scripts/dsgn_run_inference.sh` defaults to the **clean** Arka dataset (`DATA_PATH=.../testing_offline`, `SPLIT_FILE=.../test_offline.txt`). Override `DATA_PATH` for patched images. (`scripts/dsgn_run_inference_docker.sh` still defaults to `testing_offline_patched`.)

**Inference** (`dsgn_run_inference.sh` moves KITTI txt to `dsgn/detections/adria/<tag>/`):

```bash
# Clean images, full offline split (default)
bash ~/summer26/scripts/dsgn_run_inference.sh

# Patched images (adversarial)
DATA_PATH=~/summer26/dsgn/datasets/adria/testing_offline_patched \
  TAG=_patched_100_135 \
  bash ~/summer26/scripts/dsgn_run_inference.sh

# Clean images, 3-frame validation split
SPLIT_FILE=~/summer26/dsgn/datasets/arka/dsgn_awsim/test_offline_validate.txt \
  TAG=_validate_clean \
  bash ~/summer26/scripts/dsgn_run_inference.sh
```

Manual equivalent (clean default — create `data/awsim` symlinks first, or use `dsgn_run_inference.sh`):

```bash
cd ~/summer26/external/DSGN_custom

python3 tools/test_no_eval.py \
  --cfg configs/config_car_12g_awsim.py \
  --data_path ~/summer26/dsgn/datasets/arka/dsgn_awsim/testing_offline \
  --split_file ~/summer26/dsgn/datasets/arka/dsgn_awsim/test_offline.txt \
  --loadmodel ~/summer26/dsgn/checkpoints/arka/dsgn_12g_awsim_remote_downsample/finetune_60.tar \
  -btest 1 \
  -d 0
```

Copy outputs into the mount path Autoware sees, e.g.:

```bash
cp -r ~/summer26/dsgn/detections/adria/patched_100_135 \
      ~/summer26/src/dsgn_offline/resource/awsim_output_adversarial
```

Replay:

```bash
DETECTION_FOLDER=/home/aw/ros2_ws/src/dsgn_offline/resource/awsim_output_adversarial \
  bash /home/aw/scripts/dsgn_offline_run.sh
```

Adjust `--tag` on `test_no_eval.py` if you need a distinct output subdirectory name.

---

## Troubleshooting


| Issue                     | Action                                                                      |
| ------------------------- | --------------------------------------------------------------------------- |
| `package.xml not found`   | Add `src/` volume mount; restart container                                  |
| `overlay not built`       | Run Step 2                                                                  |
| MRM / engage blocked      | Unrelated to DSGN — see `notes/DEBUG_LOG.md`                                |
| Detections in wrong place | Ego not on Arka path; check pose vs `path.txt`                              |
| `ros2` shows few topics   | `ros2 daemon stop && ros2 daemon start`; wait ~2–3 min after Autoware start |


---

## Commit reminder

Path fix lives in the `**dsgn_offline` fork** (`src/dsgn_offline/`), not in `summer26`. Commit there when satisfied:

```bash
cd ~/summer26/src/dsgn_offline
git diff
git commit -am "Use container paths; path_file ROS parameter"
```

