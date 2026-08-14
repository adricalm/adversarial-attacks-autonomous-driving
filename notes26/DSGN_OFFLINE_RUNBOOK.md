# DSGN offline runbook

Copy-paste commands for replaying Arka's precomputed DSGN detections into Autoware.  
**You run everything.** This repo only provides scripts and the patched `dsgn_offline` node.

See also: [`autoware-awsim-startup.md`](autoware-awsim-startup.md) (Autoware launch) and [README](../README.md).

---

## 0. Prerequisites

Before building or running `dsgn_offline`:

- [ ] Autoware container running (`autoware_full_test`) with **canonical launch** from [`autoware-awsim-startup.md`](autoware-awsim-startup.md) (includes `src/` + `scripts/` mounts, NDT bind-mount, `launch_rviz_adaptors:=true`)
- [ ] AWSIM running
- [ ] Localization initialized and route set (`bash /home/aw/scripts/drive_route_and_engage.sh`)
- [ ] Ego driving or ready to engage (`/autoware/state` → 5)

**Synchronization note:** `dsgn_offline` picks the nearest row in `path.txt` from current `/localization/pose_with_covariance`. 

```bash
# inside Docker
ros2 topic echo /localization/pose_with_covariance --once
head -3 /home/aw/ros2_ws/src/dsgn_offline/resource/path.txt
```

---


## 1. Build ROS 2 overlay (**inside Docker**)

```bash
docker exec -it autoware_full_test bash
```

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

## 2. Run `dsgn_offline` (**inside Docker**, second shell or background)

Stack must be up with localization + route. If AWSIM was restarted, re-init the route:

```bash
bash /home/aw/scripts/drive_route_and_engage.sh \
  /home/aw/autoware_data/route_dsgn_ab.json
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

## 3. Verify detection → tracking chain (**inside Docker**)

```bash
ros2 topic hz /perception/object_recognition/detection/objects
ros2 topic echo /perception/object_recognition/detection/objects --once
ros2 topic hz /perception/object_recognition/tracking/objects
ros2 topic echo /perception/object_recognition/tracking/objects --once
```
A straigtfoward approach is to just visualize the bounding boxes in Rviz (make sure to check the Perception -> Object Recognition -> Detection in Display.
**Known node quirks** (in `dsgn_offline.py`):

- Yaw hardcoded to `0`
- Close objects get `+5.0` m on `x` in base_link
- All objects classified as Car (`label=1`)


---

## 4. Test attacked frames

Relaunch with alternate folder:

```bash
DETECTION_FOLDER=/home/aw/ros2_ws/src/dsgn_offline/resource/awsim_output_attack_ghost \
  bash /home/aw/scripts/dsgn_offline_run.sh
```

---

## Commit reminder

Path fix lives in the `**dsgn_offline` fork** (`src/dsgn_offline/`), not in `summer26`. Commit there when satisfied:

```bash
cd ~/summer26/src/dsgn_offline
git diff
git commit -am "Use container paths; path_file ROS parameter"
```

