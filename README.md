## Evaluating Adversarial Physical Attacks on Autoware + AWSIM Autonomus Vehicle stack

Research stack for **visual/physical adversarial patches** against a stereo 3D detector (DSGN), then measuring impact on Autoware driving in AWSIM.

**Stack:** AWSIM + Autoware Universe + optional RViz, ROS 2 Humble, Docker with host networking.

**Start here:** [`HANDOFF.md`](HANDOFF.md) (repos, data link, first runs). Detailed commands stay in this README and in [`notes26/`](notes26/).

---

## Quick start

1. Clone this central repo on **`main`** (ignore stale branch `dsgn2`).
2. Clone the two forks into the expected paths (they are **not** submodules):

   ```bash
   git clone git@github.com:adricalm/DSGN_custom.git external/DSGN_custom
   git clone git@github.com:adricalm/dsgn_offline.git src/dsgn_offline
   ```

3. Copy non-git assets from the shared drive. See [`HANDOFF.md`](HANDOFF.md) for the link and destination folders (AWSIM, map PCDs, Autoware `ml_models`, `dsgn/datasets`, checkpoints).
4. Bring up the stack: [`notes26/autoware-awsim-startup.md`](notes26/autoware-awsim-startup.md).
5. DSGN:  checkpoint **`finetune_48`** (PyTorch 2.6). Setup: `bash scripts/dsgn/dsgn_setup_venv.sh`.
6. Patches: [`notes26/PATCH_OPTIMIZATION.md`](notes26/PATCH_OPTIMIZATION.md).

**Not in git:** AWSIM binaries, map pointclouds, Autoware ML models, datasets/checkpoints/detections, and the two nested forks above.

---

## Environment

| Item | Value |
|------|-------|
| Project root | `~/summer26` |
| GPU | NVIDIA L40S |
| Host OS | Ubuntu 25.10 |
| Autoware | Runs **inside Docker**, not natively on the host |
| Docker image | `ghcr.io/autowarefoundation/autoware:universe-cuda-humble` |

**ROS 2 discovery:** `--network host`, `ROS_DOMAIN_ID=26`, `CYCLONEDDS_URI` unset. AWSIM, Autoware, and RViz must all share the same domain.

---

## High-level architecture

```text
┌─────────────────────────────────────────────────────────────┐
│  Host (Ubuntu 25.10, GPU passthrough, X11 / xrdp GUI)       │
│                                                             │
│  ┌──────────────┐   host network + ROS_DOMAIN_ID=26         │
│  │ AWSIM        │◄──────────────────────────────────────┐   │
│  │ (Docker)     │  publishes /clock, vehicle status     │   │
│  └──────────────┘                                       │   │
│  ┌──────────────┐   perception, localization, planning  │   │
│  │ Autoware     │◄──────────────────────────────────────┤   │
│  │ (Docker)     │                                       │   │
│  └──────────────┘                                       │   │
│  ┌──────────────┐   visualization + manual control      │   │
│  │ RViz         │◄──────────────────────────────────────┘   │
│  │ (Docker)     │                                           │
│  └──────────────┘                                           │
└─────────────────────────────────────────────────────────────┘
```

**Research path (current):** record stereo in AWSIM → DSGN inference (`finetune_48`, PT 2.6) → localize/optimize face patches → apply patches → optional `dsgn_offline` replay in Autoware → observe planning/control.

---

## Project layout

```text
~/summer26/
├── README.md
├── data/
│   ├── maps/nishishinjuku_autoware_map/
│   ├── autoware_data/                   # ml_models, route JSONs, bind-mount overrides
│   │   ├── ndt_scan_matcher.param.yaml
│   │   ├── perception.launch.xml.no_detection   # disables LiDAR CenterPoint
│   │   ├── autonomous_emergency_braking.param.yaml
│   │   └── route_*.json
│   └── awsim/                           # extracted/ + modded/ binaries
├── dsgn/                                # datasets, checkpoints, detections (bulk gitignored)
│   ├── datasets/arka/dsgn_awsim/        # Arka test / testing_offline splits
│   ├── datasets/adria/                  # patch_train / patch_test + DSGN network labels
│   ├── datasets/recordings/             # raw stereo clips (train_frontal*, test_frontal*)
│   ├── checkpoints/
│   └── detections/adria/                # train_recordings_clean, test_recordings_{clean,patched}
├── scripts/                             # mounted at /home/aw/scripts in Autoware
│   ├── awsim/                           # launch, verify, stereo mod (host)
│   ├── dsgn/                            # setup, train, inference, label tools (host)
│   ├── patch_optimization/              # localize → optimize → apply → eval
│   ├── helpers/                         # optional viz, pose capture, NPC spawn, …
│   └── drive_route_and_engage.sh, engage_*.sh, dsgn_offline_*.sh, record_kitti_dataset.sh
├── src/                                 # separate git repos or small integration code
│   ├── dsgn_offline/                    # ROS 2 offline detection publisher (fork)
│   ├── awsim_stereo_mod/                # StereoMod C# source (built → modded AWSIM)
│   └── awsim_to_kitti/                  # KITTI recorder used by record_kitti_dataset.sh
├── external/
│   └── DSGN_custom/                     # DSGN train/infer (fork)
└── notes26/                             # findings and runbooks
```

### DSGN repos (separate forks, not part of this repo)

`src/dsgn_offline/` and `external/DSGN_custom/` are **independent git checkouts**, listed in `.gitignore` so this repo does not track them as nested submodules or ghost folders.

| Path | Role | Git remote |
|------|------|------------|
| `external/DSGN_custom/` | Train/run DSGN; write detection `.txt` files | [adricalm/DSGN_custom](https://github.com/adricalm/DSGN_custom) (`upstream` = Arka’s repo) |
| `src/dsgn_offline/` | Publish offline detections into Autoware (`/perception/object_recognition/detection/objects`) | [adricalm/dsgn_offline](https://github.com/adricalm/dsgn_offline) (`upstream` = Arka’s repo) |

**Remotes (both nested repos):** clone from the adricalm URLs above (`origin`). Keep `upstream` = [DF-Autoware-AWSIM](https://github.com/DF-Autoware-AWSIM) to pull Arka’s baseline. If the checkout came from Arka’s repo directly, repoint remotes before committing:

```bash
cd external/DSGN_custom   # or src/dsgn_offline
git remote rename origin upstream
git remote add origin git@github.com:adricalm/<repo>.git
git push -u origin master   # dsgn_offline uses main
```

**Workflow:** edit inside the nested repo → `git commit` / `git push` there. Only commit integration docs, scripts, and small configs in `summer26`. Clone the forks locally after a fresh checkout (they are not bundled in this repo).

**Dataset split (disk space):** Training lives only under **`dsgn/datasets/adria/training_kitti_labels/`** (KITTI-converted `label_2`). Arka’s original `training/` was removed to free space. Keep using Arka for **`testing/`**, **`testing_offline/`**, and split files (`trainval.txt`, `test_offline*.txt`, …). Details: [`notes26/DSGN_AWSIM_FINDINGS.md`](notes26/DSGN_AWSIM_FINDINGS.md).

**DSGN on this host (L40S): two different stories:**

| Checkpoint | PT 2.6 on L40S | Notes |
|------------|----------------|-------|
| Official KITTI **`finetune_48`** | **Works** on AWSIM once config matches half-res loader | `input_size=[540,960]`, `output_size=[135,240]`; viz with `--box-convention kitti` |
| Arka AWSIM **`finetune_60`** | **Unfaithful** vs Arka baseline | Trained PT 1.3; prefer Arka’s precomputed detection dumps for A/B |

Geometry, label convention, and finetune strategy: **[`notes26/DSGN_AWSIM_FINDINGS.md`](notes26/DSGN_AWSIM_FINDINGS.md)**.  
Offline Autoware replay: [`notes26/DSGN_OFFLINE_RUNBOOK.md`](notes26/DSGN_OFFLINE_RUNBOOK.md).

**PyTorch note:** host venv uses PT 2.6 (L40S). Official KITTI `finetune_48` works; Arka `finetune_60` re-inference is unfaithful. Prefer Arka’s precomputed detection dumps for A/B.

**Useful scripts:** `dsgn/dsgn_run_inference.sh`, `dsgn/dsgn_train.sh`, `dsgn/dsgn_transform_label.py`, `helpers/visualize_dsgn_detections.py`.

---

## ROS 2 setup (run at the start of every Docker shell)

```bash
source /opt/ros/humble/setup.bash
source /opt/autoware/setup.bash
unset CYCLONEDDS_URI
export ROS_DOMAIN_ID=26
```

---

## Launch commands

### Autoware (host; canonical, always use this)

| Mount / flag | Why |
|---|---|
| `data/autoware_data/ndt_scan_matcher.param.yaml` | Nishi-Shinjuku scores ~2.2–2.4; default threshold 2.3 trips MRM `emergency_stop` while driving |
| `data/autoware_data/perception.launch.xml.no_detection` | Disables LiDAR object detection (CenterPoint). Localization LiDAR stays on. |
| `data/autoware_data/autonomous_emergency_braking.param.yaml` | Sets AEB `use_pointcloud_data: false` so raw LiDAR clusters don’t trigger AEB when detection is off |
| `launch_rviz_adaptors:=true` | Bridges RViz **2D Rough Goal Pose** clicks to `/api/routing/set_route_points` |

Also mount `scripts/` → `/home/aw/scripts` and `src/` → `/home/aw/ros2_ws/src`.

**Note:** `helpers/spawn_test_npc_car.sh` injects objects via `/simulation/dummy_perception_publisher/object_info` even with CenterPoint off. That is separate from LiDAR detection.

```bash
docker run --rm -d \
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
  -v "$HOME/summer26/data/autoware_data/perception.launch.xml.no_detection:/opt/autoware/tier4_perception_launch/share/tier4_perception_launch/launch/perception.launch.xml:ro" \
  -v "$HOME/summer26/data/autoware_data/autonomous_emergency_braking.param.yaml:/opt/autoware/autoware_launch/share/autoware_launch/config/control/autoware_autonomous_emergency_braking/autonomous_emergency_braking.param.yaml:ro" \
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

### Enter a running Autoware container

```bash
docker exec -it autoware_full_test bash
# then source ROS as above
```

### Drive the car (inside Docker)

After Autoware + AWSIM are up and `/clock` has a publisher:

```bash
bash /home/aw/scripts/drive_route_and_engage.sh
# or: bash /home/aw/scripts/drive_route_and_engage.sh /home/aw/autoware_data/route_dsgn_ab.json
```

From host: `bash ~/summer26/scripts/engage_autoware.sh` (engage + motion check).

**Startup order:** Autoware (detached) → wait ~2–3 min → AWSIM → (optional RViz) → `drive_route_and_engage.sh`.

### DSGN offline overlay (optional)

Replay precomputed KITTI-format detections into Autoware (folder of `.txt` + `path.txt`, **not** a rosbag). **Step-by-step:** [`notes26/DSGN_OFFLINE_RUNBOOK.md`](notes26/DSGN_OFFLINE_RUNBOOK.md).

The canonical `docker run` already mounts `src/` and `scripts/`. Build and run inside the container:

```bash
bash /home/aw/scripts/dsgn_offline_build.sh
source /home/aw/ros2_ws/install/setup.bash
bash /home/aw/scripts/dsgn_offline_run.sh
```

### AWSIM (inside Docker, with GUI)

AWSIM runs inside Docker (needs GUI access via `DISPLAY`/X11, i.e. the xrdp desktop,
normally `:10`). It **cannot** run natively on the host: a host run dies with
`UnsatisfiedLinkError: librcl.so`.

Do **not** source ROS 2 for the AWSIM process. Its `ros2-for-unity` is a standalone
build with its own ROS 2 libraries. Also unset the ament/colcon prefix paths.

**Manual / interactive (proven, uses the GUI `Load` button):**

```bash
cd ~/summer26/data/awsim
docker run --rm -it \
  --name awsim_gui_test \
  --device nvidia.com/gpu=all \
  --network host \
  -e DISPLAY="$DISPLAY" \
  -e HOME=/home/aw \
  -e ROS_DOMAIN_ID=26 \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v "$HOME/summer26/data/awsim:/home/aw/awsim" \
  --entrypoint /bin/bash \
  ghcr.io/autowarefoundation/autoware:universe-cuda-humble \
  -lc '
    unset CYCLONEDDS_URI
    unset ROS_DISTRO AMENT_PREFIX_PATH COLCON_PREFIX_PATH CMAKE_PREFIX_PATH
    export ROS_DOMAIN_ID=26
    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    cd /home/aw/awsim
    ./extracted/awsim_labs_v1.6.1/awsim_labs.x86_64
  '
```

`unset ROS_DISTRO` is **required**, not cosmetic. With it set, `ros2-for-unity` takes a
slower "ROS is sourced" init path and the scene can finish loading first, in which case
every C# publisher dies with `topic name is invalid` (camera, pose, odometry, IMU). It is
a race, so it sometimes appears to work; it is much more likely to lose when Autoware is
already running and loading the GPU. Always confirm with `ros2 topic list` rather than
trusting that the window looks normal. Swap `extracted/` → `modded/` for the stereo build.

**Scripted (no clicking):** `scripts/awsim/awsim_launch.sh [pristine|modded]`, then
`scripts/awsim/awsim_verify.sh <container>`. This passes `--config` so the scene auto-loads,
which additionally requires scrubbing `ROS_DISTRO` to avoid a startup race that kills
every C# publisher (`/clock`, camera, vehicle status). See
[`notes26/AWSIM_STEREO_CAMERA.md`](notes26/AWSIM_STEREO_CAMERA.md).

---

## Autoware states

| State | Name |
|-------|------|
| 1 | INITIALIZING |
| 2 | WAITING_FOR_ROUTE |
| 3 | PLANNING |
| 4 | WAITING_FOR_ENGAGE |
| 5 | DRIVING |

Healthy end state: `state: 5`, `velocity > 0`, MRM `state: 1`, `hazard emergency: false`.

---

## Key ROS topics and services

```text
/clock                                          # AWSIM only
/vehicle/status/velocity_status                 # AWSIM only

/localization/initialize                        # init pose (DIRECT method)
/api/localization/initialization_state          # state 3 = initialized

/api/routing/set_route_points                   # set route
/rviz/routing/rough_goal                        # RViz 2D Rough Goal Pose (needs launch_rviz_adaptors:=true)
/api/routing/state                              # 1=UNSET, 2=SET
/api/routing/clear_route

/autoware/state
/api/operation_mode/state                       # is_autonomous_mode_available
/api/operation_mode/change_to_autonomous
/api/operation_mode/change_to_stop

/system/emergency/hazard_status
/api/fail_safe/mrm_state

/perception/traffic_light_recognition/traffic_signals            # Autoware TL recognition output
```

---

## Helper scripts

Shared tools under `scripts/` (mounted at `/home/aw/scripts`). Session diagnostics under `data/autoware_data/`.

### Autoware session diagnostics (`data/autoware_data/`)

| Script | Purpose |
|--------|---------|
| `diagnose_stuck.sh` | Why stopped? (MRM / obstacle / loc) |
| `dsgn_chain_check.sh` | Verify detection → tracker → prediction wiring |
| `inspect_emergency.sh` | Which diagnostic is causing hazard/MRM |
| `verify_stack_ready.sh` | Post-restart health check |

### AWSIM (`scripts/awsim/`; host unless noted)

| Script | Purpose |
|--------|---------|
| `awsim/awsim_launch.sh` | Launch AWSIM (`pristine`/`modded`) in Docker with correct env |
| `awsim/awsim_verify.sh` | Check AWSIM topics, rates, live camera geometry |
| `awsim/awsim_stereo_build.sh` | Compile the stereo-camera mod (`StereoMod.dll`) |
| `awsim/awsim_stereo_install.py` | Register/`--uninstall` the mod in `modded/` |
| `awsim/awsim_stereo_check.py` | **In Docker:** quantitative stereo pair validation |

### Driving / offline replay (`scripts/` root; Docker paths)

| Script | Purpose |
|--------|---------|
| `drive_route_and_engage.sh` | Clear → init pose → goal → engage |
| `engage_autoware.sh` | **Host:** engage + motion check |
| `record_kitti_dataset.sh` | **Host:** record KITTI-layout stereo from modded AWSIM |
| `dsgn_offline_build.sh` | **Inside Docker:** colcon build `dsgn_offline` |
| `dsgn_offline_run.sh` | **Inside Docker:** publish offline detections |

### DSGN (`scripts/dsgn/`; host)

| Script | Purpose |
|--------|---------|
| `dsgn/dsgn_setup_venv.sh` | PT 2.6 + cu124 venv for L40S |
| `dsgn/dsgn_run_inference.sh` | Inference on host (PT 2.6; use `finetune_48`) |
| `dsgn/dsgn_train.sh` | Full-model fine-tune from `finetune_48` |
| `dsgn/dsgn_transform_label.py` | AWSIM `label_2` → KITTI convention |
| `dsgn/merge_dsgn_outputs.py` | Merge baseline + patched frame ranges |

### Patch optimization (`scripts/patch_optimization/`)

| Script | Purpose |
|--------|---------|
| `localize_patches.py` | Project rear-face boxes from detections → CSV |
| `optimize_patch.py` | Optimize universal adversarial patch vs DSGN |
| `apply_face_patch.py` | Apply optimized face patch to a dataset |
| `eval_patch.py` | Offline evaluate patch suppression |
| `attack_stats.py` | Compare baseline vs patched offline detections |
| `prepare_recording_datasets.py` | Split/calib prep after `record_kitti_dataset.sh` |
| `build_combined_dataset.py` | Merge recording CSVs for multi-run optimization |

### Helpers (`scripts/helpers/`; optional)

| Script | Purpose |
|--------|---------|
| `helpers/visualize_dsgn_detections.py` | Overlay KITTI/AWSIM boxes on images |
| `helpers/visualize_patch.py` | Paste optimized patch onto sample frames |
| `helpers/capture_pose.sh` | Dump ego pose for hand-built route JSON |
| `helpers/make_route_json.sh` | Combine start/goal pose fragments |
| `helpers/spawn_test_npc_car.sh` | Respawn DummyObject NPCs |
| `helpers/monitor_stop_cause.py` | 5 Hz stop-cause logger while driving |
| `helpers/download_autoware_artifacts_from_role.py` | Bootstrap Autoware artifacts (rare) |

### Notes / handoff

| Doc | Purpose |
|-----|---------|
| [`HANDOFF.md`](HANDOFF.md) | Setup orientation: repos, data link, first runs |
| [`notes26/autoware-awsim-startup.md`](notes26/autoware-awsim-startup.md) | Minimal AWSIM + Autoware startup |
| [`notes26/PATCH_OPTIMIZATION.md`](notes26/PATCH_OPTIMIZATION.md) | Patch localize → optimize → apply → eval order |
| [`notes26/AWSIM_STEREO_CAMERA.md`](notes26/AWSIM_STEREO_CAMERA.md) | Stereo camera mod on the AWSIM **binary**; launch gotchas |
| [`notes26/DSGN_OFFLINE_RUNBOOK.md`](notes26/DSGN_OFFLINE_RUNBOOK.md) | Offline Autoware replay workflow |
| [`notes26/DSGN_AWSIM_FINDINGS.md`](notes26/DSGN_AWSIM_FINDINGS.md) | Geometry, labels, finetune decisions (`finetune_48` vs `finetune_60`) |
| [`notes26/autoware_pin.md`](notes26/autoware_pin.md) | Autoware Docker image pin |

---

## Research direction

1. Inject **visual/physical adversarial patches** into the perception pipeline (DSGN-based setup).
2. Measure whether perception errors affect **driving behavior** (planning, control).
3. Evaluate **defense / recovery** methods.

See `dsgn/` for datasets, checkpoints, and detection outputs; DSGN forks above for perception code.

---

## External references

- Central repo: [adricalm/adversarial-attacks-autonomous-driving](https://github.com/adricalm/adversarial-attacks-autonomous-driving)
- Autoware pin: [`notes26/autoware_pin.md`](notes26/autoware_pin.md) (Docker image only; no local checkout)
- DSGN stereo detector: `external/DSGN_custom/` (fork and commit separately)
- DSGN offline ROS bridge: `src/dsgn_offline/` (fork and commit separately)
- Arka org (upstream forks): [DF-Autoware-AWSIM](https://github.com/orgs/DF-Autoware-AWSIM/repositories)
- Map: Autoware Nishi-Shinjuku sample (`lanelet2_map.osm` + pointcloud)
- AWSIM Labs v1.6.1: `data/awsim/`
- DSGN datasets: train = `dsgn/datasets/adria/training_kitti_labels/`; test/offline = `dsgn/datasets/arka/dsgn_awsim/` (Arka `training/` removed for disk)
