# summer26 — Autoware + AWSIM adversarial-driving research stack

KTH summer internship project evaluating **visual/physical adversarial attacks and defenses** in autonomous-driving simulation.

**Stack:** AWSIM (simulator) + Autoware Universe (perception, localization, planning) + RViz, over ROS 2 Humble, all inside Docker with host networking.

---

## For a new assistant (read this first)

1. **Do not assume the previous chat's blocker still applies.** Ask or infer the current state from the latest terminal output, then continue from there.
2. **Be concise and practical.** One debugging step at a time.
3. **Always say where a command runs:** `host` vs `inside Docker`.
4. **Explain briefly why** each command matters — not just copy-paste.
5. **Verify before fixing.** Use `ros2 topic info -v`, `ros2 topic echo --once`, `ros2 service type`, `ros2 interface show`, and logs. Do not guess silently.
6. **Server caution:** This is a shared lab server. Ask before commands with global effects. Changes under `~/summer26` (user `adria`) are generally fine.
7. **Docker no longer needs sudo** (`adria` is in the `docker` group), so agents can run containers directly. For anything else that genuinely needs `sudo`, **stop and give adria the exact command to copy-paste** — one at a time, labeled `host` vs `inside Docker`, with a one-line reason.

---

## Environment

| Item | Value |
|------|-------|
| Host user | `adria` |
| Hostname | `sys-user-PowerEdge-R7715` |
| Project root | `~/summer26` |
| GPU | NVIDIA L40S |
| Host OS | Ubuntu 25.10 |
| Autoware | Runs **inside Docker**, not natively on the host |
| Docker image | `ghcr.io/autowarefoundation/autoware:universe-cuda-humble` |

**Docker access:** `adria` **is** in the `docker` group (since Aug 2026) — run `docker ...` directly, no `sudo`, no password. Agents can therefore run Docker themselves. Note this is a shared server: `docker ps` will show other users' containers (e.g. `minikube`); never stop what you did not start.

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

**AWSIM is the only publisher** of `/clock` and `/vehicle/status/*`. Do not run fake clock/status nodes.

---

## Project layout

```text
~/summer26/
├── README.md
├── .gitignore
├── data/
│   ├── maps/
│   │   └── nishishinjuku_autoware_map/   # lanelet2_map.osm, pointcloud, projector info
│   ├── autoware_data/                    # mounted read-write into Autoware container
│   │   ├── ml_models/                   # perception models (data_path)
│   │   ├── ndt_scan_matcher.param.yaml  # ← bind-mounted into container (required)
│   │   ├── route_candidates*.json       # generated start/goal poses
│   │   └── (recovery / diagnostic shell scripts only — no tool duplicates)
│   ├── awsim/
│   │   ├── awsim_labs_v1.6.1.zip
│   │   └── extracted/                   # AWSIM binary
│   └── bags/                            # experiment rosbags (gitignored; owned by adria on host)
├── dsgn/                                # DSGN pipeline — datasets, checkpoints, detections (gitignored bulk)
│   ├── datasets/arka/dsgn_awsim/        # testing + testing_offline + split .txt (no training/)
│   ├── datasets/adria/
│   │   ├── dsgn_awsim/                  # adria patches / derived AWSIM experiments
│   │   └── training_kitti_labels/       # ONLY training set (KITTI-converted label_2)
│   ├── checkpoints/                   # .tar weights only (arka / kitti / adria)
│   ├── detections/                    # per-frame KITTI .txt inference outputs
│   └── training_logs/
├── scripts/                             # all helpers (mounted at /home/aw/scripts in Autoware)
│   ├── dsgn_*.sh / dsgn_*.py            # DSGN venv, train, infer, viz, merge, offline bridge
│   ├── dsgn2_*.sh / dsgn2_*.py          # DSGN++ detour (kept; candidate for future cleanup)
│   ├── apply_route_from_osm.py          # localize + set route (inside Docker)
│   ├── traffic_light_green_bridge.py    # force GREEN TLs (inside Docker)
│   ├── drive_route_and_engage.sh        # clear → pose → goal → engage
│   └── archive/dsgn_pt_experiments/     # failed PT 1.3/1.10 host experiments
├── external/
│   ├── autoware/                        # upstream Autoware reference (gitignored)
│   ├── DSGN_custom/                     # stereo 3D detector — separate git repo (gitignored)
│   └── DSGN2_awsim/                     # DSGN++ fork (feasibility detour; gitignored)
├── logs/                                # script output logs (gitignored)
├── notes/
│   └── DEBUG_LOG.md                     # pitfalls, diagnostics chain, changelog
└── src/
    └── dsgn_offline/                    # ROS 2 offline perception bridge — separate git repo (gitignored)
```

**Map:** Nishi-Shinjuku (`data/maps/nishishinjuku_autoware_map/`). MGRS grid `54SUE`. Elevation ~40.9 m — never use z=0.

### DSGN repos (separate forks, not part of this repo)

`src/dsgn_offline/` and `external/DSGN_custom/` are **independent git checkouts** from the prior intern’s work, listed in `.gitignore` so this repo does not track them as nested submodules or ghost folders.

| Path | Role | Where to commit changes |
|------|------|-------------------------|
| `external/DSGN_custom/` | Train/run DSGN; write detection `.txt` files | Your fork: [adricalm/DSGN_custom](https://github.com/adricalm/DSGN_custom) (`upstream` = Arka’s repo) |
| `src/dsgn_offline/` | Publish offline detections into Autoware (`/perception/object_recognition/detection/objects`) | Your fork: [adricalm/dsgn_offline](https://github.com/adricalm/dsgn_offline) (`upstream` = Arka’s repo) |

**Why:** keeps Autoware integration (`summer26`) separate from ML/ROS code you will modify; avoids accidental commits of large datasets or model weights; preserves a clear diff against the prior baseline via `upstream`.

**Remotes (both nested repos):** `origin` = your fork (push here); `upstream` = [DF-Autoware-AWSIM](https://github.com/DF-Autoware-AWSIM) (pull Arka’s updates). If you cloned Arka’s repo directly, repoint before your first commit:

```bash
cd external/DSGN_custom   # or src/dsgn_offline
git remote rename origin upstream
git remote add origin git@github.com:adricalm/<repo>.git
git push -u origin master   # dsgn_offline uses main
```

**Workflow:** edit inside the nested repo → `git commit` / `git push` there. Only commit integration docs, scripts, and small configs in `summer26`. Clone the forks locally after a fresh checkout (they are not bundled in this repo).

**Dataset split (disk space):** Training lives only under **`dsgn/datasets/adria/training_kitti_labels/`** (KITTI-converted `label_2`). Arka’s original `training/` was removed to free space. Keep using Arka for **`testing/`**, **`testing_offline/`**, and split files (`trainval.txt`, `test_offline*.txt`, …). Details: [`notes/DSGN_AWSIM_FINDINGS.md`](notes/DSGN_AWSIM_FINDINGS.md).

**DSGN on this host (L40S) — two different stories:**

| Checkpoint | PT 2.6 on L40S | Notes |
|------------|----------------|-------|
| Official KITTI **`finetune_48`** | **Works** on AWSIM once config matches half-res loader | `input_size=[540,960]`, `output_size=[135,240]`; viz with `--box-convention kitti` |
| Arka AWSIM **`finetune_60`** | **Unfaithful** vs Arka baseline | Trained PT 1.3; prefer precomputed dumps or old-GPU Docker |

Geometry, label convention, and finetune strategy: **[`notes/DSGN_AWSIM_FINDINGS.md`](notes/DSGN_AWSIM_FINDINGS.md)**.  
PyTorch / GPU matrix: [`notes/DSGN_PYTORCH_VERSIONING.md`](notes/DSGN_PYTORCH_VERSIONING.md). Offline Autoware replay: [`notes/DSGN_OFFLINE_RUNBOOK.md`](notes/DSGN_OFFLINE_RUNBOOK.md).

**Useful scripts:** `dsgn_run_inference.sh`, `dsgn_train.sh`, `dsgn_transform_label.py`, `visualize_dsgn_detections.py`.  
(Historical `dsgn_finetune_awsim.sh` with `MODE=det_head|gentle` was removed; see [`notes/DSGN_ADAPT_DET_HEAD_KITTI_LABELS.md`](notes/DSGN_ADAPT_DET_HEAD_KITTI_LABELS.md) for that experiment’s recipe.)

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

### Autoware (host — canonical, always use this)

| Mount / flag | Why |
|---|---|
| `ndt_scan_matcher.param.yaml` | Nishi-Shinjuku scores ~2.2–2.4; default threshold 2.3 trips MRM `emergency_stop` while driving |
| `perception.launch.xml.no_detection` | Disables LiDAR object detection (CenterPoint / detection launch `if="false"`). Localization LiDAR stays on. |
| `autonomous_emergency_braking.param.yaml` | Sets AEB `use_pointcloud_data: false` so raw LiDAR clusters don’t trigger AEB when detection is off |
| `launch_rviz_adaptors:=true` | Bridges RViz **2D Rough Goal Pose** clicks to `/api/routing/set_route_points` |

Also mount `scripts/` → `/home/aw/scripts` and `src/` → `/home/aw/ros2_ws/src`. Optional: `data/bags` → `/home/aw/bags` for rosbag A/B tests.

**Note:** `spawn_test_npc_car.sh` still injects objects via `/simulation/dummy_perception_publisher/object_info` even with CenterPoint off — that path is separate from LiDAR detection.

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
  -v "$HOME/summer26/logs:/home/aw/logs:rw" \
  -v "$HOME/summer26/logs/perception.launch.xml.no_detection:/opt/autoware/tier4_perception_launch/share/tier4_perception_launch/launch/perception.launch.xml:ro" \
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
sudo docker exec -it autoware_full_test bash
# then source ROS as above
```

### DSGN offline overlay (optional)

Replay Arka's precomputed stereo detections into Autoware without replacing the full perception stack. **Step-by-step commands:** [`notes/DSGN_OFFLINE_RUNBOOK.md`](notes/DSGN_OFFLINE_RUNBOOK.md).

The canonical `docker run` already mounts `src/` and `scripts/`. Build and run inside the container:

```bash
bash /home/aw/scripts/dsgn_offline_build.sh
source /home/aw/ros2_ws/install/setup.bash
bash /home/aw/scripts/dsgn_offline_run.sh
```

### AWSIM (inside Docker, with GUI)

AWSIM runs inside Docker (needs GUI access via `DISPLAY`/X11, i.e. the xrdp desktop —
normally `:10`). It **cannot** run natively on the host: a host run dies with
`UnsatisfiedLinkError: librcl.so`.

Do **not** source ROS 2 for the AWSIM process — its `ros2-for-unity` is a standalone
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
every C# publisher dies with `topic name is invalid` — camera, pose, odometry, IMU. It is
a race, so it sometimes appears to work; it is much more likely to lose when Autoware is
already running and loading the GPU. Always confirm with `ros2 topic list` rather than
trusting that the window looks normal. Swap `extracted/` → `modded/` for the stereo build.

**Scripted (no clicking):** `scripts/awsim_launch.sh [pristine|modded]`, then
`scripts/awsim_verify.sh <container>`. This passes `--config` so the scene auto-loads,
which additionally requires scrubbing `ROS_DISTRO` to avoid a startup race that kills
every C# publisher (`/clock`, camera, vehicle status). See
[`notes/AWSIM_STEREO_CAMERA.md`](notes/AWSIM_STEREO_CAMERA.md).

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

/perception/traffic_light_recognition/external/traffic_signals   # bridge publishes here
/perception/traffic_light_recognition/traffic_signals            # arbiter merged output
```

---

## Helper scripts

Shared tools live under `scripts/` (mounted at `/home/aw/scripts`). Recovery/diagnostics that are session-specific stay under `data/autoware_data/`.

### Autoware / AWSIM (inside Docker unless noted)

| Script | Purpose |
|--------|---------|
| `scripts/awsim_launch.sh` | **Host:** launch AWSIM (`pristine`/`modded`) in Docker with correct env |
| `scripts/awsim_verify.sh` | **Host:** check AWSIM topics, rates, live camera geometry |
| `scripts/awsim_stereo_build.sh` | **Host:** compile the stereo-camera mod (`StereoMod.dll`) |
| `scripts/awsim_stereo_install.py` | **Host:** register/`--uninstall` the mod in `modded/` |
| `scripts/awsim_stereo_check.py` | **In Docker:** quantitative stereo pair validation |
| `scripts/apply_route_from_osm.py` | Localize + set route from route JSON |
| `scripts/traffic_light_green_bridge.py` | Publish all map lights as GREEN |
| `scripts/run_traffic_light_bridge.sh` | **Host:** start bridge inside container |
| `scripts/run_traffic_light_bridge_inside_container.sh` | Start bridge (already inside Docker) |
| `scripts/drive_route_and_engage.sh` | Clear → init pose → goal → engage |
| `scripts/engage_autoware.sh` | **Host:** engage + motion check |
| `scripts/find_route_candidates.py` | **Host:** parse OSM → route JSON |
| `scripts/apply_route_candidates.sh` | **Host:** find candidates + apply inside container |
| `data/autoware_data/recover_after_awsim_restart.sh` | Full recovery after AWSIM restart |
| `data/autoware_data/recover_from_mrm_emergency_stop.sh` | Recover from MRM latch (localization) |
| `data/autoware_data/unstick_at_traffic_light.sh` | Stop → clear → re-route → engage |
| `data/autoware_data/quick_motion_check.sh` | Snapshot: bridge / state / velocity |
| `data/autoware_data/diagnose_stuck.sh` | Why stopped? (MRM / obstacle / loc) |
| `data/autoware_data/inspect_emergency.sh` | Which diagnostic is causing hazard/MRM |
| `data/autoware_data/verify_stack_ready.sh` | Post-restart health check |

### DSGN (host unless noted)

| Script | Purpose |
|--------|---------|
| `scripts/dsgn_setup_venv.sh` | PT 2.6 + cu124 venv for L40S |
| `scripts/dsgn_run_inference.sh` | Inference on host (PT 2.6; use `finetune_48`) |
| `scripts/dsgn_run_inference_docker.sh` | Faithful PT 1.3 inference (needs old GPU, not L40S) |
| `scripts/dsgn_bench_inference.sh` | GPU timing bench (`test_no_eval_timing.py`) |
| `scripts/dsgn_train.sh` | Full-model fine-tune from `finetune_48` |
| `scripts/dsgn_transform_label.py` | AWSIM `label_2` → KITTI convention |
| `scripts/visualize_dsgn_detections.py` | Viz with `--box-convention {kitti,awsim}` |
| `scripts/merge_dsgn_outputs.py` | Merge baseline + patched frame ranges |
| `scripts/dsgn_offline_build.sh` | **Inside Docker:** colcon build `dsgn_offline` |
| `scripts/dsgn_offline_run.sh` | **Inside Docker:** publish offline detections |

### DSGN++ (kept; future cleanup candidate)

| Script | Purpose |
|--------|---------|
| `scripts/dsgn2_build_docker.sh` | Build PT 1.7.1 DSGN++ image |
| `scripts/dsgn2_run_inference.sh` | AWSIM-layout inference in Docker |
| `scripts/dsgn2_run_kitti_inference.sh` | Real KITTI-subset inference |
| `scripts/dsgn2_feasibility_all.sh` | Run feasibility gates in order |

### Notes

| Doc | Purpose |
|-----|---------|
| [`notes/AWSIM_STEREO_CAMERA.md`](notes/AWSIM_STEREO_CAMERA.md) | Adding a stereo camera to the AWSIM **binary** (no Unity); AWSIM launch gotchas |
| [`notes/DSGN_OFFLINE_RUNBOOK.md`](notes/DSGN_OFFLINE_RUNBOOK.md) | Offline Autoware replay workflow |
| [`notes/DSGN_AWSIM_FINDINGS.md`](notes/DSGN_AWSIM_FINDINGS.md) | Geometry, labels, finetune decisions |
| [`notes/DSGN_ADAPT_DET_HEAD_KITTI_LABELS.md`](notes/DSGN_ADAPT_DET_HEAD_KITTI_LABELS.md) | Historical det_head adapt (script removed) |
| [`notes/DSGN_PYTORCH_VERSIONING.md`](notes/DSGN_PYTORCH_VERSIONING.md) | PT / GPU matrix |

---

## Research direction

1. Inject **visual/physical adversarial patches** into the perception pipeline (DSGN-based setup from prior intern work).
2. Measure whether perception errors affect **driving behavior** (planning, control).
3. Evaluate **defense / recovery** methods.

See `dsgn/` for datasets, checkpoints, and detection outputs; DSGN forks above for perception code.

---

## External references

- Autoware upstream: `external/autoware/` (see [`notes/external_autoware_ref.md`](notes/external_autoware_ref.md))
- DSGN stereo detector: `external/DSGN_custom/` — fork and commit separately (see layout above)
- DSGN offline ROS bridge: `src/dsgn_offline/` — fork and commit separately (see layout above)
- Map: Autoware Nishi-Shinjuku sample (`lanelet2_map.osm` + pointcloud)
- AWSIM Labs v1.6.1: `data/awsim/`
- DSGN datasets: train = `dsgn/datasets/adria/training_kitti_labels/`; test/offline = `dsgn/datasets/arka/dsgn_awsim/` (Arka `training/` removed for disk)
- Pitfalls, diagnostics chain, changelog: [`notes/DEBUG_LOG.md`](notes/DEBUG_LOG.md)
- DSGN offline replay: [`notes/DSGN_OFFLINE_RUNBOOK.md`](notes/DSGN_OFFLINE_RUNBOOK.md)
- DSGN AWSIM findings (geometry, labels, finetune): [`notes/DSGN_AWSIM_FINDINGS.md`](notes/DSGN_AWSIM_FINDINGS.md)
- DSGN PyTorch / GPU constraints: [`notes/DSGN_PYTORCH_VERSIONING.md`](notes/DSGN_PYTORCH_VERSIONING.md)
