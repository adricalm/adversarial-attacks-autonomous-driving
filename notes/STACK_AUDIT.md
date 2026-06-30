# Autoware stack audit — nodes, topics, and evidence levels

**Purpose:** Supervisor-ready inventory of what exists in our AWSIM + Autoware Universe launch, what is **wired**, what is **fed** (data flowing), and what is **used** (affects driving).

**Sources:**

- Frozen graph: `logs/nodes_with_dsgn_offline.txt` (194 nodes), `logs/topics_with_dsgn_offline.txt` (796 topics)
- Live probes: `ros2 topic info -v`, `ros2 node info`, user A/B test (dsgn on/off → obstacle stop)
- Image: `ghcr.io/autowarefoundation/autoware:universe-cuda-humble`, `e2e_simulator.launch.xml`, `awsim_labs_sensor_kit`

**Companion docs:** [PERCEPTION_PIPELINE.md](PERCEPTION_PIPELINE.md) · [DSGN_OFFLINE_RUNBOOK.md](DSGN_OFFLINE_RUNBOOK.md)

**Live re-audit (inside Docker):**

```bash
bash /home/aw/scripts/audit_stack_usage.sh engaged_driving
```

---

## 1. Evidence levels (read this first)

We distinguish four claims. **Do not conflate them in reports.**

| Level | Symbol | Meaning | How to prove |
|-------|--------|---------|--------------|
| **Exists** | EX | Node/topic appears in `ros2 node/topic list` | Snapshot or live list |
| **Wired** | WI | Publisher and subscriber share a topic (`topic info -v`) | Graph topology |
| **Fed** | FE | Non-empty messages at runtime (`topic hz`, `echo`) | Live audit, car/sim running |
| **Used** | US | Downstream **decisions** change if input removed | A/B test, planner debug factors |

**Examples from this project:**

| Claim | Level | Evidence |
|-------|-------|----------|
| `clustering/objects` topic exists | EX | In topic list |
| `multi_object_tracker` subscribes to `centerpoint/validation/objects` | WI | `ros2 node info multi_object_tracker` |
| `dsgn_offline` publishes at 10 Hz | FE | `ros2 topic hz` |
| Fictitious dsgn cars trigger obstacle stop | US | dsgn on/off while driving |
| `detection/objects` merged topic | EX + WI? | Exists; **0 subscribers** → **not fed forward** |

---

## 2. Does the car need to be moving? Route vs data flow

**Short answer:** Most **sensing and perception** pipelines run whenever AWSIM + Autoware are up and localization is initialized. **Route** and **engage** change *what planning outputs*, not whether LiDAR publishes.

| Subsystem | Needs AWSIM running | Needs pose init | Needs route set | Needs engage (Auto) | Needs ego moving |
|-----------|--------------------|-----------------|-----------------|---------------------|------------------|
| `/clock`, LiDAR, IMU | Yes | No | No | No | No |
| NDT + EKF localization | Yes | Yes (2D pose) | No | No | Helps; can work stationary |
| CenterPoint, obstacle seg, TL camera | Yes | Yes (for map frame) | No | No | No |
| `dsgn_offline` | Yes | Yes | No | No | No (uses pose, not speed) |
| Mission route `/planning/mission_planning/route` | — | Yes | **Yes** | No | No |
| Behavior/motion planning trajectory | Yes | Yes | Yes | Usually | Not always (can plan stop at 0 speed) |
| `/control/command/control_cmd` → vehicle | Yes | Yes | Yes | **Yes** | No (can hold brake) |

**For your audit:** Capture `audit_stack_usage.sh` with tags like `stationary_routed`, `engaged_driving`, `dsgn_on` in the filename. Compare reports.

---

## 3. End-to-end driving inputs (supervisor summary)

What actually shapes autonomous motion on our straight-line Nishi-Shinjuku tests:

```text
┌───────────── AWSIM ─────────────┐
│ 3× LiDAR, IMU, vehicle status, │
│ traffic-light camera, /clock    │
└───────────────┬─────────────────┘
                ▼
┌──────── Map (once) ────────────┐     ┌── dsgn_offline (overlay) ──┐
│ vector_map, pointcloud_map     │     │ pose → KITTI txt → boxes   │
└───────────────┬────────────────┘     └─────────────┬──────────────┘
                ▼                                      │
┌──────── Localization ──────────┐                     │
│ NDT → EKF pose / twist         │─────────────────────┘
└───────────────┬────────────────┘
                ▼
┌──────── Perception ────────────────────────────────────────────┐
│ A) Objects: CenterPoint → validator ─┐                         │
│    detection_by_tracker ─────────────┼→ tracker → prediction   │
│    dsgn_offline ─────────────────────┘   → /perception/.../objects │
│ B) Obstacle points: obstacle_segmentation → AEB                │
│ C) Traffic lights: camera pipeline (+ green bridge override)     │
│ D) Occupancy grid (parking / costmap; minor in lane driving)    │
└───────────────┬────────────────────────────────────────────────┘
                ▼
┌──────── Planning ──────────────────────────────────────────────┐
│ route → behavior_path → behavior_velocity → motion_velocity    │
│ (obstacle_stop / slow / cruise use predicted objects)          │
│ → scenario_trajectory → velocity_smoother                      │
└───────────────┬────────────────────────────────────────────────┘
                ▼
┌──────── Control ───────────────────────────────────────────────┐
│ trajectory_follower → vehicle_cmd_gate → AWSIM vehicle         │
│ parallel: AEB, collision_detector (objects + pointcloud)       │
└────────────────────────────────────────────────────────────────┘
```

---

## 4. All 194 nodes — by subsystem

**Legend:** Primary evidence from frozen graph + targeted live probes.  
`FE/US` = confirmed fed or used only after live audit / experiment.

### 4.1 Simulation (2 nodes) — **EX, FE**

| Node | Role | Publishes (main) | Evidence |
|------|------|------------------|----------|
| `/AWSIM` | Simulator bridge | `/clock`, `/vehicle/status/*`, sensor raw topics | EX; FE when sim running |
| `/RobotecGPULidar` | GPU LiDAR driver in sim | LiDAR-related streams | EX |

### 4.2 Sensing — Autoware preprocessors (20 nodes) — **EX; FE when sim up**

| Node | Role |
|------|------|
| `/sensing/lidar/top/*` | Top LiDAR: distortion, crop, ring filter |
| `/sensing/lidar/left/*` | Left LiDAR chain |
| `/sensing/lidar/right/*` | Right LiDAR chain |
| `/sensing/lidar/concatenate_data` | Sync + fuse → `/sensing/lidar/concatenated/pointcloud` |
| `/sensing/imu/imu_corrector` | IMU correction |
| `/sensing/imu/gyro_bias_scale_validator` | IMU validation |
| `/sensing/vehicle_velocity_converter` | Wheel speed → twist |

**Key output topics:**

| Topic | Fed when |
|-------|----------|
| `/sensing/lidar/{top,left,right}/pointcloud_raw` | AWSIM running |
| `/sensing/lidar/{top,left,right}/pointcloud` | After preprocess |
| `/sensing/lidar/concatenated/pointcloud` | **Main LiDAR bus** → perception + localization |
| `/sensing/lidar/concatenated/pointcloud/cuda` | GPU copy for CenterPoint |
| `/sensing/imu/imu_data` | Always with sim |
| `/sensing/camera/traffic_light/image_raw` | TL perception input |
| `/sensing/gnss/pose_with_covariance` | EX in graph; **likely idle** in AWSIM-only localization |

### 4.3 Map (6 nodes) — **EX, FE (latched once at startup)**

| Node | Output topic |
|------|--------------|
| `lanelet2_map_loader` | `/map/vector_map` |
| `pointcloud_map_loader` | `/map/pointcloud_map` |
| `map_projection_loader` | `/map/map_projector_info` |
| `lanelet2_map_visualization` | `/map/vector_map_marker` |
| `vector_map_tf_generator` | TF map frames |
| `map_hash_generator` | API hash |

**Used by:** all planners, prediction, NDT, traffic-light map detector.

### 4.4 Localization (14 nodes) — **EX; FE after pose init**

| Node | Function | Key output |
|------|----------|------------|
| `ndt_scan_matcher` | Scan-to-map pose | `/localization/pose_estimator/pose_with_covariance` |
| `gyro_odometer` | Twist from IMU | `/localization/twist_estimator/twist_with_covariance` |
| `ekf_localizer` | Fuse pose + twist | `/localization/pose_with_covariance`, `/localization/kinematic_state` |
| `stop_filter`, `twist2accel` | Post-process motion | kinematic chain |
| `pose_initializer`, `automatic_pose_initializer` | Initial pose services | — |
| `crop_box_filter_measurement_range`, downsample filters | NDT input clouds | `/localization/util/downsample/pointcloud` |
| `localization_error_monitor` | Diagnostics | `/diagnostics` |

**Used by:** entire stack; **dsgn_offline** reads `/localization/pose_with_covariance`.

### 4.5 Perception — object recognition (15 nodes)

| Node | Output topic | Wired downstream? | Fed? | Used? |
|------|--------------|---------------------|------|-------|
| `lidar_centerpoint` | `.../centerpoint/objects` | → validator | FE (sim) | WI |
| `obstacle_pointcloud_based_validator_node` | `.../centerpoint/validation/objects` | → **tracker** | FE | WI |
| **`dsgn_offline`** | same validation topic | → **tracker** | FE @ 10Hz | **US** (A/B proven) |
| `euclidean_cluster` | `.../clustering/clusters` | internal | FE | WI only |
| `shape_estimation` | `.../objects_with_feature` | → feature_remover | FE | WI only |
| `detected_object_feature_remover` | `.../clustering/objects` | **none (0 subs)** | FE | **NOT USED** in object pipeline |
| `detection_by_tracker_node` | `.../detection_by_tracker/objects` | → **tracker** | FE? | WI; US unproven alone |
| `voxel_based_compare_map_filter` | filtered cloud for detection | → centerpoint chain | FE | WI |
| `voxel_grid_downsample_filter` | downsample | internal | FE | WI |
| **`multi_object_tracker`** | `.../tracking/objects` | → prediction | FE | **US** (via objects) |
| **`map_based_prediction`** | **`/perception/object_recognition/objects`** | → **12 subscribers** | FE | **US** |
| `perception_analytics_publisher` | analytics metrics | logging | FE | metrics only |

**Tracker inputs (proven wired):**

- `/perception/object_recognition/detection/centerpoint/validation/objects`
- `/perception/object_recognition/detection/detection_by_tracker/objects`
- `/perception/object_recognition/detection/camera_only/objects`

**Orphan topic (proven NOT wired):**

- `/perception/object_recognition/detection/objects` — legacy; **0 subscribers**

### 4.6 Perception — obstacle segmentation + occupancy (4 nodes)

| Node | Output | Consumers | Used? |
|------|--------|-----------|-------|
| `crop_box_filter`, `common_ground_filter`, `occupancy_grid_based_outlier_filter` | obstacle pointcloud chain | validator, AEB, OGM | WI; US for AEB likely |
| `occupancy_grid_map_node` | `/perception/occupancy_grid_map/map` | parking costmap | Conditional (parking scenario) |

**Key topic:** `/perception/obstacle_segmentation/pointcloud` — monitored by `/system/topic_state_monitor_obstacle_segmentation_pointcloud`.

### 4.7 Perception — traffic lights (15 nodes)

| Stage | Node(s) | Output |
|-------|---------|--------|
| Decompress / relay | `traffic_light_image_decompressor`, `traffic_light_camera_info_relay` | images |
| Map ROIs | `traffic_light_map_based_detector` | expect/rois |
| Fine detector | `traffic_light_fine_detector` | rois |
| Classifiers | car / ped classifiers, occlusion predictor | class signals |
| Fusion / arbiter | `traffic_light_multi_camera_fusion`, `traffic_light_arbiter` | judged signals |
| Crosswalk estimator | `crosswalk_traffic_light_estimator` | — |
| **Final** | (merged) | `/perception/traffic_light_recognition/traffic_signals` |

**Lab overlay:** `traffic_light_green_bridge.py` → `/perception/traffic_light_recognition/external/traffic_signals` (**US** for straight-line tests — without it, car stops at lights).

**Consumer:** `behavior_velocity_planner` (traffic_light module).

### 4.8 Mission + scenario planning (29 nodes)

| Node | Role | Route needed? | Uses predicted objects? |
|------|------|---------------|-------------------------|
| `mission_planner`, `route_selector` | Global route | **Yes** | No |
| `scenario_selector` | lane_driving vs parking | Yes | No |
| `behavior_path_planner` | Path geometry, avoidance | Yes | **Yes** |
| `behavior_velocity_planner` | Rules, TL, intersections | Yes | **Yes** |
| `motion_velocity_planner` | **obstacle_stop/slow/cruise** | Yes | **Yes** |
| `path_optimizer`, `elastic_band_smoother` | Path shape | Yes | No |
| `velocity_smoother` | Jerk limits on trajectory | Yes | No |
| `costmap_generator`, `freespace_planner` | Parking | Parking scenario | Partial |
| `planning_validator`, `planning_evaluator` | Safety / metrics | Yes | Partial |
| `remaining_distance_time_calculator` | HMI | Yes | No |
| `goal_pose_visualizer`, `manual_lane_change_handler` | UI / manual | — | No |

**Primary driving output:** `/planning/scenario_planning/trajectory` — monitored by system topic_state_monitor.

### 4.9 Control (15 nodes)

| Node | Role | Key inputs | Used? |
|------|------|------------|-------|
| `controller_node_exe` (trajectory_follower) | Track planned trajectory | trajectory, kinematic_state | **US** when engaged |
| `vehicle_cmd_gate` | Mode gating, emergency override | control_cmd, MRM | **US** |
| `autonomous_emergency_braking` | Emergency brake | **predicted objects + obstacle pointcloud** | Conditional |
| `collision_detector` | Collision check | predicted objects | WI |
| `autoware_operation_mode_transition_manager` | Engage/disengage | API state | **US** |
| `autoware_shift_decider` | Gear | — | US |
| `control_validator`, `control_evaluator` | Safety / metrics | — | Support |
| `external_cmd_selector`, `external_cmd_converter` | Manual override paths | — | Idle in Auto |
| `lane_departure_checker_node` | Departure warning | — | Support |

**Vehicle command output:** `/control/command/control_cmd` → AWSIM.

### 4.10 System + fail-safe (28 nodes)

| Category | Nodes | Role |
|----------|-------|------|
| MRM | `mrm_handler`, emergency/comfortable stop operators | Safe stop on fault |
| Monitors | `topic_state_monitor_*` (13 topics) | **Prove pipeline "fed"** — see §6 |
| Health | `component_state_monitor`, `duplicated_node_checker`, `processing_time_checker` | Launch health |
| Diagnostics | `aggregator`, `hazard_status_converter`, `logging_diag_graph` | `/diagnostics`, emergency |
| Pipeline latency | `pipeline_latency_monitor` | End-to-end timing |

### 4.11 AD API (20 nodes under `/adapi/`) — **EX; mirror of stack state**

Expose `/api/*` topics (routing, perception objects, operation mode, diagnostics). Used by RViz adaptors and tools — **not separate perception**, mirrors internal topics.

### 4.12 Infrastructure (not semantic data paths) — **EX only**

| Node pattern | Count (approx) | Note |
|--------------|----------------|------|
| `transform_listener_impl_*` | 19 | TF helpers — no driving semantics |
| `*_container` | 8 | Composable node hosts |
| `/launch_ros_80` | 1 | Launch parent |
| `/robot_state_publisher` | 1 | URDF / joints |
| `/rviz` | 1 | Visualization |
| `/trajectory_relay` | 1 | Relay |
| `/raw_vehicle_cmd_converter` | 1 | Actuation mapping |
| `/default_adapi/helpers/*` | 3 | RViz routing/pose adaptors |
| `/pointcloud_container` | 1 | GPU perception container |

---

## 5. Topic inventory by function (796 total)

Full list: `logs/topics_with_dsgn_offline.txt`. Below: **semantic outputs only** (excluding `/debug/` and `processing_time`).

### 5.1 Critical driving topics (audit these)

| Topic | Type | Publishers (typical) | Subscribers (driving-relevant) | Min evidence |
|-------|------|----------------------|--------------------------------|--------------|
| `/sensing/lidar/concatenated/pointcloud` | PointCloud2 | concatenate_data | NDT, CenterPoint, obstacle seg | FE |
| `/localization/pose_with_covariance` | Pose | EKF | planners, dsgn | FE |
| `/localization/kinematic_state` | Odometry | EKF | control, planners | FE |
| `/map/vector_map` | LaneletMap | map_loader | planners, prediction | FE (latched) |
| `/perception/object_recognition/detection/centerpoint/validation/objects` | DetectedObjects | validator, **dsgn** | **tracker** | FE + US |
| `/perception/object_recognition/tracking/objects` | TrackedObjects | tracker | prediction, det_by_tracker | FE |
| `/perception/object_recognition/objects` | PredictedObjects | prediction | **motion_velocity_planner**, behavior planners, AEB | FE + US |
| `/perception/obstacle_segmentation/pointcloud` | PointCloud2 | obstacle seg | AEB, OGM | FE |
| `/perception/traffic_light_recognition/traffic_signals` | TrafficSignalArray | TL stack | behavior_velocity_planner | FE |
| `/planning/mission_planning/route` | LaneletRoute | mission_planner | scenario planning | FE when routed |
| `/planning/scenario_planning/trajectory` | Trajectory | velocity_smoother | trajectory_follower | FE when engaged |
| `/planning/planning_factors/obstacle_stop` | PlanningFactors | motion_velocity_planner | debug / HMI | US when stopping |
| `/control/command/control_cmd` | Control | vehicle_cmd_gate | AWSIM vehicle | FE when engaged |

### 5.2 Known orphan / dead-end outputs

| Topic | Status |
|-------|--------|
| `/perception/object_recognition/detection/objects` | **0 subscribers** — do not use for injection |
| `/perception/object_recognition/detection/clustering/objects` | **0 subscribers** — clustering runs but output unused by tracker |

### 5.3 Debug-only topics (~400+)

Topics under `/debug/`, `processing_time_ms`, `virtual_wall`, `marker`, compressed image relays — **instrumentation only**, not driving inputs. Safe to ignore in supervisor reports unless profiling.

---

## 6. System monitors — Autoware's own "is it fed?" checks

These `topic_state_monitor` nodes exist precisely to detect **missing publishers** on critical topics:

| Monitor node | Watched topic |
|--------------|---------------|
| `topic_state_monitor_object_recognition_objects` | `/perception/object_recognition/objects` |
| `topic_state_monitor_obstacle_segmentation_pointcloud` | `/perception/obstacle_segmentation/pointcloud` |
| `topic_state_monitor_pose_twist_fusion_filter_pose` | EKF pose |
| `topic_state_monitor_scenario_planning_trajectory` | `/planning/scenario_planning/trajectory` |
| `topic_state_monitor_traffic_light_recognition_traffic_signals` | TL output |
| `topic_state_monitor_control_command_control_cmd` | control cmd |
| `topic_state_monitor_mission_planning_route` | route |
| `topic_state_monitor_vector_map` | vector map |
| `topic_state_monitor_pointcloud_map` | point cloud map |
| `topic_state_monitor_transform_map_to_base_link` | TF |
| `topic_state_monitor_vehicle_status_velocity_status` | ego speed |
| `topic_state_monitor_vehicle_status_steering_status` | steering |

If a monitor trips, Autoware may block engage or trigger MRM — **fed ≠ used**, but **fed is necessary**.

---

## 7. Transformation chain (inputs → decisions)

| Step | Input | Transform node(s) | Output |
|------|-------|-------------------|--------|
| 1 | Sim LiDAR raw | sensing preprocess ×3 + concatenate | fused pointcloud |
| 2 | Fused cloud + map | NDT scan matcher | pose estimate |
| 3 | Pose + IMU + wheel | EKF | `pose_with_covariance`, `kinematic_state` |
| 4 | Fused cloud + map | CenterPoint | 3D detections |
| 5 | Detections + cloud | obstacle_pointcloud validator | validated detections |
| 5b | Pose + path.txt | **dsgn_offline** | validated detections (injected) |
| 6 | Validated + det_by_tracker + camera | multi_object_tracker | tracked objects |
| 7 | Tracked + vector_map | map_based_prediction | **predicted objects** |
| 8 | Predicted objects + route + map | behavior_path + behavior_velocity | path with rules |
| 9 | Path + predicted objects | motion_velocity_planner | speed profile (stop/slow) |
| 10 | Trajectory | velocity_smoother → trajectory_follower | steering/throttle |
| 11 | Parallel: obstacle points | AEB | emergency override |

---

## 8. What we have **proven** vs **assumed**

### Proven (report confidently to supervisor)

1. **dsgn_offline → obstacle stop:** A/B test (node on/off) changes stopping for fictitious cars.
2. **Tracker wiring:** `multi_object_tracker` subscribes only to validation, det_by_tracker, camera_only — **not** `detection/objects`.
3. **Clustering not in object pipeline:** `clustering/objects` has no downstream subscriber.
4. **Planners subscribe to predicted objects:** 12 subscribers on `/perception/object_recognition/objects` including `motion_velocity_planner`.
5. **796 topics / 194 nodes** in our launch with dsgn overlay — reproducible via `freeze_ros_graph.sh`.

### Assumed (run `audit_stack_usage.sh` to confirm per session)

- CenterPoint publishes non-empty detections during sim
- detection_by_tracker contributes measurably vs CenterPoint alone
- AEB uses obstacle pointcloud on our routes (not just predicted objects)
- Traffic-light pipeline fed vs bypassed entirely by green bridge

### Recommended capture protocol before supervisor meeting

```bash
# inside Docker — stationary, routed, not engaged
bash /home/aw/scripts/audit_stack_usage.sh routed_stationary

# inside Docker — engaged, driving, dsgn on
bash /home/aw/scripts/audit_stack_usage.sh engaged_dsgn_on

# inside Docker — engaged, driving, dsgn off
bash /home/aw/scripts/audit_stack_usage.sh engaged_dsgn_off
```

Attach the three reports from `~/summer26/logs/` plus this document.

---

## 9. Changelog

| Date | Change |
|------|--------|
| 2026-06-29 | Initial audit from frozen graph + dsgn A/B evidence |
