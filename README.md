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
7. **Sudo is run by adria, not by the agent.** The agent cannot enter a `sudo` password. When a task needs `sudo` (almost always `sudo docker ...`), **stop and give adria the exact command to copy-paste** — one command at a time, labeled `host` vs `inside Docker`, with a one-line reason. Do not assume adria knows the project; include enough context to run it blindly.

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

**Docker access:** `adria` is not in the `docker` group — `docker` commands need `sudo` and an interactive password. Agents must delegate those to adria.

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
│   │   ├── traffic_light_green_bridge.py
│   │   ├── apply_route_from_osm.py
│   │   ├── route_candidates*.json       # generated start/goal poses
│   │   └── (recovery / diagnostic shell scripts)
│   └── awsim/
│       ├── awsim_labs_v1.6.1.zip
│       └── extracted/                   # AWSIM binary
│   └── arka/                            # AWSIM dataset from prior intern (gitignored)
├── scripts/
│   ├── find_route_candidates.py         # host: parse OSM map → route JSON
│   ├── apply_route_candidates.sh        # host wrapper (needs sudo docker exec)
│   ├── 01_gui_preflight.sh              # GUI / display sanity check
│   └── download_autoware_artifacts_from_role.py  # one-time model download
├── external/
│   ├── autoware/                        # upstream Autoware reference (gitignored)
│   └── DSGN_custom/                     # stereo 3D detector — separate git repo (gitignored)
├── logs/                                # script output logs (gitignored)
├── notes/
│   └── DEBUG_LOG.md                     # pitfalls, diagnostics chain, changelog
├── models/                              # ML configs / experiment notes (weights gitignored)
└── src/
    └── dsgn_offline/                    # ROS 2 offline perception bridge — separate git repo (gitignored)
```

**Map:** Nishi-Shinjuku (`data/maps/nishishinjuku_autoware_map/`). MGRS grid `54SUE`. Elevation ~40.9 m — never use z=0.

### DSGN repos (separate forks, not part of this repo)

`src/dsgn_offline/` and `external/DSGN_custom/` are **independent git checkouts** from the prior intern’s work, listed in `.gitignore` so this repo does not track them as nested submodules or ghost folders.

| Path | Role | Where to commit changes |
|------|------|-------------------------|
| `external/DSGN_custom/` | Train/run DSGN; write detection `.txt` files | Your fork of [DSGN_custom](https://github.com/DF-Autoware-AWSIM/DSGN_custom) |
| `src/dsgn_offline/` | Publish offline detections into Autoware (`/perception/object_recognition/detection/objects`) | Your fork: [adricalm/dsgn_offline](https://github.com/adricalm/dsgn_offline) (`upstream` = Arka’s repo) |

**Why:** keeps Autoware integration (`summer26`) separate from ML/ROS code you will modify; avoids accidental commits of large datasets or model weights; preserves a clear diff against the prior baseline via `upstream`.

**Workflow:** edit inside the nested repo → `git commit` / `git push` there. Only commit integration docs, scripts, and small configs in `summer26`. Clone the forks locally after a fresh checkout (they are not bundled in this repo).

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

Two things are required beyond the base launch:

| Mount / flag | Why |
|---|---|
| `ndt_scan_matcher.param.yaml` bind-mount | Nishi-Shinjuku scores ~2.2–2.4; default threshold 2.3 trips MRM `emergency_stop` while driving |
| `launch_rviz_adaptors:=true` | Bridges RViz **2D Rough Goal Pose** clicks to `/api/routing/set_route_points` |

```bash
sudo docker stop autoware_full_test 2>/dev/null || true
sudo docker run --rm -d \
  --name autoware_full_test \
  --device nvidia.com/gpu=all \
  --network host \
  -e HOME=/home/aw \
  -e ROS_DOMAIN_ID=26 \
  -v "$HOME/summer26/data/maps:/home/aw/maps:ro" \
  -v "$HOME/summer26/data/autoware_data:/home/aw/autoware_data" \
  -v "$HOME/summer26/data/autoware_data/ndt_scan_matcher.param.yaml:/opt/autoware/autoware_launch/share/autoware_launch/config/localization/ndt_scan_matcher/ndt_scan_matcher.param.yaml" \
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

### AWSIM (inside Docker, with GUI)

AWSIM runs inside Docker (needs GUI access via `DISPLAY`/X11).

---

## Manual drive workflow (RViz)

Once Autoware and AWSIM are up:

1. Open RViz with the Autoware config: `rviz2 -d /opt/autoware/autoware_launch/share/autoware_launch/rviz/autoware.rviz`
2. **Fixed Frame:** set to `viewer` (not `map` — MGRS coords cause click errors).
3. **Initialize localization:** use the **2D Pose Estimate** tool and click on the car's position in the map.
4. **Set route:** use the **2D Rough Goal Pose** tool (Autoware's, not the default one) and click a destination lane 30–80 m ahead.
5. **Engage:** press the **Auto** button in the Autoware panel. The car should start driving (`/autoware/state` → 5).

**If the car stops at intersections:** the traffic-light green bridge may not be running. Start it inside the container:

```bash
python3 /home/aw/autoware_data/traffic_light_green_bridge.py \
  /home/aw/maps/nishishinjuku_autoware_map/lanelet2_map.osm &
```

---

## Scripted routing (alternative to RViz, useful for reproducible experiments)

**Host** — generate route JSON from the OSM map:

```bash
python3 ~/summer26/scripts/find_route_candidates.py \
  --json-out ~/summer26/data/autoware_data/route_candidates.json
```

**Inside Docker** — apply localization + route:

```bash
python3 /home/aw/autoware_data/apply_route_from_osm.py \
  /home/aw/autoware_data/route_candidates.json
```

**Verify:**

```bash
ros2 topic echo --once /api/routing/state    # expect state: 2
ros2 topic echo --once /autoware/state       # expect state: 4
```

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

## Helper scripts (inside Docker unless noted)

| Script | Purpose |
|--------|---------|
| `data/autoware_data/traffic_light_green_bridge.py` | Publishes all map traffic lights as GREEN so the car doesn't stop at intersections |
| `data/autoware_data/apply_route_from_osm.py` | Scripted localize + set route (alternative to RViz) |
| `data/autoware_data/recover_after_awsim_restart.sh` | Full recovery after AWSIM restart (wait for clock → stop → bridge → route → engage) |
| `data/autoware_data/quick_motion_check.sh` | 6-line snapshot: bridge running? state? velocity? |
| `data/autoware_data/diagnose_stuck.sh` | Why is the car stopped? (MRM / obstacle / localization) |
| `data/autoware_data/inspect_emergency.sh` | Which diagnostic is causing hazard/MRM emergency? |
| `data/autoware_data/verify_stack_ready.sh` | Post-restart health check before engaging |
| `data/autoware_data/recover_from_mrm_emergency_stop.sh` | Recover from MRM latch caused by localization degradation |
| `scripts/find_route_candidates.py` | **Host:** parse OSM map → route JSON |
| `scripts/apply_route_candidates.sh` | **Host wrapper:** find_route_candidates + apply inside container (needs sudo) |
| `scripts/01_gui_preflight.sh` | **Host:** GUI / display sanity checks |

---

## Research direction

1. Inject **visual/physical adversarial patches** into the perception pipeline (DSGN-based setup from prior intern work).
2. Measure whether perception errors affect **driving behavior** (planning, control).
3. Evaluate **defense / recovery** methods.

See `models/` for experiment configs and the DSGN forks above for perception code.

---

## External references

- Autoware upstream: `external/autoware/` (see [`notes/external_autoware_ref.md`](notes/external_autoware_ref.md))
- DSGN stereo detector: `external/DSGN_custom/` — fork and commit separately (see layout above)
- DSGN offline ROS bridge: `src/dsgn_offline/` — fork and commit separately (see layout above)
- Map: Autoware Nishi-Shinjuku sample (`lanelet2_map.osm` + pointcloud)
- AWSIM Labs v1.6.1: `data/awsim/`
- AWSIM training data (prior intern): `data/arka/` (on disk only, not in git)
- Pitfalls, diagnostics chain, changelog: [`notes/DEBUG_LOG.md`](notes/DEBUG_LOG.md)
