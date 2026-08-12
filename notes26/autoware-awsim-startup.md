## AWSIM–Autoware startup

This is intentionally a Docker setup. The server host is newer than the supported Autoware host OS, and AWSIM's Unity ROS plugin needs to use the same Docker/host-network DDS environment as Autoware. 

### What is configurable, and what is a project convention?

- `ROOT` is the only path a new user should normally change. It is the checkout directory; mine was `~/summer26`.
- Keep `DOMAIN=26` for this checkout. A ROS domain is not machine-specific, but every ROS process must use the same one. Several helper scripts currently embed `26`, so changing it means updating those scripts consistently.
- `MAP_NAME`, `IMAGE`, `AUTOWARE_CONTAINER`, and `AWSIM_DISPLAY` are named settings below.
- The commands assume Docker GPU access is already enabled for the lab user (the `docker run … --device nvidia.com/gpu=all` commands must work).

### One-time asset check

Run on the host after obtaining the repository and the non-git assets. Large AWSIM binaries, maps, models, datasets, and the two nested research repositories are deliberately not committed to this repository.

```bash
export ROOT="${ROOT:-$HOME/summer26}"
export DOMAIN=26
export IMAGE=ghcr.io/autowarefoundation/autoware:universe-cuda-humble
export MAP_NAME=nishishinjuku_autoware_map
export AUTOWARE_CONTAINER=autoware_full_test

test -d "$ROOT/data/maps/$MAP_NAME"
test -d "$ROOT/data/autoware_data/ml_models"
test -x "$ROOT/data/awsim/extracted/awsim_labs_v1.6.1/awsim_labs.x86_64"
test -f "$ROOT/data/autoware_data/ndt_scan_matcher.param.yaml"
docker info >/dev/null
```

For a pristine (non-stereo) baseline of AWSIM, `extracted/` is sufficient. The `modded/` AWSIM binary is only for recording data in stereo format.

### Why the Autoware mounts exist

They are not arbitrary, but they fall into two categories:

- **Required assets:** `data/maps` supplies the map; `data/autoware_data` supplies ML models, route JSON, and experiment support files.
- **Map/experiment fixes:** the three individual configuration mounts override image defaults. The NDT threshold prevents a known Nishi-Shinjuku localization score from triggering MRM; the perception override disables CenterPoint so it does not publish alongside `dsgn_offline`; the AEB override prevents raw LiDAR clusters from stopping the car after that detector is disabled. Keep all three for the DSGN experiment. Remove them only when deliberately comparing against an upstream/default Autoware run.
- **Development plumbing:** `scripts` makes the documented helpers available in the container, and `src` exposes the `dsgn_offline` source when needed. Neither contains core Autoware assets, but both are part of this project's reproducible workflow.

`--network host` is required for the ROS 2/DDS participants to discover each other. `unset CYCLONEDDS_URI` prevents an inherited, machine-local DDS XML configuration from isolating the containers. Both Autoware and AWSIM must use the same `ROS_DOMAIN_ID`.

### Start a clean baseline

Start Autoware detached:

```bash
docker run --rm -d \
  --name "$AUTOWARE_CONTAINER" \
  --device nvidia.com/gpu=all \
  --network host \
  -e HOME=/home/aw \
  -e ROS_DOMAIN_ID="$DOMAIN" \
  -v "$ROOT/data/maps:/home/aw/maps:ro" \
  -v "$ROOT/data/autoware_data:/home/aw/autoware_data" \
  -v "$ROOT/data/autoware_data/ndt_scan_matcher.param.yaml:/opt/autoware/autoware_launch/share/autoware_launch/config/localization/ndt_scan_matcher/ndt_scan_matcher.param.yaml:ro" \
  -v "$ROOT/src:/home/aw/ros2_ws/src:ro" \
  -v "$ROOT/scripts:/home/aw/scripts:ro" \
  -v "$ROOT/data/autoware_data/perception.launch.xml.no_detection:/opt/autoware/tier4_perception_launch/share/tier4_perception_launch/launch/perception.launch.xml:ro" \
  -v "$ROOT/data/autoware_data/autonomous_emergency_braking.param.yaml:/opt/autoware/autoware_launch/share/autoware_launch/config/control/autoware_autonomous_emergency_braking/autonomous_emergency_braking.param.yaml:ro" \
  --entrypoint /bin/bash \
  "$IMAGE" -lc '
    source /opt/ros/humble/setup.bash
    source /opt/autoware/setup.bash
    unset CYCLONEDDS_URI
    export ROS_DOMAIN_ID='"$DOMAIN"'
    ros2 launch autoware_launch e2e_simulator.launch.xml \
      vehicle_model:=awsim_labs_vehicle \
      sensor_model:=awsim_labs_sensor_kit \
      map_path:=/home/aw/maps/'"$MAP_NAME"' \
      data_path:=/home/aw/autoware_data/ml_models \
      launch_vehicle_interface:=true \
      rviz:=false \
      rviz_respawn:=false \
      launch_rviz_adaptors:=true
  '

docker ps --filter "name=$AUTOWARE_CONTAINER"
```

Autoware normally needs roughly 2–3 minutes before it is useful. A running container only proves that the launch process has not exited; it does not prove that the system is ready.

### Start and verify AWSIM

Run this portion from the **local xrdp graphical desktop** (not a plain SSH session). `DISPLAY` must identify that desktop, usually `:10`; use the actual value printed by `echo "$DISPLAY"` rather than assuming it.

```bash
export AWSIM_DISPLAY="${AWSIM_DISPLAY:-$DISPLAY}"
: "${AWSIM_DISPLAY:?Open a terminal in the xrdp desktop or set AWSIM_DISPLAY (usually :10).}"
xhost +SI:localuser:root

cd "$ROOT"
ROS_DOMAIN_ID="$DOMAIN" AWSIM_DISPLAY="$AWSIM_DISPLAY" \
  bash scripts/awsim_launch.sh pristine

# Wait for scene loading (~60–75 s), then require the real ROS publishers.
ROS_DOMAIN_ID="$DOMAIN" bash scripts/awsim_verify.sh awsim_pristine
```

Use `scripts/awsim_launch.sh modded` only for the stereo build. Do not source `/opt/ros/...` before launching AWSIM: its bundled `ros2-for-unity` library must not inherit Autoware's ROS environment. The launch script removes those variables for this reason. `awsim_verify.sh` is the gate: a normal-looking GUI without `/clock`, camera, and LiDAR topics is a failed launch.

### Initialize, route, and drive

After AWSIM verification succeeds, open a shell in Autoware:

```bash
docker exec -it "$AUTOWARE_CONTAINER" bash
```

Then, inside that container:

```bash
bash /home/aw/autoware_data/verify_stack_ready.sh
bash /home/aw/scripts/drive_route_and_engage.sh \
  /home/aw/autoware_data/route_dsgn_ab.json
```

The drive helper clears the old route, publishes the route's initial pose (which teleports/initializes AWSIM), sets its goal, and requests autonomous mode. Give it a different route JSON only when its saved coordinates belong to this map.

### Optional layers, added only after baseline motion works

- **RViz:** optional visualization. The launch above includes `launch_rviz_adaptors:=true`, so RViz's *2D Rough Goal Pose* can set a route.
- **`dsgn_offline`:** build/replay only after the vehicle completes the baseline route. Follow [`notes26/DSGN_OFFLINE_RUNBOOK.md`](notes26/DSGN_OFFLINE_RUNBOOK.md); the `src` and `scripts` mounts above are already present.
- **Stereo/data recording/patch experiments:** use the modded AWSIM build and the dedicated scripts/runbooks. They are research layers, not startup prerequisites.
