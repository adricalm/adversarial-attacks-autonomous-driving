## AWSIM–Autoware startup

This is intentionally a Docker setup. The server host is newer than the supported Autoware host OS, and AWSIM's Unity ROS plugin needs to use the same Docker/host-network DDS environment as Autoware. 

### What is configurable, and what is a project convention?

- `ROOT` is the only path that normally needs changing. It is the checkout directory; default is `~/summer26`.
- Keep `DOMAIN=26` for this checkout. A ROS domain is not machine-specific, but every ROS process must use the same one. Several helper scripts currently embed `26`, so changing it means updating those scripts consistently.
- `MAP_NAME`, `IMAGE`, and `AUTOWARE_CONTAINER` are named settings below.
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

### Start AWSIM

Run from the **local xrdp graphical desktop** (not plain SSH). Use the actual `DISPLAY` if it is not `:10` (`echo "$DISPLAY"`).

```bash
# on host, once
export DISPLAY=:10
xhost +local:root   # or: xhost +local:

cd ~/summer26/data/awsim
sudo docker run --rm -it \
  --name awsim_gui_test \
  --device nvidia.com/gpu=all \
  --network host \
  -e DISPLAY=:10 \
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
    ./modded/awsim_labs_v1.6.1/awsim_labs.x86_64
  '
```

**Build choice:** use `modded/` for the stereo build (default above). For the pristine baseline, switch to `./extracted/awsim_labs_v1.6.1/awsim_labs.x86_64`.

A normal-looking GUI without `/clock`, camera, and LiDAR topics is a failed launch. See [`AWSIM_STEREO_CAMERA.md`](AWSIM_STEREO_CAMERA.md) for the `ROS_DISTRO` race if that happens.

**Alternative:** `scripts/awsim/awsim_launch.sh [pristine|modded]` auto-loads via `--config` (detached container, no GUI clicking). The command above is the usual interactive workflow.

### Start RViz (optional)

Run from the xrdp desktop (same `DISPLAY` as AWSIM). Start after Autoware is up; RViz is optional for visualization and manual routing.

```bash
export DISPLAY=:10   # or echo "$DISPLAY" and use that value
xhost +local:root    # if not already done for AWSIM

sudo docker run --rm -it \
  --name autoware_rviz_test \
  --device nvidia.com/gpu=all \
  --network host \
  -e DISPLAY="$DISPLAY" \
  -e HOME=/home/aw \
  -e ROS_DOMAIN_ID=26 \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  --entrypoint /bin/bash \
  ghcr.io/autowarefoundation/autoware:universe-cuda-humble \
  -lc '
    source /opt/ros/humble/setup.bash
    source /opt/autoware/setup.bash
    unset CYCLONEDDS_URI
    export ROS_DOMAIN_ID=26
    rviz2 -d /opt/autoware/autoware_launch/share/autoware_launch/rviz/autoware.rviz
  '
```

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

- **`dsgn_offline`:** build/replay only after the vehicle completes the baseline route. Follow [`notes26/DSGN_OFFLINE_RUNBOOK.md`](notes26/DSGN_OFFLINE_RUNBOOK.md); the `src` and `scripts` mounts above are already present.
- **Stereo/data recording/patch experiments:** use the modded AWSIM build and the dedicated scripts/runbooks.
