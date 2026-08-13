#!/usr/bin/env bash
# Launch AWSIM Labs v1.6.1 (binary) inside the Autoware container with GUI on the
# xrdp desktop.
#
# CRITICAL: AWSIM's ros2-for-unity plugin is a *standalone* build. If it detects a
# sourced ROS 2 (via ROS_DISTRO / AMENT_PREFIX_PATH) it uses the system ROS 2 libs
# instead of its bundled ones, and every C# publisher then fails to be created with
# "topic name is invalid" -- no /clock, no camera, no vehicle status. Those vars are
# baked into the Autoware image as Docker ENV, so they must be scrubbed explicitly.
# Do NOT `source /opt/ros/humble/setup.bash` before launching AWSIM.
#
# Usage:
#   scripts/awsim/awsim_launch.sh [pristine|modded] [extra awsim args...]
set -euo pipefail

BUILD="${1:-pristine}"; shift || true

case "$BUILD" in
  pristine) SUBDIR="extracted/awsim_labs_v1.6.1" ;;
  modded)   SUBDIR="modded/awsim_labs_v1.6.1" ;;
  *) echo "usage: $0 [pristine|modded] [extra args...]" >&2; exit 2 ;;
esac

AWSIM_HOST_DIR="$HOME/summer26/data/awsim"
if [[ ! -x "$AWSIM_HOST_DIR/$SUBDIR/awsim_labs.x86_64" ]]; then
  echo "ERROR: no AWSIM binary at $AWSIM_HOST_DIR/$SUBDIR" >&2
  exit 1
fi

NAME="${AWSIM_CONTAINER:-awsim_${BUILD}}"
DISPLAY_ID="${AWSIM_DISPLAY:-:10}"
DOMAIN="${ROS_DOMAIN_ID:-26}"
WIDTH="${AWSIM_WIDTH:-1280}"
HEIGHT="${AWSIM_HEIGHT:-720}"
LOG="awsim_${BUILD}_$(date +%Y%m%d_%H%M%S).log"

# Resolved inside the container.
A="/home/aw/awsim/$SUBDIR"
CONFIG="${AWSIM_CONFIG:-$A/awsim_labs_Data/StreamingAssets/config.json}"

docker rm -f "$NAME" >/dev/null 2>&1 || true

docker run --rm -d \
  --name "$NAME" \
  --device nvidia.com/gpu=all \
  --network host \
  -e DISPLAY="$DISPLAY_ID" \
  -e HOME=/home/aw \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v "$AWSIM_HOST_DIR:/home/aw/awsim" \
  -v "$HOME/summer26/logs:/home/aw/logs:rw" \
  --entrypoint /bin/bash \
  ghcr.io/autowarefoundation/autoware:universe-cuda-humble \
  -c "exec env -u ROS_DISTRO -u AMENT_PREFIX_PATH -u CYCLONEDDS_URI -u COLCON_PREFIX_PATH \
        ROS_DOMAIN_ID=$DOMAIN RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
        '$A/awsim_labs.x86_64' --config '$CONFIG' \
        -screen-width $WIDTH -screen-height $HEIGHT -screen-fullscreen 0 \
        -logFile /home/aw/logs/$LOG $*"

echo "container : $NAME"
echo "build     : $BUILD ($SUBDIR)"
echo "log       : $HOME/summer26/logs/$LOG"
echo
echo "Scene load takes ~60-75 s. Verify with:"
echo "  scripts/awsim/awsim_verify.sh $NAME"
