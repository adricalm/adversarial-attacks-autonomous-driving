#!/usr/bin/env bash
# Build dsgn_offline inside autoware_full_test (needs src/ mount).
set -euo pipefail

source /opt/ros/humble/setup.bash
source /opt/autoware/setup.bash
unset CYCLONEDDS_URI
export ROS_DOMAIN_ID=26

WS=/home/aw/ros2_ws
PKG_SRC="${WS}/src/dsgn_offline"

if [[ ! -f "${PKG_SRC}/package.xml" ]]; then
  echo "ERROR: ${PKG_SRC}/package.xml not found."
  echo "Add to docker run: -v \"\$HOME/summer26/src:/home/aw/ros2_ws/src:ro\""
  exit 1
fi

mkdir -p "${WS}"
cd "${WS}"
colcon build --symlink-install --packages-select dsgn_offline

echo ""
echo "Build OK. In this shell, source the overlay before running the node:"
echo "  source ${WS}/install/setup.bash"
