#!/usr/bin/env bash
# Run inside the Autoware Docker container (no sudo needed there).
set -eo pipefail

# ROS setup.bash references optional env vars (e.g. AMENT_TRACE_SETUP_FILES).
set +u
source /opt/ros/humble/setup.bash
source /opt/autoware/setup.bash
set -u
unset CYCLONEDDS_URI
export ROS_DOMAIN_ID=26

MAP_PATH="${1:-/home/aw/maps/nishishinjuku_autoware_map/lanelet2_map.osm}"
SCRIPT="/home/aw/scripts/traffic_light_green_bridge.py"
LOG="/home/aw/autoware_data/traffic_light_green_bridge.log"
ARBITER_NODE="/perception/traffic_light_recognition/traffic_light_arbiter"

if [[ ! -f "${SCRIPT}" ]]; then
  echo "Missing ${SCRIPT}. Mount scripts: -v \"\$HOME/summer26/scripts:/home/aw/scripts:ro\"" >&2
  exit 1
fi

echo "==> Stopping any previous bridge process"
pkill -f traffic_light_green_bridge.py || true
sleep 1

echo "NOTE: Use external-input mode only. Do NOT use --bypass-arbiter once merged signals work;"
echo "      bypass + arbiter both publishing to traffic_signals causes MRM/planner flicker."

echo "==> Prefer external traffic-light source in arbiter (if node is up)"
if ros2 node list 2>/dev/null | grep -q traffic_light_arbiter; then
  ros2 param set "${ARBITER_NODE}" source_priority external 2>/dev/null || \
    ros2 param set "${ARBITER_NODE}" external_priority true 2>/dev/null || \
    echo "Could not set arbiter priority param (continuing)"
fi

echo "==> Starting traffic light green bridge (background, log: ${LOG})"
nohup python3 "${SCRIPT}" "${MAP_PATH}" >> "${LOG}" 2>&1 &
sleep 3

echo "=== /clock (sim time reference) ==="
timeout 5 ros2 topic echo --once /clock || true

echo "=== external/traffic_signals (once) ==="
timeout 5 ros2 topic echo --once /perception/traffic_light_recognition/external/traffic_signals || true

echo "=== traffic_signals arbiter output (once) ==="
timeout 5 ros2 topic echo --once /perception/traffic_light_recognition/traffic_signals || true

echo "Bridge log tail:"
tail -15 "${LOG}" || true
