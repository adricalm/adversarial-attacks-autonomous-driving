#!/usr/bin/env bash
# Host wrapper: copy bridge script into mounted volume and run inside Autoware Docker.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAP_PATH="${MAP_PATH:-/home/aw/maps/nishishinjuku_autoware_map/lanelet2_map.osm}"
LOG_DIR="${HOME}/summer26/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/11_traffic_light_bridge_${TIMESTAMP}.txt"

detect_autoware_container() {
  local pid cid
  pid="$(pgrep -f "e2e_simulator.launch" | head -1 || true)"
  if [[ -n "${pid}" && -r "/proc/${pid}/cgroup" ]]; then
    cid="$(grep -oE 'docker-[0-9a-f]{64}' "/proc/${pid}/cgroup" | head -1 | sed 's/^docker-//')"
    if [[ -n "${cid}" ]]; then
      echo "${cid:0:12}"
      return 0
    fi
  fi
  echo "autoware_full_test"
}

CONTAINER="${AUTOWARE_CONTAINER:-$(detect_autoware_container)}"
mkdir -p "${LOG_DIR}"

echo "==> Traffic light green bridge"
echo "Container: ${CONTAINER}"
echo "Timestamp: ${TIMESTAMP}"
echo "Log: ${LOG_FILE}"

echo "==> Copying bridge script into mounted autoware_data volume"
cp "${SCRIPT_DIR}/traffic_light_green_bridge.py" "${HOME}/summer26/data/autoware_data/traffic_light_green_bridge.py"
chmod +x "${HOME}/summer26/data/autoware_data/traffic_light_green_bridge.py"

echo "==> Diagnosis (before bridge)"
{
  echo "=== ros2 topic types ==="
  sudo docker exec "${CONTAINER}" bash -lc '
    source /opt/ros/humble/setup.bash
    source /opt/autoware/setup.bash
    unset CYCLONEDDS_URI
    export ROS_DOMAIN_ID=26
    ros2 topic type /perception/traffic_light_recognition/traffic_signals
    ros2 topic type /perception/traffic_light_recognition/external/traffic_signals
    echo "=== external topic info ==="
    ros2 topic info /perception/traffic_light_recognition/external/traffic_signals -v
    echo "=== external echo (once) ==="
    timeout 5 ros2 topic echo --once /perception/traffic_light_recognition/external/traffic_signals || true
  '
} | tee "${LOG_FILE}"

echo "==> Starting bridge in background inside Docker (Ctrl+C in that shell stops it)"
echo "    To stop later: sudo docker exec ${CONTAINER} pkill -f traffic_light_green_bridge.py"

sudo docker exec -d "${CONTAINER}" bash -lc "
  source /opt/ros/humble/setup.bash
  source /opt/autoware/setup.bash
  unset CYCLONEDDS_URI
  export ROS_DOMAIN_ID=26
  nohup python3 /home/aw/autoware_data/traffic_light_green_bridge.py '${MAP_PATH}' \
    >> /home/aw/autoware_data/traffic_light_green_bridge.log 2>&1 &
"

sleep 2

echo "==> Verification (after bridge)"
{
  echo "=== external topic info ==="
  sudo docker exec "${CONTAINER}" bash -lc '
    source /opt/ros/humble/setup.bash
    source /opt/autoware/setup.bash
    unset CYCLONEDDS_URI
    export ROS_DOMAIN_ID=26
    ros2 topic info /perception/traffic_light_recognition/external/traffic_signals -v
    echo "=== external echo (once) ==="
    timeout 5 ros2 topic echo --once /perception/traffic_light_recognition/external/traffic_signals || true
    echo "=== merged traffic_signals echo (once) ==="
    timeout 5 ros2 topic echo --once /perception/traffic_light_recognition/traffic_signals || true
    echo "=== bridge log tail ==="
    tail -20 /home/aw/autoware_data/traffic_light_green_bridge.log || true
  '
} | tee -a "${LOG_FILE}"

echo "==> Done. Full log: ${LOG_FILE}"
