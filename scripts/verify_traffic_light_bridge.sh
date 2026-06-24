#!/usr/bin/env bash
# Host wrapper: verify bridge + motion topics inside Autoware Docker.
set -euo pipefail

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
LOG_DIR="${HOME}/summer26/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/12_traffic_light_verify_${TIMESTAMP}.txt"
mkdir -p "${LOG_DIR}"

{
  echo "Container: ${CONTAINER}"
  echo "Timestamp: ${TIMESTAMP}"
  sudo docker exec "${CONTAINER}" bash -lc '
    source /opt/ros/humble/setup.bash
    source /opt/autoware/setup.bash
    unset CYCLONEDDS_URI
    export ROS_DOMAIN_ID=26

    echo "=== bridge process ==="
    pgrep -af traffic_light_green_bridge.py || echo "bridge not running"

    echo "=== external/traffic_signals ==="
    ros2 topic info /perception/traffic_light_recognition/external/traffic_signals -v
    timeout 5 ros2 topic echo --once /perception/traffic_light_recognition/external/traffic_signals || true

    echo "=== traffic_signals (arbiter output) ==="
    timeout 5 ros2 topic echo --once /perception/traffic_light_recognition/traffic_signals || true

    echo "=== autoware state ==="
    timeout 5 ros2 topic echo --once /autoware/state || true

    echo "=== control_cmd ==="
    timeout 5 ros2 topic echo --once /control/command/control_cmd || true

    echo "=== velocity_status ==="
    timeout 5 ros2 topic echo --once /vehicle/status/velocity_status || true
  '
} | tee "${LOG_FILE}"

echo "Wrote ${LOG_FILE}"
