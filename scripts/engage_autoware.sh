#!/usr/bin/env bash
# Engage Autoware and verify motion (host). Requires scripts mount in Autoware container.
set -euo pipefail

RESULT_JSON="${HOME}/summer26/data/autoware_data/engage_result.json"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

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

echo "==> Engage Autoware vehicle motion"
echo "Container: ${CONTAINER}"
echo "Timestamp: ${TIMESTAMP}"

echo "==> Step 1-4: verify trajectory, engage, check control + velocity (inside Docker)"
docker exec "${CONTAINER}" bash -lc '
  source /opt/ros/humble/setup.bash
  source /opt/autoware/setup.bash
  unset CYCLONEDDS_URI
  export ROS_DOMAIN_ID=26

  echo "--- /autoware/state (before) ---"
  ros2 topic echo --once /autoware/state || true

  echo "--- /planning/scenario_planning/trajectory info ---"
  ros2 topic info /planning/scenario_planning/trajectory -v || true

  echo "--- trajectory velocity sample ---"
  ros2 topic echo --once /planning/scenario_planning/trajectory \
    | grep -E "longitudinal_velocity_mps" | head -20 || true

  python3 /home/aw/scripts/engage_autoware.py
  exit_code=$?

  echo "--- /autoware/state (after) ---"
  ros2 topic echo --once /autoware/state || true

  echo "--- /system/operation_mode/state ---"
  ros2 topic echo --once /system/operation_mode/state || true

  echo "--- /control/command/control_cmd info ---"
  ros2 topic info /control/command/control_cmd -v || true

  echo "--- /control/command/control_cmd sample ---"
  ros2 topic echo --once /control/command/control_cmd || true

  echo "--- /vehicle/status/velocity_status info ---"
  ros2 topic info /vehicle/status/velocity_status -v || true

  echo "--- /vehicle/status/velocity_status sample ---"
  ros2 topic echo --once /vehicle/status/velocity_status || true

  exit "${exit_code}"
'
exit_code=$?

echo "==> Engage result JSON"
if [[ -f "${RESULT_JSON}" ]]; then
  cat "${RESULT_JSON}"
else
  echo "Missing ${RESULT_JSON}"
fi
exit "${exit_code}"
