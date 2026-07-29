#!/usr/bin/env bash
# In-container capture helper. Prefer: bash /home/aw/scripts/engage_capture_inside_container.sh
# Requires scripts mount: -v "$HOME/summer26/scripts:/home/aw/scripts:ro"
set -u

LOG=/home/aw/autoware_data/engage_capture.log
RESULT=/home/aw/autoware_data/engage_result.json
exit_code=1

: >"${LOG}"
{
  echo "=== engage capture start $(date -Iseconds) ==="
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

  if [[ -f "${RESULT}" ]]; then
    echo "--- engage_result.json ---"
    cat "${RESULT}"
  fi

  echo "=== engage capture end exit_code=${exit_code} $(date -Iseconds) ==="
} >>"${LOG}" 2>&1

exit "${exit_code}"
