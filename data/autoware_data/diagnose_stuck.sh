#!/usr/bin/env bash
# Why is the car stopped? (inside Docker) — MRM vs obstacle vs localization.
set -eo pipefail
set +u
source /opt/ros/humble/setup.bash
source /opt/autoware/setup.bash
set -u
unset CYCLONEDDS_URI
export ROS_DOMAIN_ID=26

echo "=== Autoware state / operation mode ==="
timeout 4 ros2 topic echo --once /autoware/state 2>/dev/null | grep -E "state:|stamp:" || true
timeout 4 ros2 topic echo --once /api/operation_mode/state 2>/dev/null \
  | grep -E "mode:|is_autonomous_mode_available:" || true

echo ""
echo "=== MRM + hazard (fail-safe layer) ==="
timeout 4 ros2 topic echo --once /api/fail_safe/mrm_state 2>/dev/null \
  | grep -E "state:|behavior:" || true
timeout 4 ros2 topic echo --once /system/emergency/hazard_status 2>/dev/null \
  | grep -E "emergency:|level:" | head -2 || true

echo ""
echo "=== Control output ==="
timeout 4 ros2 topic echo --once /control/command/control_cmd 2>/dev/null \
  | grep -A6 "longitudinal:" | head -8 || true
timeout 4 ros2 topic echo --once /vehicle/status/velocity_status 2>/dev/null \
  | grep "longitudinal_velocity:" || true

echo ""
echo "=== Obstacle stop (planning layer) ==="
timeout 4 ros2 topic echo --once /planning/planning_factors/obstacle_stop 2>/dev/null \
  | grep -E "module:|distance:|behavior:|is_safe:" || true

echo ""
echo "=== Detected objects (class + position) ==="
OBJ=$(timeout 4 ros2 topic echo --once /perception/object_recognition/objects 2>/dev/null || true)
echo "${OBJ}" | grep -E "object_id:|classification:|label:|position:" | head -24 || echo "(none)"

echo ""
echo "=== Localization faults (ERROR/WARN only) ==="
timeout 5 ros2 topic echo --once /diagnostics 2>/dev/null \
  | grep -B1 -A8 "scan_matching_status\|ekf_localizer" \
  | grep -E "name:|message:|level:" | head -20 || true

echo ""
echo "=== Trajectory (first 3 waypoint speeds) ==="
timeout 4 ros2 topic echo --once /planning/scenario_planning/trajectory 2>/dev/null \
  | grep "longitudinal_velocity_mps" | head -3 || true
