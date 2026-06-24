#!/usr/bin/env bash
# Six-line motion snapshot (inside Docker). No huge dumps.
set -eo pipefail
set +u
source /opt/ros/humble/setup.bash
source /opt/autoware/setup.bash
set -u
unset CYCLONEDDS_URI
export ROS_DOMAIN_ID=26

echo "1) bridge:" $(pgrep -af traffic_light_green_bridge.py | head -1 || echo "NOT RUNNING")
echo "2) traffic_signals GREEN count:"
timeout 4 ros2 topic echo --once /perception/traffic_light_recognition/traffic_signals 2>/dev/null \
  | grep -c "color: 3" || echo 0
echo "3) autoware state:" $(timeout 4 ros2 topic echo --once /autoware/state 2>/dev/null | grep "^state:" || echo "?")
echo "4) operation_mode:" $(timeout 4 ros2 topic echo --once /api/operation_mode/state 2>/dev/null | grep "mode:" | head -1 || echo "?")
echo "5) control_cmd velocity:" $(timeout 4 ros2 topic echo --once /control/command/control_cmd 2>/dev/null | grep -A1 "longitudinal:" | grep "velocity:" || echo "?")
echo "6) velocity_status:" $(timeout 4 ros2 topic echo --once /vehicle/status/velocity_status 2>/dev/null | grep "longitudinal_velocity:" || echo "?")
