#!/usr/bin/env bash
# Verify dsgn_offline is wired into tracking → prediction → planning (inside Docker).
set -eo pipefail
set +u
source /opt/ros/humble/setup.bash
source /opt/autoware/setup.bash
set -u
unset CYCLONEDDS_URI
export ROS_DOMAIN_ID=26

DET_TOPIC="${1:-/perception/object_recognition/detection/centerpoint/validation/objects}"

echo "=== dsgn detection topic (must have multi_object_tracker subscriber) ==="
echo "topic: ${DET_TOPIC}"
ros2 topic info -v "${DET_TOPIC}" 2>/dev/null | grep -E "Node name:|Subscription count:|Publisher count:" || true

echo ""
echo "=== Raw detections (dsgn + centerpoint validator may both publish here) ==="
timeout 4 ros2 topic echo --once "${DET_TOPIC}" 2>/dev/null \
  | grep -E "frame_id:|existence_probability:|label:|position:" | head -24 || echo "(no message)"

echo ""
echo "=== Tracked objects (output of multi_object_tracker) ==="
timeout 4 ros2 topic echo --once /perception/object_recognition/tracking/objects 2>/dev/null \
  | grep -E "object_id:|existence_probability:|label:|position:" | head -24 || echo "(none)"

echo ""
echo "=== Predicted objects (input to motion/behavior planners) ==="
timeout 4 ros2 topic echo --once /perception/object_recognition/objects 2>/dev/null \
  | grep -E "object_id:|existence_probability:|label:|position:" | head -24 || echo "(none)"

echo ""
echo "=== Obstacle stop planner ==="
timeout 4 ros2 topic echo --once /planning/planning_factors/obstacle_stop 2>/dev/null \
  | grep -E "module:|distance:|behavior:|is_safe:" || echo "(no obstacle_stop factors)"

echo ""
echo "=== Ego speed ==="
timeout 4 ros2 topic echo --once /vehicle/status/velocity_status 2>/dev/null \
  | grep "longitudinal_velocity:" || true
