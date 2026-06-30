#!/usr/bin/env bash
# Quick check: is AWSIM actually talking to ROS? (inside Docker)
set -euo pipefail
set +u
source /opt/ros/humble/setup.bash
source /opt/autoware/setup.bash
set -u
unset CYCLONEDDS_URI
export ROS_DOMAIN_ID=26

echo "=== ROS environment ==="
echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID}"
echo "CYCLONEDDS_URI=${CYCLONEDDS_URI:-<unset>}"
echo "topics in graph: $(ros2 topic list 2>/dev/null | wc -l)"

check_topic() {
  local topic="$1"
  echo ""
  echo "=== ${topic} ==="
  if ! ros2 topic list 2>/dev/null | grep -Fxq "${topic}"; then
    echo "NOT in topic list"
    return 1
  fi
  ros2 topic info "${topic}" 2>/dev/null | grep -E "Type:|Publisher count" || true
  echo "sample:"
  timeout 3 ros2 topic echo --once "${topic}" 2>/dev/null | head -12 || echo "(no message in 3s)"
}

check_topic /clock || true
check_topic /awsim/ground_truth/vehicle/pose || true
check_topic /vehicle/status/velocity_status || true
check_topic /autoware/state || true

echo ""
echo "If AWSIM window is open but all samples are empty:"
echo "  → press Play in AWSIM, or restart AWSIM with ROS_DOMAIN_ID=26"
echo "  → then: ros2 daemon stop && ros2 daemon start"
