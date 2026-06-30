#!/usr/bin/env bash
# Live audit: who subscribes to object-recognition topics (inside Docker).
#
# Rosbags record messages, NOT the subscriber graph. To see who receives
# /perception/object_recognition/objects you must query the live ROS graph.
#
# Usage (inside Docker, stack running):
#   bash /home/aw/scripts/audit_object_topic_subscribers.sh
#   bash /home/aw/scripts/audit_object_topic_subscribers.sh > /home/aw/logs/object_subs.txt
set -euo pipefail
set +u
source /opt/ros/humble/setup.bash
source /opt/autoware/setup.bash
set -u
unset CYCLONEDDS_URI
export ROS_DOMAIN_ID=26

print_topic_subscribers() {
  local topic="$1"
  local note="${2:-}"

  echo "================================================================"
  echo "TOPIC: ${topic}"
  [[ -n "${note}" ]] && echo "NOTE:  ${note}"
  echo "----------------------------------------------------------------"

  if ! ros2 topic list 2>/dev/null | grep -Fxq "${topic}"; then
    echo "STATUS: NOT_IN_GRAPH"
    echo
    return
  fi

  ros2 topic info -v "${topic}" 2>/dev/null | awk '
    /^Type:/ { print; next }
    /^Publisher count:/ { print; next }
    /^Subscription count:/ { subs = $3; print; next }
    /^Node name:/ {
      if (in_sub) print "  SUBSCRIBER: " $3
      next
    }
    /^Endpoint type:/ {
      in_sub = ($3 == "SUBSCRIPTION")
      if ($3 == "PUBLISHER") in_sub = 0
      next
    }
    END {
      if (subs == "0") print "  (no subscribers — orphan topic)"
    }
  '

  echo "--- publishers ---"
  ros2 topic info -v "${topic}" 2>/dev/null | awk '
    /^Node name:/ { node = $3; next }
    /^Endpoint type:/ {
      if ($3 == "PUBLISHER" && node != "") print "  PUBLISHER: " node
    }
  '
  echo
}

echo "Object recognition topic subscribers (live graph)"
echo "timestamp: $(date -Iseconds)"
echo "ROS_DOMAIN_ID: ${ROS_DOMAIN_ID}"
echo
echo "Legend:"
echo "  SUBSCRIBER nodes consume messages from this topic."
echo "  For /perception/object_recognition/objects → behavior + motion planners + AEB."
echo

print_topic_subscribers \
  "/perception/object_recognition/detection/centerpoint/validation/objects" \
  "validated detections — dsgn_offline publishes here"

print_topic_subscribers \
  "/perception/object_recognition/tracking/objects" \
  "tracked objects — output of multi_object_tracker"

print_topic_subscribers \
  "/perception/object_recognition/objects" \
  "predicted objects — MAIN input to motion/behavior planners"

print_topic_subscribers \
  "/planning/planning_factors/obstacle_stop" \
  "debug: why motion_velocity_planner triggered obstacle stop"

echo "Key downstream nodes (subscriptions + publications):"
for node in \
  "/perception/object_recognition/tracking/multi_object_tracker" \
  "/perception/object_recognition/prediction/map_based_prediction" \
  "/planning/scenario_planning/lane_driving/motion_planning/motion_velocity_planner" \
  "/planning/scenario_planning/lane_driving/behavior_planning/behavior_path_planner" \
  "/planning/scenario_planning/lane_driving/behavior_planning/behavior_velocity_planner" \
  "/control/autonomous_emergency_braking"; do
  echo "----------------------------------------------------------------"
  echo "NODE: ${node}"
  if ros2 node list 2>/dev/null | grep -Fxq "${node}"; then
    ros2 node info "${node}" 2>/dev/null | sed -n '/^  Subscribers:/,/^  Publishers:/p' | head -40
  else
    echo "  NOT_IN_GRAPH"
  fi
  echo
done

echo "Tip: full stack audit → bash /home/aw/scripts/audit_stack_usage.sh with_dsgn"
