#!/usr/bin/env bash
# Live audit: for each critical topic, report publishers, subscribers, and brief hz.
# Run inside autoware_full_test after AWSIM + stack are up.
#
# Usage:
#   bash /home/aw/scripts/audit_stack_usage.sh [label]
#
# Output: ~/summer26/logs/stack_audit_<label>_<timestamp>.txt (needs logs volume mount)
set -euo pipefail
set +u
source /opt/ros/humble/setup.bash
source /opt/autoware/setup.bash
set -u
unset CYCLONEDDS_URI
export ROS_DOMAIN_ID=26

LABEL="${1:-live}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-/home/aw/logs}"
mkdir -p "${OUT_DIR}"
REPORT="${OUT_DIR}/stack_audit_${LABEL}_${STAMP}.txt"

HZ_SEC="${HZ_SEC:-2}"

audit_topic() {
  local topic="$1"
  local note="${2:-}"
  {
    echo "================================================================"
    echo "TOPIC: ${topic}"
    [[ -n "${note}" ]] && echo "NOTE: ${note}"
    echo "----------------------------------------------------------------"
    if ! ros2 topic list 2>/dev/null | grep -Fxq "${topic}"; then
      echo "STATUS: NOT_IN_GRAPH"
      echo
      return
    fi
    ros2 topic info -v "${topic}" 2>/dev/null | grep -E "^(Type:|Publisher count:|Subscription count:|Node name:|Endpoint type:)" || true
    echo "--- hz (${HZ_SEC}s) ---"
  } >> "${REPORT}"
  timeout "${HZ_SEC}" ros2 topic hz "${topic}" >> "${REPORT}" 2>&1 || echo "(no messages in ${HZ_SEC}s)" >> "${REPORT}"
  echo >> "${REPORT}"
}

audit_node() {
  local node="$1"
  {
    echo "================================================================"
    echo "NODE: ${node}"
    echo "----------------------------------------------------------------"
  } >> "${REPORT}"
  if ! ros2 node list 2>/dev/null | grep -Fxq "${node}"; then
    echo "STATUS: NOT_IN_GRAPH" >> "${REPORT}"
    echo >> "${REPORT}"
    return
  fi
  ros2 node info "${node}" 2>/dev/null | sed -n '1,80p' >> "${REPORT}" || true
  echo >> "${REPORT}"
}

{
  echo "Stack usage audit"
  echo "timestamp: ${STAMP}"
  echo "label: ${LABEL}"
  echo "ROS_DOMAIN_ID: ${ROS_DOMAIN_ID}"
  echo
  echo "Legend:"
  echo "  NOT_IN_GRAPH     topic/node absent"
  echo "  Publisher count 0 / hz timeout  likely not fed"
  echo "  Subscription count 0            orphan output (not wired downstream)"
  echo
  echo "Motion/route context (fill manually when capturing):"
  echo "  autoware_state: $(timeout 2 ros2 topic echo --once /autoware/state 2>/dev/null | grep 'state:' | head -1 || echo 'unknown')"
  echo "  route_state:    $(timeout 2 ros2 topic echo --once /api/routing/state 2>/dev/null | grep -E 'state:|uuid:' | head -2 || echo 'unknown')"
  echo "  ego_speed:      $(timeout 2 ros2 topic echo --once /vehicle/status/velocity_status 2>/dev/null | grep longitudinal_velocity || echo 'unknown')"
  echo
} > "${REPORT}"

# --- Simulation & time ---
audit_topic "/clock" "sim time — required for entire stack"
audit_topic "/vehicle/status/velocity_status" "ego motion feedback"

# --- Sensing ---
audit_topic "/sensing/lidar/top/pointcloud_raw" "AWSIM raw top LiDAR"
audit_topic "/sensing/lidar/left/pointcloud_raw" "AWSIM raw left LiDAR"
audit_topic "/sensing/lidar/right/pointcloud_raw" "AWSIM raw right LiDAR"
audit_topic "/sensing/lidar/concatenated/pointcloud" "fused LiDAR — main perception input"
audit_topic "/sensing/imu/imu_data" "IMU"
audit_topic "/sensing/camera/traffic_light/image_raw" "traffic-light camera"
audit_topic "/sensing/gnss/pose_with_covariance" "GNSS (may be unused in AWSIM path)"

# --- Map ---
audit_topic "/map/vector_map" "lanelet2 — latched"
audit_topic "/map/pointcloud_map" "NDT map — latched"

# --- Localization ---
audit_topic "/localization/pose_with_covariance" "EKF pose — dsgn sync"
audit_topic "/localization/kinematic_state" "planners + control"

# --- Object recognition chain ---
audit_topic "/perception/object_recognition/detection/centerpoint/objects" "CenterPoint raw"
audit_topic "/perception/object_recognition/detection/centerpoint/validation/objects" "validated detections + dsgn"
audit_topic "/perception/object_recognition/detection/clustering/objects" "clustering — check subs=0"
audit_topic "/perception/object_recognition/detection/detection_by_tracker/objects" "tracker-assisted detections"
audit_topic "/perception/object_recognition/detection/camera_only/objects" "camera detector input to tracker"
audit_topic "/perception/object_recognition/detection/objects" "legacy merged — check subs=0"
audit_topic "/perception/object_recognition/tracking/objects" "tracked objects"
audit_topic "/perception/object_recognition/objects" "predicted objects — main planner input"
audit_topic "/perception/object_recognition/prediction/maneuver" "maneuver prediction side channel"

# --- Obstacle / grid ---
audit_topic "/perception/obstacle_segmentation/pointcloud" "obstacle points — AEB path"
audit_topic "/perception/occupancy_grid_map/map" "occupancy grid"

# --- Traffic lights ---
audit_topic "/perception/traffic_light_recognition/external/traffic_signals" "green bridge injects here"
audit_topic "/perception/traffic_light_recognition/traffic_signals" "final TL output"

# --- Planning outputs ---
audit_topic "/planning/mission_planning/route" "route — needs mission planning"
audit_topic "/planning/scenario_planning/scenario" "lane_driving vs parking"
audit_topic "/planning/scenario_planning/trajectory" "final planned trajectory"
audit_topic "/planning/planning_factors/obstacle_stop" "why stop for objects"

# --- Control ---
audit_topic "/control/command/control_cmd" "commands to vehicle"
audit_topic "/control/trajectory_follower/control_cmd" "controller output"

# --- Key nodes ---
audit_node "/dsgn_offline"
audit_node "/perception/object_recognition/detection/centerpoint/lidar_centerpoint"
audit_node "/perception/object_recognition/detection/obstacle_pointcloud_based_validator_node"
audit_node "/perception/object_recognition/tracking/multi_object_tracker"
audit_node "/perception/object_recognition/prediction/map_based_prediction"
audit_node "/planning/scenario_planning/lane_driving/motion_planning/motion_velocity_planner"
audit_node "/control/autonomous_emergency_braking"

echo "Wrote ${REPORT}"
echo "On host: ~/summer26/logs/$(basename "${REPORT}")"
