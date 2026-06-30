#!/usr/bin/env bash
# Record a lightweight experiment rosbag for dsgn A/B testing (inside Docker).
#
# Bags are written to /home/aw/bags/ (host: ~/summer26/data/bags/).
# Prefer the host wrapper (pipes this script in; files owned by your user):
#   bash ~/summer26/scripts/record_experiment_bag_host.sh run_a_baseline_001
#
# Usage (inside Docker):
#   bash /home/aw/scripts/record_experiment_bag.sh run_a_baseline_001
#   bash /home/aw/scripts/record_experiment_bag.sh   # auto timestamp name
set -eo pipefail
set +u
source /opt/ros/humble/setup.bash
source /opt/autoware/setup.bash
set -u
unset CYCLONEDDS_URI
export ROS_DOMAIN_ID=26

BAG_ROOT="/home/aw/bags"
RUN_ID="${1:-run_$(date +%Y%m%d_%H%M%S)}"
BAG_DIR="${BAG_ROOT}/${RUN_ID}"

if [[ ! -d "${BAG_ROOT}" ]]; then
  echo "ERROR: ${BAG_ROOT} missing — add Docker mount:" >&2
  echo '  -v "$HOME/summer26/data/bags:/home/aw/bags"' >&2
  exit 1
fi

echo "Recording to ${BAG_DIR}"
echo "Host path: ~/summer26/data/bags/${RUN_ID}"
echo "Press Ctrl+C when the run is finished."

exec ros2 bag record -o "${BAG_DIR}" \
  /clock \
  /autoware/state \
  /api/operation_mode/state \
  /api/routing/state \
  /localization/pose_with_covariance \
  /awsim/ground_truth/vehicle/pose \
  /vehicle/status/velocity_status \
  /vehicle/status/steering_status \
  /perception/object_recognition/detection/centerpoint/validation/objects \
  /perception/object_recognition/detection/objects \
  /perception/object_recognition/tracking/objects \
  /perception/object_recognition/objects \
  /planning/scenario_planning/trajectory \
  /planning/planning_factors/obstacle_stop \
  /planning/planning_factors/obstacle_cruise
