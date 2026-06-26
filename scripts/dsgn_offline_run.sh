#!/usr/bin/env bash
# Run dsgn_offline node — inside autoware_full_test container, after build.
#
# Optional env:
#   DETECTION_FOLDER  override precomputed .txt directory (e.g. adversarial outputs)
#   PATH_FILE         override ego trajectory file (default: resource/path.txt)
set -euo pipefail

source /opt/ros/humble/setup.bash
source /opt/autoware/setup.bash
unset CYCLONEDDS_URI
export ROS_DOMAIN_ID=26

WS=/home/aw/ros2_ws
RESOURCE="${WS}/src/dsgn_offline/resource"

if [[ ! -f "${WS}/install/setup.bash" ]]; then
  echo "ERROR: overlay not built. Run dsgn_offline_build.sh first."
  exit 1
fi
# shellcheck source=/dev/null
source "${WS}/install/setup.bash"

DETECTION_FOLDER="${DETECTION_FOLDER:-${RESOURCE}/awsim_output_offline}"
PATH_FILE="${PATH_FILE:-${RESOURCE}/path.txt}"

if [[ ! -d "${DETECTION_FOLDER}" ]]; then
  echo "ERROR: detection folder not found: ${DETECTION_FOLDER}"
  exit 1
fi
if [[ ! -f "${PATH_FILE}" ]]; then
  echo "ERROR: path file not found: ${PATH_FILE}"
  exit 1
fi

echo "detection_folder=${DETECTION_FOLDER}"
echo "path_file=${PATH_FILE}"

exec ros2 run dsgn_offline dsgn_offline \
  --ros-args \
  -p "detection_folder:=${DETECTION_FOLDER}" \
  -p "path_file:=${PATH_FILE}"
