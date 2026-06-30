#!/usr/bin/env bash
# Snapshot ros2 topic/node lists — inside autoware_full_test container.
# Writes to /home/aw/logs/ (host: ~/summer26/logs/) when logs volume is mounted.
set -euo pipefail
set +u
source /opt/ros/humble/setup.bash
source /opt/autoware/setup.bash
set -u
unset CYCLONEDDS_URI
export ROS_DOMAIN_ID=26

LABEL="${1:-snapshot}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-/home/aw/logs}"

mkdir -p "${OUT_DIR}"

TOPICS="${OUT_DIR}/topics_${LABEL}_${STAMP}.txt"
NODES="${OUT_DIR}/nodes_${LABEL}_${STAMP}.txt"

ros2 topic list > "${TOPICS}"
ros2 node list > "${NODES}"

echo "Wrote ${TOPICS}"
echo "Wrote ${NODES}"
echo "On host: ~/summer26/logs/$(basename "${TOPICS}")"
