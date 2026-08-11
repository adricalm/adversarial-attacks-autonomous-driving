#!/usr/bin/env bash
# Print current ego pose in map frame — for building route JSON by hand.
#
# Usage (inside Docker, after localization is running):
#   bash /home/aw/scripts/helpers/capture_pose.sh           # human-readable
#   bash /home/aw/scripts/helpers/capture_pose.sh --json    # JSON fragment for route file
#   bash /home/aw/scripts/helpers/capture_pose.sh --json start > /tmp/route.json
#
# Tip: park the car where you want start or goal, then run this.
set -euo pipefail
set +u
source /opt/ros/humble/setup.bash
source /opt/autoware/setup.bash
set -u
unset CYCLONEDDS_URI
export ROS_DOMAIN_ID=26
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

TMP="$(mktemp)"
trap 'rm -f "${TMP}"' EXIT

if ! timeout 5 ros2 topic echo --once /localization/pose_with_covariance > "${TMP}" 2>/dev/null; then
  echo "ERROR: no pose on /localization/pose_with_covariance (initialize localization first)." >&2
  echo "Alternative (live TF):" >&2
  echo "  ros2 run tf2_ros tf2_echo map base_link" >&2
  exit 1
fi

X=$(grep -A1 "position:" "${TMP}" | grep "x:" | head -1 | awk '{print $2}')
Y=$(grep -A2 "position:" "${TMP}" | grep "y:" | head -1 | awk '{print $2}')
Z=$(grep -A3 "position:" "${TMP}" | grep "z:" | head -1 | awk '{print $2}')
QX=$(grep -A1 "orientation:" "${TMP}" | grep "x:" | head -1 | awk '{print $2}')
QY=$(grep -A2 "orientation:" "${TMP}" | grep "y:" | head -1 | awk '{print $2}')
QZ=$(grep -A3 "orientation:" "${TMP}" | grep "z:" | head -1 | awk '{print $2}')
QW=$(grep -A4 "orientation:" "${TMP}" | grep "w:" | head -1 | awk '{print $2}')

if [[ -z "${X}" || -z "${Y}" ]]; then
  echo "ERROR: could not parse pose message." >&2
  cat "${TMP}" >&2
  exit 1
fi

if [[ "${1:-}" == "--json" ]]; then
  SLOT="${2:-pose}"
  cat <<EOF
  "${SLOT}": {
    "position": { "x": ${X}, "y": ${Y}, "z": ${Z} },
    "orientation": { "x": ${QX}, "y": ${QY}, "z": ${QZ}, "w": ${QW} }
  }
EOF
  exit 0
fi

echo "Current ego pose (map frame, from /localization/pose_with_covariance):"
echo "  x=${X}"
echo "  y=${Y}"
echo "  z=${Z}"
echo "  orientation: x=${QX} y=${QY} z=${QZ} w=${QW}"
echo ""
echo "JSON fragment:"
bash "$0" --json pose
