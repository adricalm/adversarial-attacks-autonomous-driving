#!/usr/bin/env bash
# Drive route: clear → init pose (AWSIM teleport + localise) → goal → auto engage.
# Reads start/goal from route JSON (default: route_dsgn_ab.json).
#
# Usage (inside Docker):
#   bash /home/aw/scripts/drive_route_and_engage.sh
#   bash /home/aw/scripts/drive_route_and_engage.sh /path/to/other_route.json
#   bash /home/aw/scripts/drive_route_and_engage.sh --no-engage
set -eo pipefail
set +u
source /opt/ros/humble/setup.bash
source /opt/autoware/setup.bash
set -u
unset CYCLONEDDS_URI
export ROS_DOMAIN_ID=26

ROUTE_JSON="/home/aw/autoware_data/route_dsgn_ab.json"
NO_ENGAGE=0
for arg in "$@"; do
  case "${arg}" in
    --no-engage) NO_ENGAGE=1 ;;
    --*) ;;
    *) ROUTE_JSON="${arg}" ;;
  esac
done

if [[ ! -f "${ROUTE_JSON}" ]]; then
  echo "ERROR: route file not found: ${ROUTE_JSON}" >&2; exit 1
fi

# Parse positions from JSON
read_json() { python3 -c "import json,sys; d=json.load(open('${ROUTE_JSON}')); print(d$1)"; }

SX=$(read_json "['start']['position']['x']")
SY=$(read_json "['start']['position']['y']")
SZ=$(read_json "['start']['position']['z']")
SQX=$(read_json "['start']['orientation']['x']")
SQY=$(read_json "['start']['orientation']['y']")
SQZ=$(read_json "['start']['orientation']['z']")
SQW=$(read_json "['start']['orientation']['w']")

GX=$(read_json "['goal']['position']['x']")
GY=$(read_json "['goal']['position']['y']")
GZ=$(read_json "['goal']['position']['z']")
GQX=$(read_json "['goal']['orientation']['x']")
GQY=$(read_json "['goal']['orientation']['y']")
GQZ=$(read_json "['goal']['orientation']['z']")
GQW=$(read_json "['goal']['orientation']['w']")

echo "========================================================"
echo " drive_route_and_engage"
echo " Route : ${ROUTE_JSON}"
echo " Start : (${SX}, ${SY}, ${SZ})"
echo " Goal  : (${GX}, ${GY}, ${GZ})"
echo " $(date -Iseconds)"
echo "========================================================"

# ── 1. Clear any existing route ──────────────────────────────────
echo ""
echo "── 1. Clear route ──"
ros2 service call /api/routing/clear_route \
  autoware_adapi_v1_msgs/srv/ClearRoute "{}" 2>/dev/null || true
sleep 1

# ── 2. Initial pose → teleports AWSIM car + initialises localisation ─
echo ""
echo "── 2. Set initial pose (teleports AWSIM + initialise localisation) ──"
ros2 topic pub --once /initialpose \
  geometry_msgs/msg/PoseWithCovarianceStamped \
  "{header: {frame_id: map}, pose: {pose: {
      position:    {x: ${SX}, y: ${SY}, z: ${SZ}},
      orientation: {x: ${SQX}, y: ${SQY}, z: ${SQZ}, w: ${SQW}}}}}"
echo "  Waiting 10 s for localisation to settle..."
sleep 10

# ── 3. Set goal pose ─────────────────────────────────────────────
echo ""
echo "── 3. Set goal pose ──"
ros2 service call /api/routing/set_route_points \
  autoware_adapi_v1_msgs/srv/SetRoutePoints \
  "{header: {frame_id: map},
    goal: {
      position:    {x: ${GX}, y: ${GY}, z: ${GZ}},
      orientation: {x: ${GQX}, y: ${GQY}, z: ${GQZ}, w: ${GQW}}},
    waypoints: []}"
echo "  Waiting 5 s for route/planning to settle..."
sleep 5

if [[ "${NO_ENGAGE}" -eq 1 ]]; then
  echo ""
  echo "── Skipping engage (--no-engage). ──"
  echo "========================================================"
  echo " Done (route set; engage manually)."
  echo "========================================================"
  exit 0
fi

# ── 4. Engage autonomous mode ─────────────────────────────────────
echo ""
echo "── 4. Engage autonomous mode ──"
ros2 service call /api/operation_mode/change_to_autonomous \
  autoware_adapi_v1_msgs/srv/ChangeOperationMode "{}"

echo ""
echo "========================================================"
echo " Done."
echo "========================================================"
