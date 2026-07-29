#!/usr/bin/env bash
# Recover after MRM emergency_stop latched due to localization degradation.
# Typical trigger: NDT score below threshold + EKF pose_no_update_count > 100.
#
# Run inside autoware_full_test after AWSIM is up and /clock has 1 publisher.
set -eo pipefail
set +u
source /opt/ros/humble/setup.bash
source /opt/autoware/setup.bash
set -u
unset CYCLONEDDS_URI
export ROS_DOMAIN_ID=26

ROUTE_JSON="${1:-/home/aw/autoware_data/route_candidates.json}"

echo "============================================================"
echo " Recover from MRM emergency_stop (localization reset)"
echo " $(date -Iseconds)"
echo "============================================================"

# Optional lab helper — not required for localization / MRM recovery.
BRIDGE_PY="/home/aw/scripts/traffic_light_green_bridge.py"
if pgrep -f traffic_light_green_bridge.py >/dev/null 2>&1; then
  echo "Traffic-light green bridge already running (optional)."
elif [[ -f "${BRIDGE_PY}" ]]; then
  bash /home/aw/scripts/run_traffic_light_bridge_inside_container.sh || \
    echo "WARNING: failed to start traffic-light bridge — continuing without it"
else
  echo "NOTE: no traffic-light green bridge (optional). Car may stop at red lights."
fi

echo ""
echo "── Step 1: stop operation mode (clears autonomous, allows re-init) ──"
timeout 15 ros2 service call /api/operation_mode/change_to_stop \
  autoware_adapi_v1_msgs/srv/ChangeOperationMode "{}" || true
sleep 3

echo ""
echo "── Step 2: clear route (required before re-localize while route in use) ──"
timeout 15 ros2 service call /api/routing/clear_route \
  autoware_adapi_v1_msgs/srv/ClearRoute "{}" || true
sleep 1

echo ""
echo "── Step 3: re-localize + set route (resets EKF / NDT pose pipeline) ──"
python3 /home/aw/scripts/apply_route_from_osm.py "${ROUTE_JSON}" || {
  echo "WARNING: apply_route exited non-zero — continuing"
}
sleep 5

echo ""
echo "── Step 4: wait for hazard emergency to clear (up to 30 s) ──"
for i in $(seq 1 15); do
  EMERG=$(timeout 4 ros2 topic echo --once /system/emergency/hazard_status 2>/dev/null \
    | grep "emergency:" | awk '{print $2}' || echo "true")
  echo "  attempt ${i}/15: hazard emergency=${EMERG}"
  if [[ "${EMERG}" == "false" ]]; then
    echo "  Hazard cleared."
    break
  fi
  sleep 2
done

echo ""
echo "── Step 5: wait for autoware state 4 ──"
for i in $(seq 1 30); do
  STATE=$(timeout 4 ros2 topic echo --once /autoware/state 2>/dev/null \
    | grep "^state:" | awk '{print $2}' || echo 0)
  echo "  attempt ${i}/30: state=${STATE}"
  if [[ "${STATE}" -eq 4 ]]; then
    break
  fi
  sleep 2
done

echo ""
echo "── Step 6: engage autonomous ──"
timeout 15 ros2 service call /api/operation_mode/change_to_autonomous \
  autoware_adapi_v1_msgs/srv/ChangeOperationMode "{}" || true
sleep 1
timeout 5 ros2 service call /api/motion/accept_start \
  autoware_adapi_v1_msgs/srv/AcceptStart "{}" 2>/dev/null || true

echo ""
echo "── Step 7: snapshot ──"
bash /home/aw/autoware_data/quick_motion_check.sh
echo ""
echo "── MRM state ──"
timeout 4 ros2 topic echo --once /api/fail_safe/mrm_state 2>/dev/null || true
