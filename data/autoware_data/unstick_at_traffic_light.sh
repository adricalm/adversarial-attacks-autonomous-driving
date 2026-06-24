#!/usr/bin/env bash
# Reset operation mode + route to clear a latched traffic-light stop (inside Docker).
#
# Prerequisite: restart AWSIM first if localization may be desynced, then run the
# traffic-light bridge before applying route.
set -eo pipefail
set +u
source /opt/ros/humble/setup.bash
source /opt/autoware/setup.bash
set -u
unset CYCLONEDDS_URI
export ROS_DOMAIN_ID=26

ROUTE_JSON="${1:-/home/aw/autoware_data/route_candidates.json}"

echo "==> Ensure traffic-light bridge is running"
if ! pgrep -f traffic_light_green_bridge.py >/dev/null 2>&1; then
  bash /home/aw/autoware_data/run_traffic_light_bridge_inside_container.sh
fi

echo "==> Stop (not change_to_manual — that service does not exist)"
timeout 15 ros2 service call /api/operation_mode/change_to_stop \
  autoware_adapi_v1_msgs/srv/ChangeOperationMode "{}" || true
sleep 2

echo "==> Clear route"
timeout 15 ros2 service call /api/routing/clear_route \
  autoware_adapi_v1_msgs/srv/ClearRoute "{}" || true
sleep 1

echo "==> Re-apply route from ${ROUTE_JSON}"
python3 /home/aw/autoware_data/apply_route_from_osm.py "${ROUTE_JSON}"
sleep 2

echo "==> Engage autonomous"
timeout 15 ros2 service call /api/operation_mode/change_to_autonomous \
  autoware_adapi_v1_msgs/srv/ChangeOperationMode "{}" || true
sleep 1

echo "==> Accept start (no-op if unavailable)"
timeout 5 ros2 service call /api/motion/accept_start \
  autoware_adapi_v1_msgs/srv/AcceptStart "{}" 2>/dev/null || true

echo "==> Quick check"
bash /home/aw/autoware_data/quick_motion_check.sh
