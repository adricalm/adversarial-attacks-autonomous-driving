#!/usr/bin/env bash
# Full recovery after AWSIM restart (run inside autoware_full_test container).
#
# Steps: wait for /clock → stop if driving → bridge → localize + route → engage.
set -eo pipefail
set +u
source /opt/ros/humble/setup.bash
source /opt/autoware/setup.bash
set -u
unset CYCLONEDDS_URI
export ROS_DOMAIN_ID=26

ROUTE_JSON="${1:-/home/aw/autoware_data/route_candidates_straight.json}"

autoware_state() {
  timeout 4 ros2 topic echo --once /autoware/state 2>/dev/null \
    | grep "^state:" | awk '{print $2}' || echo 0
}

operation_mode() {
  timeout 4 ros2 topic echo --once /api/operation_mode/state 2>/dev/null \
    | grep "^mode:" | awk '{print $2}' || echo 0
}

echo "============================================================"
echo " Autoware full recovery after AWSIM restart"
echo " Route: ${ROUTE_JSON}"
echo " $(date -Iseconds)"
echo "============================================================"

# ── 1. Wait for /clock ───────────────────────────────────────────
echo ""
echo "── Step 1: waiting for /clock publisher (up to 60 s) ──"
for i in $(seq 1 30); do
  COUNT=$(ros2 topic info /clock 2>/dev/null | grep "Publisher count:" | awk '{print $NF}' || echo 0)
  if [[ "${COUNT}" -ge 1 ]]; then
    echo "  /clock: Publisher count=${COUNT} — OK"
    break
  fi
  echo "  attempt ${i}/30: Publisher count=${COUNT}, waiting 2 s..."
  sleep 2
  if [[ "${i}" -eq 30 ]]; then
    echo "ERROR: /clock still has 0 publishers after 60 s."
    exit 1
  fi
done

# ── 2. Stop if still engaged from a previous session ─────────────
echo ""
echo "── Step 2: stop if already driving (state 5 or mode 2) ──"
STATE=$(autoware_state)
MODE=$(operation_mode)
echo "  current state=${STATE} mode=${MODE}"
if [[ "${STATE}" -eq 5 || "${MODE}" -eq 2 ]]; then
  timeout 15 ros2 service call /api/operation_mode/change_to_stop \
    autoware_adapi_v1_msgs/srv/ChangeOperationMode "{}" || true
  sleep 3
  echo "  after stop: state=$(autoware_state) mode=$(operation_mode)"
fi

# ── 3. Bridge ────────────────────────────────────────────────────
echo ""
echo "── Step 3: restart traffic-light green bridge ──"
pkill -f traffic_light_green_bridge.py 2>/dev/null || true
sleep 1
LOG="/home/aw/autoware_data/traffic_light_green_bridge.log"
nohup python3 /home/aw/autoware_data/traffic_light_green_bridge.py \
  /home/aw/maps/nishishinjuku_autoware_map/lanelet2_map.osm \
  >> "${LOG}" 2>&1 &
echo "  Bridge PID=$!, log: ${LOG}"
sleep 3
tail -1 "${LOG}" | sed 's/^/    /'

# ── 4. Clear route + localize + set route ───────────────────────
echo ""
echo "── Step 4: clear route, localize, set route ──"
timeout 10 ros2 service call /api/routing/clear_route \
  autoware_adapi_v1_msgs/srv/ClearRoute "{}" 2>/dev/null || true
sleep 1
python3 /home/aw/autoware_data/apply_route_from_osm.py "${ROUTE_JSON}" || {
  echo "  WARNING: apply_route exited non-zero — continuing"
}

# ── 5. Wait for state 4 (or already 5 after bad prior session) ───
echo ""
echo "── Step 5: wait for state 4 (WAITING_FOR_ENGAGE, up to 60 s) ──"
READY=0
for i in $(seq 1 30); do
  STATE=$(autoware_state)
  echo "  attempt ${i}/30: autoware/state = ${STATE}"
  if [[ "${STATE}" -eq 4 ]]; then
    echo "  State 4 — ready to engage."
    READY=1
    break
  fi
  if [[ "${STATE}" -eq 5 ]]; then
    echo "  Still state 5 — forcing stop again..."
    timeout 15 ros2 service call /api/operation_mode/change_to_stop \
      autoware_adapi_v1_msgs/srv/ChangeOperationMode "{}" || true
    sleep 3
  fi
  sleep 2
done
if [[ "${READY}" -eq 0 ]]; then
  echo "WARNING: state 4 not reached. Will try engage anyway."
fi

# ── 6. Engage ────────────────────────────────────────────────────
echo ""
echo "── Step 6: change_to_autonomous ──"
timeout 15 ros2 service call /api/operation_mode/change_to_autonomous \
  autoware_adapi_v1_msgs/srv/ChangeOperationMode "{}" || true
sleep 2

echo ""
echo "── Step 7: motion snapshot ──"
bash /home/aw/autoware_data/quick_motion_check.sh

echo ""
echo "── Step 8: stuck diagnosis (if velocity ~0) ──"
bash /home/aw/autoware_data/diagnose_stuck.sh

echo ""
echo "============================================================"
echo " Done. Want: state=5, velocity>0, MRM state=1, hazard=false"
echo "============================================================"
