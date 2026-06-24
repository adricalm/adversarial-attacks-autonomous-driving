#!/usr/bin/env bash
# Post-restart health check (run inside autoware_full_test via docker exec).
# Confirms a HEALTHY launch aggregator + real operation-mode availability before engage.
set +u
source /opt/ros/humble/setup.bash
source /opt/autoware/setup.bash
set -u
unset CYCLONEDDS_URI
export ROS_DOMAIN_ID=26

echo "============================================================"
echo " Stack readiness check  $(date -Iseconds)"
echo "============================================================"

echo ""
echo "── Resetting ros2 CLI daemon (avoids stale ~24-topic view) ──"
ros2 daemon stop >/dev/null 2>&1 || true
sleep 2
ros2 daemon start >/dev/null 2>&1 || true
sleep 3

echo ""
echo "── aggregator_node process (must be the LAUNCH's own, healthy) ──"
pgrep -af aggregator_node || echo "  WARNING: no aggregator_node running yet"

echo ""
echo "── /clock publisher count (AWSIM, expect >=1) ──"
ros2 topic info /clock 2>/dev/null | grep -i "Publisher count" || echo "  /clock not visible yet"

echo ""
echo "── /autoware/state (1=init 2=wait_route 3=plan 4=wait_engage 5=drive) ──"
timeout 6 ros2 topic echo --once /autoware/state 2>/dev/null | grep "^state:" || echo "  (no state yet)"

echo ""
echo "── /system/operation_mode/availability (transient_local) ──"
timeout 8 ros2 topic echo --once /system/operation_mode/availability \
  --qos-durability transient_local --qos-reliability reliable 2>/dev/null \
  | grep -E "autonomous:|stop:|local:|remote:|emergency_stop:|comfortable_stop:|pull_over:" \
  || echo "  (availability not published — aggregator/converter may still be starting)"

echo ""
echo "── /api/operation_mode/state (is_autonomous_mode_available) ──"
timeout 6 ros2 topic echo --once /api/operation_mode/state 2>/dev/null \
  | grep -E "mode:|is_autonomous_mode_available:" || echo "  (no operation_mode/state yet)"

echo ""
echo "── Current non-OK diagnostic leaves (steady-state snapshot) ──"
timeout 6 ros2 run autoware_diagnostic_graph_utils dump_node > /tmp/graph.txt 2>/dev/null || true
echo "  graph lines: $(wc -l < /tmp/graph.txt 2>/dev/null || echo 0)"
echo "  --- modes/* current levels ---"
grep -E "modes/(autonomous|comfortable_stop|pull_over|stop|local|remote|emergency_stop)" /tmp/graph.txt \
  | sed 's/  */ /g' | sort -u || true
echo "  --- non-OK leaves ---"
grep -E "STALE|WARN|ERROR" /tmp/graph.txt | sed 's/  */ /g' | sort -u || true

echo ""
echo "============================================================"
echo " Want: aggregator running, /clock>=1, availability autonomous: true"
echo "============================================================"
