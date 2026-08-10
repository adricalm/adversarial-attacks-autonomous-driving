#!/usr/bin/env bash
# Identify exactly which diagnostic is driving hazard_status.emergency=true / MRM emergency_stop.
# Run inside autoware_full_test while the car is stuck.
set +u
source /opt/ros/humble/setup.bash
source /opt/autoware/setup.bash
set -u
unset CYCLONEDDS_URI
export ROS_DOMAIN_ID=26
# Without this the node falls back to the default RMW and sees an empty graph.
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

echo "============================================================"
echo " Emergency / MRM root-cause inspection  $(date -Iseconds)"
echo "============================================================"

echo ""
echo "── /api/fail_safe/mrm_state ──"
timeout 5 ros2 topic echo --once /api/fail_safe/mrm_state 2>/dev/null \
  | grep -E "state:|behavior:" || echo "  (none)"

echo ""
echo "── /system/operation_mode/availability (transient_local) ──"
timeout 6 ros2 topic echo --once /system/operation_mode/availability \
  --qos-durability transient_local --qos-reliability reliable 2>/dev/null \
  | grep -E "autonomous:|stop:|local:|remote:|emergency_stop:|comfortable_stop:|pull_over:" \
  || echo "  (not published)"

echo ""
echo "── /system/emergency/hazard_status : faulting diagnostics by level ──"
HZ=$(timeout 6 ros2 topic echo --once /system/emergency/hazard_status 2>/dev/null || true)
echo "  level/emergency:"
echo "${HZ}" | grep -E "^  level:|^  emergency:|emergency_holding:" | head -4 | sed 's/^/    /'
echo "  single_point_fault diagnostics (THE BLOCKER):"
echo "${HZ}" | awk '/diag_single_point_fault:/{f=1} /diag_latent_fault:/{f=0} f&&/name:/{print "    "$0} f&&/message:/{print "    "$0}'
echo "  latent_fault diagnostics:"
echo "${HZ}" | awk '/diag_latent_fault:/{f=1} /diag_no_fault:/{f=0} f&&/name:/{print "    "$0} f&&/message:/{print "    "$0}'

echo ""
echo "── diagnostic_graph: leaves currently NON-OK (steady state) ──"
timeout 5 ros2 run autoware_diagnostic_graph_utils dump_node > /tmp/g.txt 2>/dev/null || true
echo "  modes/autonomous + its children's current levels:"
grep -E "modes/autonomous|/autoware/localization |/autoware/planning |/autoware/control |/adapi/mrm_request" /tmp/g.txt \
  | sed 's/  */ /g' | sort -u | sed 's/^/    /'
echo "  all non-OK leaves:"
grep -E "STALE|WARN|ERROR" /tmp/g.txt | sed 's/  */ /g' | sort -u | sed 's/^/    /'
echo "============================================================"
