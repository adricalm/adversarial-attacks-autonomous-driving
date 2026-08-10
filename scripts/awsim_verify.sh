#!/usr/bin/env bash
# Verify a running AWSIM container is publishing a healthy sensor set, and print the
# live camera geometry so it can be compared against the KITTI calib files.
#
# Usage: scripts/awsim_verify.sh [container_name]
set -uo pipefail

NAME="${1:-awsim_pristine}"
DOMAIN="${ROS_DOMAIN_ID:-26}"

if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "ERROR: container '$NAME' is not running" >&2
  exit 1
fi

# ROS may be sourced freely *here*: this is a separate process from the Unity plugin.
docker exec "$NAME" bash -lc "
source /opt/ros/humble/setup.bash
unset CYCLONEDDS_URI
export ROS_DOMAIN_ID=$DOMAIN

echo '===== required topics ====='
need='/clock
/sensing/camera/traffic_light/image_raw
/sensing/camera/traffic_light/camera_info
/sensing/lidar/top/pointcloud_raw
/vehicle/status/velocity_status'
have=\$(ros2 topic list 2>/dev/null)
missing=0
while read -r t; do
  if grep -qx \"\$t\" <<<\"\$have\"; then echo \"  OK      \$t\"
  else echo \"  MISSING \$t\"; missing=1; fi
done <<<\"\$need\"

echo
echo '===== rates ====='
for t in /clock /sensing/camera/traffic_light/image_raw /sensing/lidar/top/pointcloud_raw; do
  printf '  %-48s ' \"\$t\"
  timeout 8 ros2 topic hz \"\$t\" 2>/dev/null | head -1 || echo '(no data)'
done

echo
echo '===== camera geometry (compare with calib P0) ====='
timeout 20 ros2 topic echo --once /sensing/camera/traffic_light/camera_info 2>/dev/null \
  | grep -A14 -E '^(height|width|k:)' | head -24

exit \$missing
"
rc=$?

echo
if [[ $rc -eq 0 ]]; then
  echo "RESULT: healthy."
  echo "Expected geometry: 1920x1080, fx=960.0 fy=959.3908081054688 cx=960.5 cy=540.5"
  echo "Expected rates with timeScale=0.6: camera/lidar ~6 Hz (10 Hz x 0.6)."
else
  echo "RESULT: topics missing. Most likely cause: ROS 2 was sourced for the AWSIM"
  echo "process. Check the player log for 'should not source' / 'topic name is invalid'."
fi
exit $rc
