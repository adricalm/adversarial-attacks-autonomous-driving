#!/usr/bin/env bash
# Spawn the two stationary NPC cars at dsgn_offline (+5) map poses.
# Flip SHOW_CARS below. Run inside Docker (AWSIM + Autoware up):
#   bash /home/aw/scripts/spawn_test_npc_car.sh
#
# AWSIM RVIZNPCSpawner destroys NPCs after ~30 s (Unity "Despawn time").
# With SHOW_CARS=true this script keeps re-spawning until Ctrl+C.
SHOW_CARS=true   # false = do nothing
RESPAWN_SEC=26   # must be < AWSIM despawnTime (default 30)

set -euo pipefail
set +u
source /opt/ros/humble/setup.bash
source /opt/autoware/setup.bash
set -u
unset CYCLONEDDS_URI
export ROS_DOMAIN_ID=26

if [[ "${SHOW_CARS}" != "true" && "${SHOW_CARS}" != "1" ]]; then
  echo "SHOW_CARS=${SHOW_CARS} → skip spawn."
  exit 0
fi

TOPIC="/simulation/dummy_perception_publisher/object_info"
MSG_TYPE="$(ros2 topic type "${TOPIC}" 2>/dev/null || true)"
if [[ -z "${MSG_TYPE}" ]]; then
  if ros2 interface show tier4_simulation_msgs/msg/DummyObject &>/dev/null; then
    MSG_TYPE="tier4_simulation_msgs/msg/DummyObject"
  else
    MSG_TYPE="autoware_simulation_msgs/msg/SimulatedObject"
  fi
fi

spawn() {
  local x="$1" y="$2" uid="$3"
  ros2 topic pub --once "${TOPIC}" "${MSG_TYPE}" "{
    header: {frame_id: map},
    id: {uuid: [${uid}, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]},
    initial_state: {pose_covariance: {pose: {
      position: {x: ${x}, y: ${y}, z: 0.0},
      orientation: {x: 0.0, y: 0.0, z: 0.33, w: 0.94}}}},
    classification: {label: 1, probability: 1.0},
    shape: {type: 0, dimensions: {x: 4.0, y: 1.8, z: 1.5}},
    max_velocity: 0.0, min_velocity: 0.0, action: 0}" >/dev/null
}

echo "SHOW_CARS=true — respawning A+B every ${RESPAWN_SEC}s (Ctrl+C to stop)."
while true; do
  spawn 81446.2 49967.1 1   # A (+5) # 81446.2 49967.1 1
  spawn 81451.5 49976.2 2   # B
  echo "  spawned $(date -Iseconds) — next in ${RESPAWN_SEC}s"
  sleep "${RESPAWN_SEC}"
done
