#!/usr/bin/env bash
# Spawn stationary NPC car(s) for AWSIM (DummyObject / RVIZNPCSpawner).
#
# AWSIM destroys NPCs after ~30 s (Unity "Despawn time"); this script
# re-spawns until Ctrl+C.
#
# Usage (inside Docker, AWSIM + Autoware up):
#   bash /home/aw/scripts/helpers/spawn_test_npc_car.sh
#   bash /home/aw/scripts/helpers/spawn_test_npc_car.sh --xy 81404.99,49954.37
#   bash /home/aw/scripts/helpers/spawn_test_npc_car.sh \
#     --xy 81404.99,49954.37 \
#     --quat -0.0065,-0.0045,0.7381,0.6747
#   bash /home/aw/scripts/helpers/spawn_test_npc_car.sh --xy A,B --xy C,D
#
# Flags:
#   --xy X,Y          map position (repeatable). If omitted, uses legacy A+B.
#   --quat X,Y,Z,W    orientation for the next --xy (default: legacy z=0.33,w=0.94)
#   --respawn SEC     respawn period (default 27; must be < AWSIM despawnTime)
#   --show false      skip spawn and exit
SHOW_CARS=true
RESPAWN_SEC=27

set -euo pipefail

XY_LIST=()
QUAT_LIST=()   # parallel to XY_LIST; empty string → default quat
PENDING_QUAT=""  # --quat before --xy

while [[ $# -gt 0 ]]; do
  case "$1" in
    --xy)
      XY_LIST+=("$2")
      QUAT_LIST+=("${PENDING_QUAT}")
      PENDING_QUAT=""
      shift 2
      ;;
    --quat)
      # Applies to previous --xy if any, else to the next --xy.
      if [[ ${#XY_LIST[@]} -gt 0 && -z "${QUAT_LIST[-1]:-}" ]]; then
        QUAT_LIST[-1]="$2"
      else
        PENDING_QUAT="$2"
      fi
      shift 2
      ;;
    --respawn)
      RESPAWN_SEC="$2"
      shift 2
      ;;
    --show)
      SHOW_CARS="$2"
      shift 2
      ;;
    -h|--help)
      sed -n '2,20p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown arg: $1 (try --help)" >&2
      exit 1
      ;;
  esac
done

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

# Default orientation (legacy hardcoded cars)
DEFAULT_QX=0.0
DEFAULT_QY=0.0
DEFAULT_QZ=0.33
DEFAULT_QW=0.94

spawn() {
  local x="$1" y="$2" uid="$3" qx="$4" qy="$5" qz="$6" qw="$7"
  ros2 topic pub --once "${TOPIC}" "${MSG_TYPE}" "{
    header: {frame_id: map},
    id: {uuid: [${uid}, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]},
    initial_state: {pose_covariance: {pose: {
      position: {x: ${x}, y: ${y}, z: 0.0},
      orientation: {x: ${qx}, y: ${qy}, z: ${qz}, w: ${qw}}}}},
    classification: {label: 1, probability: 1.0},
    shape: {type: 0, dimensions: {x: 4.0, y: 1.8, z: 1.5}},
    max_velocity: 0.0, min_velocity: 0.0, action: 0}" >/dev/null
}

parse_xy() {
  local s="$1"
  if [[ "${s}" != *,* ]]; then
    echo "ERROR: --xy expects X,Y got '${s}'" >&2
    exit 1
  fi
  X="${s%%,*}"
  Y="${s#*,}"
}

parse_quat() {
  local s="$1"
  IFS=',' read -r QX QY QZ QW <<< "${s}"
  if [[ -z "${QW:-}" ]]; then
    echo "ERROR: --quat expects X,Y,Z,W got '${s}'" >&2
    exit 1
  fi
}

# Build spawn list: either CLI --xy... or legacy A+B
SPAWN_X=() SPAWN_Y=() SPAWN_QX=() SPAWN_QY=() SPAWN_QZ=() SPAWN_QW=()

if [[ ${#XY_LIST[@]} -eq 0 ]]; then
  SPAWN_X=(81441.3 81451.1)
  SPAWN_Y=(49963.7 49975)
  SPAWN_QX=("${DEFAULT_QX}" "${DEFAULT_QX}")
  SPAWN_QY=("${DEFAULT_QY}" "${DEFAULT_QY}")
  SPAWN_QZ=("${DEFAULT_QZ}" "${DEFAULT_QZ}")
  SPAWN_QW=("${DEFAULT_QW}" "${DEFAULT_QW}")
  echo "No --xy given — legacy A+B poses."
else
  for i in "${!XY_LIST[@]}"; do
    parse_xy "${XY_LIST[$i]}"
    SPAWN_X+=("${X}")
    SPAWN_Y+=("${Y}")
    if [[ -n "${QUAT_LIST[$i]:-}" ]]; then
      parse_quat "${QUAT_LIST[$i]}"
      SPAWN_QX+=("${QX}")
      SPAWN_QY+=("${QY}")
      SPAWN_QZ+=("${QZ}")
      SPAWN_QW+=("${QW}")
    else
      SPAWN_QX+=("${DEFAULT_QX}")
      SPAWN_QY+=("${DEFAULT_QY}")
      SPAWN_QZ+=("${DEFAULT_QZ}")
      SPAWN_QW+=("${DEFAULT_QW}")
    fi
  done
fi

N=${#SPAWN_X[@]}
echo "Respawning ${N} NPC(s) every ${RESPAWN_SEC}s (Ctrl+C to stop)."
for i in "${!SPAWN_X[@]}"; do
  echo "  [$i] x=${SPAWN_X[$i]} y=${SPAWN_Y[$i]} quat=${SPAWN_QX[$i]},${SPAWN_QY[$i]},${SPAWN_QZ[$i]},${SPAWN_QW[$i]}"
done

while true; do
  for i in "${!SPAWN_X[@]}"; do
    uid=$((i + 1))
    spawn "${SPAWN_X[$i]}" "${SPAWN_Y[$i]}" "${uid}" \
      "${SPAWN_QX[$i]}" "${SPAWN_QY[$i]}" "${SPAWN_QZ[$i]}" "${SPAWN_QW[$i]}"
  done
  echo "  spawned $(date -Iseconds) — next in ${RESPAWN_SEC}s"
  sleep "${RESPAWN_SEC}"
done
