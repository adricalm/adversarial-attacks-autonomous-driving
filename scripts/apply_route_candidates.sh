#!/usr/bin/env bash
# Host script: compute route from OSM map, then apply inside Autoware Docker container.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JSON_OUT="${HOME}/summer26/data/autoware_data/route_candidates.json"

detect_autoware_container() {
  local pid cid
  pid="$(pgrep -f "e2e_simulator.launch" | head -1 || true)"
  if [[ -n "${pid}" && -r "/proc/${pid}/cgroup" ]]; then
    cid="$(grep -oE 'docker-[0-9a-f]{64}' "/proc/${pid}/cgroup" | head -1 | sed 's/^docker-//')"
    if [[ -n "${cid}" ]]; then
      echo "${cid:0:12}"
      return 0
    fi
  fi
  echo "autoware_full_test"
}

CONTAINER="${AUTOWARE_CONTAINER:-$(detect_autoware_container)}"

echo "==> Computing route candidates from lanelet2 map (host)"
python3 "${SCRIPT_DIR}/find_route_candidates.py" --json-out "${JSON_OUT}"

echo "==> Copying apply script into mounted autoware_data volume"
cp "${SCRIPT_DIR}/apply_route_from_osm.py" "${HOME}/summer26/data/autoware_data/apply_route_from_osm.py"

echo "==> Applying localization + routing inside Docker container: ${CONTAINER}"
sudo docker exec "${CONTAINER}" bash -lc '
  source /opt/ros/humble/setup.bash
  source /opt/autoware/setup.bash
  unset CYCLONEDDS_URI
  export ROS_DOMAIN_ID=26
  python3 /home/aw/autoware_data/apply_route_from_osm.py /home/aw/autoware_data/route_candidates.json
'

echo "==> Checking Autoware state"
sudo docker exec "${CONTAINER}" bash -lc '
  source /opt/ros/humble/setup.bash
  source /opt/autoware/setup.bash
  unset CYCLONEDDS_URI
  export ROS_DOMAIN_ID=26
  ros2 topic echo /api/routing/state --once
  ros2 topic echo /autoware/state --once
'
