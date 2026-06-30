#!/usr/bin/env bash
# Write metadata.yaml next to a recorded bag (host or inside Docker).
#
# Usage:
#   bash ~/summer26/scripts/write_bag_metadata.sh run_a_baseline_001 baseline
#   bash ~/summer26/scripts/write_bag_metadata.sh run_b_dsgn_on_001 dsgn_on
set -euo pipefail

RUN_ID="${1:?run id required, e.g. run_a_baseline_001}"
CONDITION="${2:?condition required: baseline | dsgn_on | attack}"

if [[ -d /home/aw/bags ]]; then
  BAG_DIR="/home/aw/bags/${RUN_ID}"
else
  BAG_DIR="${HOME}/summer26/data/bags/${RUN_ID}"
fi

if [[ ! -d "${BAG_DIR}" ]]; then
  echo "ERROR: bag directory not found: ${BAG_DIR}" >&2
  exit 1
fi

case "${CONDITION}" in
  baseline)
    DSGN="off"
    FOLDER="null"
    NOTES="Run A baseline, dsgn_offline not running"
    ;;
  dsgn_on)
    DSGN="on"
    FOLDER="/home/aw/ros2_ws/src/dsgn_offline/resource/awsim_output_offline"
    NOTES="Run B with dsgn_offline publishing fake detections"
    ;;
  attack)
    DSGN="on"
    FOLDER="/home/aw/ros2_ws/src/dsgn_offline/resource/awsim_output_attack_ghost"
    NOTES="Run with adversarial detection folder"
    ;;
  *)
    echo "ERROR: unknown condition: ${CONDITION}" >&2
    exit 1
    ;;
esac

cat > "${BAG_DIR}/metadata.yaml" <<EOF
condition: ${CONDITION}
dsgn_offline: ${DSGN}
route_json: /home/aw/autoware_data/route_dsgn_ab.json
detection_folder: ${FOLDER}
path_file: /home/aw/ros2_ws/src/dsgn_offline/resource/path.txt
date: $(date -Iseconds)
notes: "${NOTES}"
EOF

echo "Wrote ${BAG_DIR}/metadata.yaml"
