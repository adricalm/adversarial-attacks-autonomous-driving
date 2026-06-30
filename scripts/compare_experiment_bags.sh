#!/usr/bin/env bash
# Host wrapper: compare two experiment bags (runs analysis inside Docker).
#
# Usage (host):
#   bash ~/summer26/scripts/compare_experiment_bags.sh run_a_baseline_001 run_b_dsgn_offline_001
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <baseline_run_id> <dsgn_on_run_id>" >&2
  echo "Example: $0 run_a_baseline_001 run_b_dsgn_offline_001" >&2
  exit 1
fi

BASELINE_ID="$1"
TREATMENT_ID="$2"
CONTAINER="${AUTOWARE_CONTAINER:-autoware_full_test}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BAG_ROOT="${HOME}/summer26/data/bags"

for id in "${BASELINE_ID}" "${TREATMENT_ID}"; do
  if [[ ! -d "${BAG_ROOT}/${id}" ]]; then
    echo "ERROR: bag not found: ${BAG_ROOT}/${id}" >&2
    exit 1
  fi
done

sudo docker exec "${CONTAINER}" bash -lc "
  set +u
  source /opt/ros/humble/setup.bash
  source /opt/autoware/setup.bash
  set -u
  python3 /home/aw/scripts/compare_experiment_bags.py \
    /home/aw/bags/${BASELINE_ID} /home/aw/bags/${TREATMENT_ID}
"
