#!/usr/bin/env bash
# Host wrapper: record experiment bag as your user (no root-owned files).
#
# Pipes the record script from the host (avoids /home/aw permission issues when
# using docker exec -u). Bags land in ~/summer26/data/bags/<run_id>/.
#
# Usage (host):
#   bash ~/summer26/scripts/record_experiment_bag_host.sh run_a_baseline_001
#   bash ~/summer26/scripts/record_experiment_bag_host.sh   # auto timestamp name
#
# Requires container mounts:
#   -v "$HOME/summer26/data/bags:/home/aw/bags"
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNER_SCRIPT="${SCRIPT_DIR}/record_experiment_bag.sh"
RUN_ID="${1:-run_$(date +%Y%m%d_%H%M%S)}"
CONTAINER="${AUTOWARE_CONTAINER:-autoware_full_test}"
HOST_BAG_DIR="${HOME}/summer26/data/bags/${RUN_ID}"

mkdir -p "${HOME}/summer26/data/bags"

echo "Container: ${CONTAINER}"
echo "Run id   : ${RUN_ID}"
echo "Host path: ${HOST_BAG_DIR}"
echo "Press Ctrl+C when the run is finished."

# Run as host uid so bag files are owned by adria on the mounted volume.
sudo docker exec -u "$(id -u):$(id -g)" -it "${CONTAINER}" \
  bash -s "${RUN_ID}" < "${INNER_SCRIPT}"
