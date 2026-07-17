#!/usr/bin/env bash
# Gates 2-3: GPU visibility + spconv import inside DSGN++ Docker image.
# Requires: image built via dsgn2_build_docker.sh
#
# Usage: bash ~/summer26/scripts/dsgn2_smoke_test.sh
set -euo pipefail

ROOT="${HOME}/summer26"
# shellcheck disable=SC1091
source "${ROOT}/scripts/dsgn2_docker_common.sh"
IMAGE="${DSGN2_DOCKER_IMAGE}"
LOG_DIR="${ROOT}/logs"
LOG_FILE="${LOG_DIR}/dsgn2_smoke_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "${LOG_DIR}"

echo "=== Gate 2: GPU visibility ===" | tee "${LOG_FILE}"
sudo docker run --rm "${DSGN2_GPU_FLAG[@]}" "${DSGN2_LIB_ENV[@]}" "${IMAGE}" \
  python -c "import torch; print('cuda_available=', torch.cuda.is_available()); print('device=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')" \
  2>&1 | tee -a "${LOG_FILE}"

echo "" | tee -a "${LOG_FILE}"
echo "=== Gate 3: spconv import ===" | tee -a "${LOG_FILE}"
sudo docker run --rm "${DSGN2_GPU_FLAG[@]}" "${DSGN2_LIB_ENV[@]}" "${IMAGE}" \
  python -c "import spconv; print('spconv OK')" \
  2>&1 | tee -a "${LOG_FILE}"

echo "" | tee -a "${LOG_FILE}"
echo "=== Gate 3b: mmcv + pcdet import ===" | tee -a "${LOG_FILE}"
sudo docker run --rm "${DSGN2_GPU_FLAG[@]}" "${DSGN2_LIB_ENV[@]}" "${IMAGE}" \
  python -c "import mmcv; import pcdet; print('mmcv OK', mmcv.__version__); print('pcdet OK')" \
  2>&1 | tee -a "${LOG_FILE}"

echo ""
echo "Smoke test log: ${LOG_FILE}"
