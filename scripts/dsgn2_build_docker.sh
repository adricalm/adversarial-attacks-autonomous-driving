#!/usr/bin/env bash
# Build Docker image for DSGN++ inference (PyTorch 1.7.1, CUDA 11.0).
# Requires sudo docker (adria runs this on host).
#
# Usage: bash ~/summer26/scripts/dsgn2_build_docker.sh
set -euo pipefail

ROOT="${HOME}/summer26"
IMAGE="${DSGN2_DOCKER_IMAGE:-dsgn2:pt171}"
LOG_DIR="${ROOT}/logs"
LOG_FILE="${LOG_DIR}/dsgn2_build_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "${LOG_DIR}"

echo "Building ${IMAGE} (log: ${LOG_FILE})"
echo "This may take 30-60 minutes (spconv + mmcv + OpenPCDet CUDA compile)."

cd "${ROOT}"
sudo docker build -f docker/dsgn2_inference/Dockerfile -t "${IMAGE}" . 2>&1 | tee "${LOG_FILE}"

echo ""
echo "Built ${IMAGE}"
echo "Smoke test: bash ${ROOT}/scripts/dsgn2_smoke_test.sh"
echo "Inference:    bash ${ROOT}/scripts/dsgn2_run_inference.sh"
