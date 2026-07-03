#!/usr/bin/env bash
# Build Docker image for Arka-compatible DSGN inference (PyTorch 1.3).
# Requires sudo docker (adria runs this on host).
#
# Usage: bash ~/summer26/scripts/dsgn_build_docker.sh
set -euo pipefail

ROOT="${HOME}/summer26"
IMAGE="${DSGN_DOCKER_IMAGE:-dsgn-pt13:cuda10.1}"

cd "${ROOT}"
sudo docker build -f docker/dsgn_inference/Dockerfile -t "${IMAGE}" .

echo ""
echo "Built ${IMAGE}"
echo "Run inference: bash ${ROOT}/scripts/dsgn_run_inference_docker.sh"
