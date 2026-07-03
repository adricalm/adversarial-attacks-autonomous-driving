#!/usr/bin/env bash
# Run DSGN inference inside PyTorch 1.3 Docker image.
# Requires: sudo docker, image built via dsgn_build_docker.sh
#
# GPU warning: CUDA 10.1 in this image does NOT support L40S/Ada. Expect runtime
# failure with --gpus all on this host; use CPU fallback or an older GPU machine.
#
# Usage:
#   bash scripts/dsgn_run_inference_docker.sh
#   USE_GPU=0 bash scripts/dsgn_run_inference_docker.sh   # CPU fallback
set -euo pipefail

ROOT="${HOME}/summer26"
DSGN="${ROOT}/external/DSGN_custom"
IMAGE="${DSGN_DOCKER_IMAGE:-dsgn-pt13:cuda10.1}"
PATCH_FILE="${ROOT}/scripts/patches/dsgn_csrc_pt26.patch"

DATA_PATH="${DATA_PATH:-${ROOT}/data/arka/awsim/testing_offline_patched}"
SPLIT_FILE="${SPLIT_FILE:-${ROOT}/data/arka/awsim/test_offline.txt}"
LOADMODEL="${LOADMODEL:-${ROOT}/models/arka/dsgn_12g_awsim_remote_downsample/finetune_60.tar}"
TAG="${TAG:-_patched_100_135_pt13_docker}"
USE_GPU="${USE_GPU:-1}"

GPU_FLAG=()
if [[ "${USE_GPU}" == "1" ]]; then
  GPU_FLAG=(--gpus all)
else
  GPU_FLAG=(-e CUDA_VISIBLE_DEVICES=)
fi

sudo docker run --rm -it "${GPU_FLAG[@]}" \
  -v "${ROOT}:${ROOT}" \
  -w "${DSGN}/tools" \
  -e PYTHONPATH="${DSGN}/tools" \
  -e HOME=/tmp \
  "${IMAGE}" \
  bash -lc "
    set -euo pipefail
    cd ${DSGN}
    if [[ -f ${PATCH_FILE} ]] && ! git diff --quiet -- dsgn/csrc 2>/dev/null; then
      git checkout HEAD -- dsgn/csrc
    fi
    pip install -q -e . --no-build-isolation
    cd dsgn/utils/rotate_iou && bash compile.sh
  "

sudo docker run --rm "${GPU_FLAG[@]}" \
  -v "${ROOT}:${ROOT}" \
  -w "${DSGN}/tools" \
  -e PYTHONPATH="${DSGN}/tools" \
  -e HOME=/tmp \
  "${IMAGE}" \
  python test_no_eval.py \
    --cfg "${DSGN}/configs/config_car_12g_awsim.py" \
    --data_path "${DATA_PATH}" \
    --split_file "${SPLIT_FILE}" \
    --loadmodel "${LOADMODEL}" \
    --btest 1 \
    --devices 0 \
    --tag "${TAG}"

OUT_DIR="$(dirname "${LOADMODEL}")/awsim_output_2${TAG}"
echo ""
echo "Detections written to: ${OUT_DIR}"
