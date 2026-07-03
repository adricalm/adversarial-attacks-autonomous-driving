#!/usr/bin/env bash
# Run DSGN with PyTorch 1.10 (GPU). See dsgn_setup_pt110.sh.
set -eo pipefail

ROOT="${HOME}/summer26"
DSGN="${ROOT}/external/DSGN_custom"
CONDA_ROOT="${ROOT}/.conda/miniconda3"
ENV_NAME="dsgn-pt110"

DATA_PATH="${DATA_PATH:-${ROOT}/data/arka/awsim/testing_offline}"
SPLIT_FILE="${SPLIT_FILE:-${ROOT}/data/arka/awsim/test_offline.txt}"
CFG="${CFG:-${DSGN}/configs/config_car_12g_awsim.py}"
LOADMODEL="${LOADMODEL:-${ROOT}/models/arka/dsgn_12g_awsim_remote_downsample/finetune_60.tar}"
TAG="${TAG:-_pt110}"
GPU="${GPU:-0}"
BATCH="${BATCH:-1}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
if [[ -n "${VIRTUAL_ENV:-}" ]]; then deactivate || true; fi
conda activate "${ENV_NAME}"
PYTHON="${CONDA_PREFIX}/bin/python"

export PYTHONPATH="${DSGN}/tools:${PYTHONPATH:-}"

mkdir -p "${DSGN}/data/awsim"
ln -sfn "${DATA_PATH}" "${DSGN}/data/awsim/training"
ln -sfn "${ROOT}/data/arka/awsim/trainval.txt" "${DSGN}/data/awsim/trainval.txt"
ln -sfn "${ROOT}/data/arka/awsim/test.txt" "${DSGN}/data/awsim/test.txt"
ln -sfn "${ROOT}/data/arka/awsim/testing" "${DSGN}/data/awsim/testing"

cd "${DSGN}/tools"
"${PYTHON}" test_no_eval.py \
  --cfg "${CFG}" \
  --data_path "${DATA_PATH}" \
  --split_file "${SPLIT_FILE}" \
  --loadmodel "${LOADMODEL}" \
  --btest "${BATCH}" \
  --devices "${GPU}" \
  --tag "${TAG}"

OUT_DIR="$(dirname "${LOADMODEL}")/awsim_output_2${TAG}"
echo "Detections written to: ${OUT_DIR}"
