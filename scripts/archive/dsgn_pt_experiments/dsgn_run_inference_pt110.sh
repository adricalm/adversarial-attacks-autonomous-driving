#!/usr/bin/env bash
# Run DSGN with PyTorch 1.10 (GPU). See dsgn_setup_pt110.sh in this directory.
set -eo pipefail

ROOT="${HOME}/summer26"
DSGN="${ROOT}/external/DSGN_custom"
CONDA_ROOT="${ROOT}/.conda/miniconda3"
ENV_NAME="dsgn-pt110"
ARKA_DS="${ROOT}/dsgn/datasets/arka/dsgn_awsim"

DATA_PATH="${DATA_PATH:-${ARKA_DS}/testing_offline}"
SPLIT_FILE="${SPLIT_FILE:-${ARKA_DS}/test_offline.txt}"
CFG="${CFG:-${DSGN}/configs/config_car_12g_awsim.py}"
LOADMODEL="${LOADMODEL:-${ROOT}/dsgn/checkpoints/arka/dsgn_12g_awsim_remote_downsample/finetune_60.tar}"
DETECTIONS_DIR="${DETECTIONS_DIR:-${ROOT}/dsgn/detections/adria}"
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
ln -sfn "${ARKA_DS}/trainval.txt" "${DSGN}/data/awsim/trainval.txt"
ln -sfn "${ARKA_DS}/test.txt" "${DSGN}/data/awsim/test.txt"
ln -sfn "${ARKA_DS}/testing" "${DSGN}/data/awsim/testing"

cd "${DSGN}/tools"
"${PYTHON}" test_no_eval.py \
  --cfg "${CFG}" \
  --data_path "${DATA_PATH}" \
  --split_file "${SPLIT_FILE}" \
  --loadmodel "${LOADMODEL}" \
  --btest "${BATCH}" \
  --devices "${GPU}" \
  --tag "${TAG}"

RAW_OUT="$(dirname "${LOADMODEL}")/awsim_output_2${TAG}"
OUT_DIR="${DETECTIONS_DIR}/${TAG#_}"
if [[ -d "${RAW_OUT}" ]]; then
  mkdir -p "${DETECTIONS_DIR}"
  rm -rf "${OUT_DIR}"
  mv "${RAW_OUT}" "${OUT_DIR}"
fi

echo "Detections written to: ${OUT_DIR}"
