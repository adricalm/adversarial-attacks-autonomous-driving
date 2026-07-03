#!/usr/bin/env bash
# Run DSGN test_no_eval.py on a KITTI-layout dataset.
#
# WARNING: Uses PyTorch 2.6 — produces WRONG detections on Arka checkpoint.
# Use scripts/dsgn_run_inference_pt13.sh (or Docker) for faithful inference.
#
# Usage:
#   bash scripts/dsgn_run_inference.sh
#   DATA_PATH=~/summer26/data/arka/awsim/testing_offline_patched TAG=_patched_100_135 bash scripts/dsgn_run_inference.sh
set -euo pipefail

ROOT="${HOME}/summer26"
DSGN="${ROOT}/external/DSGN_custom"
VENV="${DSGN}/.venv"

DATA_PATH="${DATA_PATH:-${ROOT}/data/arka/awsim/testing_offline_patched}"
SPLIT_FILE="${SPLIT_FILE:-${ROOT}/data/arka/awsim/test_offline.txt}"
CFG="${CFG:-${DSGN}/configs/config_car_12g_awsim.py}"
LOADMODEL="${LOADMODEL:-${ROOT}/models/arka/dsgn_12g_awsim_remote_downsample/finetune_60.tar}"
TAG="${TAG:-_patched_100_135}"
GPU="${GPU:-0}"
BATCH="${BATCH:-1}"

if [[ ! -f "${VENV}/bin/activate" ]]; then
  echo "DSGN venv missing. Run: bash ${ROOT}/scripts/dsgn_setup_venv.sh" >&2
  exit 1
fi

if [[ ! -f "${LOADMODEL}" ]]; then
  echo "error: checkpoint not found: ${LOADMODEL}" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "${VENV}/bin/activate"
export PYTHONPATH="${DSGN}/tools:${PYTHONPATH:-}"

# DSGN loads calib via data/awsim/training/ (hardcoded in kitti_dataset.py).
mkdir -p "${DSGN}/data/awsim"
ln -sfn "${DATA_PATH}" "${DSGN}/data/awsim/training"
ln -sfn "${ROOT}/data/arka/awsim/trainval.txt" "${DSGN}/data/awsim/trainval.txt"
ln -sfn "${ROOT}/data/arka/awsim/test.txt" "${DSGN}/data/awsim/test.txt"
ln -sfn "${ROOT}/data/arka/awsim/testing" "${DSGN}/data/awsim/testing"

cd "${DSGN}/tools"
python test_no_eval.py \
  --cfg "${CFG}" \
  --data_path "${DATA_PATH}" \
  --split_file "${SPLIT_FILE}" \
  --loadmodel "${LOADMODEL}" \
  --btest "${BATCH}" \
  --devices "${GPU}" \
  --tag "${TAG}"

OUT_DIR="$(dirname "${LOADMODEL}")/awsim_output_2${TAG}"
echo ""
echo "Detections written to: ${OUT_DIR}"
