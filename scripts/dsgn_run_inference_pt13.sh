#!/usr/bin/env bash
# Run DSGN test_no_eval.py with Arka-compatible PyTorch 1.3 env.
#
# Usage:
#   bash scripts/dsgn_run_inference_pt13.sh
#   DATA_PATH=~/summer26/data/arka/awsim/testing_offline_patched \
#     TAG=_patched_100_135_pt13 bash scripts/dsgn_run_inference_pt13.sh
#   SPLIT_FILE=~/summer26/data/arka/awsim/test_offline_100_135.txt bash scripts/dsgn_run_inference_pt13.sh
set -eo pipefail

ROOT="${HOME}/summer26"
DSGN="${ROOT}/external/DSGN_custom"
CONDA_ROOT="${ROOT}/.conda/miniconda3"
ENV_NAME="dsgn-pt13"

DATA_PATH="${DATA_PATH:-${ROOT}/data/arka/awsim/testing_offline_patched}"
SPLIT_FILE="${SPLIT_FILE:-${ROOT}/data/arka/awsim/test_offline.txt}"
CFG="${CFG:-${DSGN}/configs/config_car_12g_awsim.py}"
LOADMODEL="${LOADMODEL:-${ROOT}/models/arka/dsgn_12g_awsim_remote_downsample/finetune_60_legacy.tar}"
TAG="${TAG:-_patched_100_135_pt13}"
GPU="${GPU:-cpu}"  # PT 1.3 + CUDA 10.1 cannot use L40S; default CPU
BATCH="${BATCH:-1}"

if [[ ! -x "${CONDA_ROOT}/bin/conda" ]]; then
  echo "DSGN PT1.3 env missing. Run: bash ${ROOT}/scripts/dsgn_setup_pt13.sh" >&2
  exit 1
fi

if [[ ! -f "${LOADMODEL}" ]]; then
  echo "error: checkpoint not found: ${LOADMODEL}" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
# Drop any active PT2.6 venv that shadows conda on PATH.
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  deactivate || true
fi
conda activate "${ENV_NAME}"
PYTHON="${CONDA_PREFIX}/bin/python"

export PYTHONPATH="${DSGN}/tools:${PYTHONPATH:-}"

# DSGN loads calib via data/awsim/training/ (hardcoded in kitti_dataset.py).
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
echo ""
echo "Detections written to: ${OUT_DIR}"
