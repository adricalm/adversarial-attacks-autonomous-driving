#!/usr/bin/env bash
# Run DSGN test_no_eval.py with Arka-compatible PyTorch 1.3 env.
#
# Usage:
#   bash scripts/archive/dsgn_pt_experiments/dsgn_run_inference_pt13.sh
#   DATA_PATH=~/summer26/dsgn/datasets/adria/testing_offline_patched \
#     TAG=_patched_100_135_pt13 bash scripts/archive/dsgn_pt_experiments/dsgn_run_inference_pt13.sh
set -eo pipefail

ROOT="${HOME}/summer26"
ARCHIVE="${ROOT}/scripts/archive/dsgn_pt_experiments"
DSGN="${ROOT}/external/DSGN_custom"
CONDA_ROOT="${ROOT}/.conda/miniconda3"
ENV_NAME="dsgn-pt13"
ARKA_DS="${ROOT}/dsgn/datasets/arka/dsgn_awsim"
ADRIA_DS="${ROOT}/dsgn/datasets/adria"

DATA_PATH="${DATA_PATH:-${ADRIA_DS}/testing_offline_patched}"
SPLIT_FILE="${SPLIT_FILE:-${ARKA_DS}/test_offline.txt}"
CFG="${CFG:-${DSGN}/configs/config_car_12g_awsim.py}"
LOADMODEL="${LOADMODEL:-${ROOT}/dsgn/checkpoints/arka/dsgn_12g_awsim_remote_downsample/finetune_60_legacy.tar}"
DETECTIONS_DIR="${DETECTIONS_DIR:-${ROOT}/dsgn/detections/adria}"
TAG="${TAG:-_patched_100_135_pt13}"
GPU="${GPU:-cpu}"  # PT 1.3 + CUDA 10.1 cannot use L40S; default CPU
BATCH="${BATCH:-1}"

if [[ ! -x "${CONDA_ROOT}/bin/conda" ]]; then
  echo "DSGN PT1.3 env missing. Run: bash ${ARCHIVE}/dsgn_setup_pt13.sh" >&2
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

echo ""
echo "Detections written to: ${OUT_DIR}"
