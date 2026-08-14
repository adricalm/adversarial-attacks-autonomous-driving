#!/usr/bin/env bash
# Run DSGN test_no_eval.py. Default: finetune_48.
# Usage: dsgn_run_inference.sh [extra test_no_eval.py args]
set -euo pipefail

ROOT="${HOME}/summer26"
DSGN="${ROOT}/external/DSGN_custom"
VENV="${DSGN}/.venv"
ARKA_DS="${ROOT}/dsgn/datasets/arka/dsgn_awsim"
ADRIA_DS="${ROOT}/dsgn/datasets/adria"

DATA_PATH="${DATA_PATH:-${ARKA_DS}/testing_offline}"
SPLIT_FILE="${SPLIT_FILE:-${ARKA_DS}/test_offline_debug.txt}"
CFG="${CFG:-${DSGN}/configs/config_car_12g_awsim.py}"
LOADMODEL="${LOADMODEL:-${ROOT}/dsgn/checkpoints/kitti/dsgn_12g_b/finetune_48.tar}"
DETECTIONS_DIR="${DETECTIONS_DIR:-${ROOT}/dsgn/detections/adria}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
TAG="${TAG:-$(basename "${LOADMODEL}" .tar)_${RUN_STAMP}}"
GPU="${GPU:-0}"
BATCH="${BATCH:-1}"

if [[ ! -f "${VENV}/bin/activate" ]]; then
  echo "DSGN venv missing. Run: bash ${ROOT}/scripts/dsgn/dsgn_setup_venv.sh" >&2
  exit 1
fi

if [[ ! -f "${LOADMODEL}" ]]; then
  echo "error: checkpoint not found: ${LOADMODEL}" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "${VENV}/bin/activate"
export PYTHONPATH="${DSGN}/tools:${PYTHONPATH:-}"

mkdir -p "${DSGN}/data/awsim"
ln -sfn "${DATA_PATH}" "${DSGN}/data/awsim/training"
ln -sfn "${ARKA_DS}/trainval.txt" "${DSGN}/data/awsim/trainval.txt"
ln -sfn "${ARKA_DS}/test.txt" "${DSGN}/data/awsim/test.txt"
ln -sfn "${ARKA_DS}/testing" "${DSGN}/data/awsim/testing"

cd "${DSGN}/tools"
python test_no_eval.py \
  --cfg "${CFG}" \
  --data_path "${DATA_PATH}" \
  --split_file "${SPLIT_FILE}" \
  --loadmodel "${LOADMODEL}" \
  --btest "${BATCH}" \
  --devices "${GPU}" \
  --tag "${TAG}" \
  "$@"

RAW_OUT="$(dirname "${LOADMODEL}")/awsim_output_2${TAG}"
OUT_DIR="${DETECTIONS_DIR}/${TAG#_}"
if [[ -d "${RAW_OUT}" ]]; then
  mkdir -p "${DETECTIONS_DIR}"
  rm -rf "${OUT_DIR}"
  mv "${RAW_OUT}" "${OUT_DIR}"
fi

echo ""
echo "Detections written to: ${OUT_DIR}"
