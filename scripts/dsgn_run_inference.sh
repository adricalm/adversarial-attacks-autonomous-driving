#!/usr/bin/env bash
# Run DSGN test_no_eval.py on a KITTI-layout dataset.
#
# WARNING: Uses PyTorch 2.6 — produces WRONG detections on Arka checkpoint.
# Faithful PT 1.3 inference is not possible on this L40S host (CUDA ops + sm_89).
# For correct detections use Docker on an old GPU (dsgn_build_docker.sh) or replay
# Arka's precomputed outputs (notes/DSGN_OFFLINE_RUNBOOK.md).
#
# Usage:
#   bash scripts/dsgn_run_inference.sh
#   DATA_PATH=~/summer26/dsgn/datasets/adria/testing_offline_patched TAG=_patched_100_135 bash scripts/dsgn_run_inference.sh
#   bash scripts/dsgn_run_inference.sh --debug   # optional extra args to test_no_eval.py
# GPU timing bench (separate script): bash scripts/dsgn_bench_inference.sh
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

# test_no_eval.py writes next to the checkpoint; move to dsgn/detections/adria/.
RAW_OUT="$(dirname "${LOADMODEL}")/awsim_output_2${TAG}" # Arka's output, which we'll overwrite.
OUT_DIR="${DETECTIONS_DIR}/${TAG#_}"
if [[ -d "${RAW_OUT}" ]]; then
  mkdir -p "${DETECTIONS_DIR}"
  rm -rf "${OUT_DIR}"
  mv "${RAW_OUT}" "${OUT_DIR}"
fi

echo ""
echo "Detections written to: ${OUT_DIR}"
