#!/usr/bin/env bash
# Exclusive-GPU timing bench for DSGN (does not modify research inference path).
# Calls tools/test_no_eval_timing.py — leave test_no_eval.py alone.
#
# Usage:
#   bash scripts/dsgn_bench_inference.sh
#   SPLIT_FILE=~/summer26/dsgn/datasets/arka/dsgn_awsim/test_offline.txt \
#     bash scripts/dsgn_bench_inference.sh --max_frames 30 --warmup 5 --no_write
set -euo pipefail

ROOT="${HOME}/summer26"
DSGN="${ROOT}/external/DSGN_custom"
VENV="${DSGN}/.venv"
ARKA_DS="${ROOT}/dsgn/datasets/arka/dsgn_awsim"

DATA_PATH="${DATA_PATH:-${ARKA_DS}/testing_offline}"
SPLIT_FILE="${SPLIT_FILE:-${ARKA_DS}/test_offline.txt}"
CFG="${CFG:-${DSGN}/configs/config_car_12g_awsim.py}"
LOADMODEL="${LOADMODEL:-${ROOT}/dsgn/checkpoints/kitti/dsgn_12g_b/finetune_48.tar}"
TAG="${TAG:-_timing_bench}"
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

mkdir -p "${DSGN}/data/awsim"
ln -sfn "${DATA_PATH}" "${DSGN}/data/awsim/training"
ln -sfn "${ARKA_DS}/trainval.txt" "${DSGN}/data/awsim/trainval.txt"
ln -sfn "${ARKA_DS}/test.txt" "${DSGN}/data/awsim/test.txt"
ln -sfn "${ARKA_DS}/testing" "${DSGN}/data/awsim/testing"

cd "${DSGN}/tools"
# Default bench flags; override / extend via "$@". Empty "$@" is a no-op.
python test_no_eval_timing.py \
  --cfg "${CFG}" \
  --data_path "${DATA_PATH}" \
  --split_file "${SPLIT_FILE}" \
  --loadmodel "${LOADMODEL}" \
  --btest "${BATCH}" \
  --devices "${GPU}" \
  --tag "${TAG}" \
  --max_frames "${MAX_FRAMES:-30}" \
  --warmup "${WARMUP:-5}" \
  --no_write \
  "$@"
