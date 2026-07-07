#!/usr/bin/env bash
# Fine-tune DSGN on AWSIM stereo data (PyTorch 2.6).
#
# Usage:
#   bash scripts/dsgn_train.sh
#   DEBUG=1 EPOCHS=1 bash scripts/dsgn_train.sh   # smoke test
#   FORCE_TARGETS=1 bash scripts/dsgn_train.sh      # re-run generate_targets
set -euo pipefail

ROOT="${HOME}/summer26"
DSGN="${ROOT}/external/DSGN_custom"
VENV="${DSGN}/.venv"

DATA_PATH="${DATA_PATH:-${ROOT}/data/arka/awsim/training}"
SPLIT_FILE="${SPLIT_FILE:-${ROOT}/data/arka/awsim/trainval.txt}"
CFG="${CFG:-${DSGN}/configs/config_car_12g_awsim.py}"
LOADMODEL="${LOADMODEL:-${ROOT}/models/kitti/dsgn_12g_b/finetune_48.tar}"
SAVEMODEL="${SAVEMODEL:-${ROOT}/models/awsim_pt26/dsgn_12g_awsim}"
EPOCHS="${EPOCHS:-60}"
BTRAIN="${BTRAIN:-1}"
GPU="${GPU:-0}"
START_EPOCH="${START_EPOCH:-1}"
FORCE_TARGETS="${FORCE_TARGETS:-0}"

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

# Labels/calib are read via data/awsim/training/ (hardcoded in kitti_dataset.py).
mkdir -p "${DSGN}/data/awsim"
ln -sfn "${DATA_PATH}" "${DSGN}/data/awsim/training"
ln -sfn "${ROOT}/data/arka/awsim/trainval.txt" "${DSGN}/data/awsim/trainval.txt"
ln -sfn "${ROOT}/data/arka/awsim/train.txt" "${DSGN}/data/awsim/train.txt"
ln -sfn "${ROOT}/data/arka/awsim/val.txt" "${DSGN}/data/awsim/val.txt"
ln -sfn "${ROOT}/data/arka/awsim/test.txt" "${DSGN}/data/awsim/test.txt"
ln -sfn "${ROOT}/data/arka/awsim/testing" "${DSGN}/data/awsim/testing"

cd "${DSGN}/tools"

TARGETS_DIR="./outputs/temp/anchor_4angles_trainval_validclass_2_lesscar"
if [[ "${FORCE_TARGETS}" == "1" ]] || [[ ! -d "${TARGETS_DIR}" ]] || [[ -z "$(ls -A "${TARGETS_DIR}" 2>/dev/null)" ]]; then
  echo "==> Pre-computing bbox targets..."
  python generate_targets.py \
    --cfg "${CFG}" \
    --data_path "${DATA_PATH}" \
    --split_file "${SPLIT_FILE}"
else
  echo "==> Skipping generate_targets (targets exist in ${TARGETS_DIR}); set FORCE_TARGETS=1 to re-run"
fi

mkdir -p "${SAVEMODEL}"

TRAIN_ARGS=(
  --cfg "${CFG}"
  --data_path "${DATA_PATH}"
  --split_file "${SPLIT_FILE}"
  --loadmodel "${LOADMODEL}"
  --savemodel "${SAVEMODEL}"
  --epochs "${EPOCHS}"
  --start_epoch "${START_EPOCH}"
  -btrain "${BTRAIN}"
  -d "${GPU}"
)

if [[ "${DEBUG:-0}" == "1" ]]; then
  TRAIN_ARGS+=(--debug)
fi

echo "==> Fine-tuning DSGN (${EPOCHS} epochs, batch ${BTRAIN})..."
python train_net.py "${TRAIN_ARGS[@]}"

echo ""
echo "Checkpoints saved to: ${SAVEMODEL}"
