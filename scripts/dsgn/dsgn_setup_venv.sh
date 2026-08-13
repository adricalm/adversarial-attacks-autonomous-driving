#!/usr/bin/env bash
# DSGN inference environment (PyTorch 2.6 + CUDA 12.4).
#
# WARNING: This env compiles and runs, but produces WRONG detections on Arka's
# finetune_60.tar checkpoint (dozens of false cars on clean frames). For faithful
# For Arka finetune_60, prefer precomputed dumps; official finetune_48 works on PT 2.6.
#
# Usage: bash ~/summer26/scripts/dsgn/dsgn_setup_venv.sh
set -euo pipefail

ROOT="${HOME}/summer26"
DSGN="${ROOT}/external/DSGN_custom"
VENV="${DSGN}/.venv"
ARKA_DS="${ROOT}/dsgn/datasets/arka/dsgn_awsim"

if [[ ! -d "${DSGN}" ]]; then
  echo "error: ${DSGN} not found" >&2
  exit 1
fi

python3 -m venv "${VENV}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

pip install --upgrade pip wheel setuptools
# CUDA 12.4 wheels; matches nvcc 12.4 on this host.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

pip install \
  numpy scipy scikit-image numba yacs opencv-python fire tensorboardX \
  pillow imageio Cython

cd "${DSGN}"
pip install -e . --no-build-isolation

cd "${DSGN}/dsgn/utils/rotate_iou"
bash compile.sh

# Calib loader reads data/awsim/training/ (see kitti_dataset.py).
mkdir -p "${DSGN}/data/awsim"
ln -sfn "${ARKA_DS}/testing_offline" "${DSGN}/data/awsim/training"
ln -sfn "${ARKA_DS}/trainval.txt" "${DSGN}/data/awsim/trainval.txt"
ln -sfn "${ARKA_DS}/test.txt" "${DSGN}/data/awsim/test.txt"
ln -sfn "${ARKA_DS}/testing" "${DSGN}/data/awsim/testing"

echo ""
echo "DSGN venv ready: ${VENV}"
echo "Activate: source ${VENV}/bin/activate"
echo "Run inference: bash ${ROOT}/scripts/dsgn/dsgn_run_inference.sh"
