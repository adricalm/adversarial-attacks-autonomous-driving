#!/usr/bin/env bash
# Create DSGN venv (PT 2.6 + cu124). Usage: dsgn_setup_venv.sh
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
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

pip install \
  numpy scipy scikit-image numba yacs opencv-python fire tensorboardX \
  pillow imageio Cython

cd "${DSGN}"
pip install -e . --no-build-isolation

cd "${DSGN}/dsgn/utils/rotate_iou"
bash compile.sh

mkdir -p "${DSGN}/data/awsim"
ln -sfn "${ARKA_DS}/testing_offline" "${DSGN}/data/awsim/training"
ln -sfn "${ARKA_DS}/trainval.txt" "${DSGN}/data/awsim/trainval.txt"
ln -sfn "${ARKA_DS}/test.txt" "${DSGN}/data/awsim/test.txt"
ln -sfn "${ARKA_DS}/testing" "${DSGN}/data/awsim/testing"

echo ""
echo "DSGN venv ready: ${VENV}"
echo "Activate: source ${VENV}/bin/activate"
echo "Run inference: bash ${ROOT}/scripts/dsgn/dsgn_run_inference.sh"
