#!/usr/bin/env bash
# Experimental: PyTorch 1.10 + CUDA 11.3 on L40S — closer to Arka than PT 2.6.
# Validates whether an intermediate PT version gives faithful detections on this GPU.
#
# Usage: bash ~/summer26/scripts/archive/dsgn_pt_experiments/dsgn_setup_pt110.sh
set -eo pipefail

ROOT="${HOME}/summer26"
ARCHIVE="${ROOT}/scripts/archive/dsgn_pt_experiments"
DSGN="${ROOT}/external/DSGN_custom"
CONDA_ROOT="${ROOT}/.conda/miniconda3"
ENV_NAME="dsgn-pt110"
PATCH_FILE="${ROOT}/scripts/patches/dsgn_csrc_pt26.patch"

install_miniconda() {
  if [[ -x "${CONDA_ROOT}/bin/conda" ]]; then
    return 0
  fi
  echo "Installing Miniconda to ${CONDA_ROOT} ..."
  mkdir -p "${ROOT}/.conda"
  tmp="$(mktemp)"
  curl -fsSL "https://repo.anaconda.com/miniconda/Miniconda3-py37_4.12.0-Linux-x86_64.sh" -o "${tmp}"
  bash "${tmp}" -b -p "${CONDA_ROOT}"
  rm -f "${tmp}"
}

install_miniconda
# shellcheck disable=SC1091
source "${CONDA_ROOT}/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" python=3.8 pip
fi

conda activate "${ENV_NAME}"
conda install -y -c pytorch -c conda-forge \
  pytorch=1.10.2 torchvision=0.11.3 cudatoolkit=11.3 \
  gxx_linux-64=9.5.0 gcc_linux-64=9.5.0 libxcrypt

export CC="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc"
export CXX="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++"

pip install \
  "numpy<1.24" "scipy<1.10" "scikit-image<0.21" "numba<0.58" \
  yacs opencv-python fire tensorboardX pillow imageio Cython

cd "${DSGN}"
RESTORE_PT26=0
if [[ -f "${PATCH_FILE}" ]] && ! git diff --quiet -- dsgn/csrc; then
  RESTORE_PT26=1
  git checkout HEAD -- dsgn/csrc
fi

pip install -e . --no-build-isolation

if [[ "${RESTORE_PT26}" -eq 1 ]]; then
  git apply "${PATCH_FILE}"
fi

cd "${DSGN}/dsgn/utils/rotate_iou"
bash compile.sh

echo ""
echo "PT 1.10 env ready: conda activate ${ENV_NAME}"
echo "Validate: GPU=0 TAG=_pt110_validate_frame10 SPLIT_FILE=${ROOT}/dsgn/datasets/arka/dsgn_awsim/test_offline_frame10.txt bash ${ARCHIVE}/dsgn_run_inference_pt110.sh"
