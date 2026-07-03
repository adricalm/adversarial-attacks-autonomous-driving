#!/usr/bin/env bash
# Arka-compatible DSGN inference env: Python 3.7 + PyTorch 1.3 + torchvision 0.4.1.
#
# PyTorch 2.x (see dsgn_setup_venv.sh) produces garbage detections on this checkpoint.
# This env builds DSGN against the ORIGINAL csrc (pre-PT2 API patches).
#
# GPU note: PyTorch 1.3 + CUDA 10.1 does not support L40S/Ada GPUs. Default is CPU.
# For GPU inference you need an older NVIDIA GPU or a remote V100/Titan box.
#
# Usage: bash ~/summer26/scripts/dsgn_setup_pt13.sh
set -eo pipefail

ROOT="${HOME}/summer26"
DSGN="${ROOT}/external/DSGN_custom"
CONDA_ROOT="${ROOT}/.conda/miniconda3"
ENV_NAME="dsgn-pt13"
PATCH_FILE="${ROOT}/scripts/patches/dsgn_csrc_pt26.patch"
LEGACY_CKPT="${ROOT}/models/arka/dsgn_12g_awsim_remote_downsample/finetune_60_legacy.tar"
ORIG_CKPT="${ROOT}/models/arka/dsgn_12g_awsim_remote_downsample/finetune_60.tar"

if [[ ! -d "${DSGN}" ]]; then
  echo "error: ${DSGN} not found" >&2
  exit 1
fi

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
  echo "Creating conda env ${ENV_NAME} (python 3.7) ..."
  conda create -y -n "${ENV_NAME}" python=3.7 pip
fi

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  deactivate || true
fi
conda activate "${ENV_NAME}"

pip install --upgrade "pip<24" wheel setuptools

# Direct wheels — the torch_stable.html index is unreliable on modern pip.
pip install \
  "https://download.pytorch.org/whl/cpu/torch-1.3.0%2Bcpu-cp37-cp37m-linux_x86_64.whl" \
  "https://download.pytorch.org/whl/cpu/torchvision-0.4.1%2Bcpu-cp37-cp37m-linux_x86_64.whl"

pip install \
  "numpy<1.22" "scipy<1.8" "scikit-image<0.20" "numba<0.57" \
  "yacs" "opencv-python<4.9" "fire" "tensorboardX" "pillow==6.2.2" "imageio<2.31" "Cython"

# Host GCC 15 cannot compile PyTorch 1.3 C++ extensions; use conda GCC 9 + libcrypt.
conda install -y -c conda-forge gxx_linux-64=9.5.0 gcc_linux-64=9.5.0 libxcrypt
export CC="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc"
export CXX="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++"
export CUDA_HOME=""

# Build against original (PT 1.x) csrc — revert PT 2.6 API patches temporarily.
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

# Calib loader reads data/awsim/training/ (see kitti_dataset.py).
mkdir -p "${DSGN}/data/awsim"
ln -sfn "${ROOT}/data/arka/awsim/testing_offline" "${DSGN}/data/awsim/training"
ln -sfn "${ROOT}/data/arka/awsim/trainval.txt" "${DSGN}/data/awsim/trainval.txt"
ln -sfn "${ROOT}/data/arka/awsim/test.txt" "${DSGN}/data/awsim/test.txt"
ln -sfn "${ROOT}/data/arka/awsim/testing" "${DSGN}/data/awsim/testing"

echo ""
echo "DSGN Arka-compatible env ready: conda activate ${ENV_NAME}"
echo "  (or: source ${CONDA_ROOT}/etc/profile.d/conda.sh && conda activate ${ENV_NAME})"
echo "Run inference: bash ${ROOT}/scripts/dsgn_run_inference_pt13.sh"
echo ""
echo "Note: CPU-only by default. Expect ~minutes per frame on L40S host."

# PyTorch 1.3 cannot read zip-serialized checkpoints (saved with PT >= 1.6).
if [[ ! -f "${LEGACY_CKPT}" && -f "${ORIG_CKPT}" ]]; then
  echo "Converting checkpoint to legacy format for PyTorch 1.3 ..."
  if [[ -x "${ROOT}/external/DSGN_custom/.venv/bin/python" ]]; then
    "${ROOT}/external/DSGN_custom/.venv/bin/python" - <<PY
import torch
src = "${ORIG_CKPT}"
dst = "${LEGACY_CKPT}"
d = torch.load(src, map_location="cpu")
torch.save(d, dst, _use_new_zipfile_serialization=False)
print("Wrote", dst)
PY
  else
    echo "warning: PT2.6 venv missing; create ${LEGACY_CKPT} manually before PT1.3 inference" >&2
  fi
fi
