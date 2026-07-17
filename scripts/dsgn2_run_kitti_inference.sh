#!/usr/bin/env bash
# Run DSGN++ inference inside PyTorch 1.7.1 Docker image on real KITTI frames.
# Data source: dsgn/datasets/adria/kitti-subset (51 frames, 000000-000050)
#
# Uses dsgn2.yaml (KITTI 1242x375 crop) — NOT dsgn2_awsim.yaml (1920x1080).
# Staging area: dsgn/datasets/adria/dsgn2_kitti/ (separate from dsgn2_awsim/).
#
# Outputs (per TAG):
#   dsgn/detections/dsgn2_kitti/<TAG>/*.txt     — 3D boxes (what Autoware needs)
#   dsgn/detections/dsgn2_kitti/<TAG>/2d/*.txt  — auxiliary 2D head only
#
# Usage:
#   bash ~/summer26/scripts/dsgn2_run_kitti_inference.sh
#   TAG=my_run bash ~/summer26/scripts/dsgn2_run_kitti_inference.sh
set -euo pipefail

ROOT="${HOME}/summer26"
# shellcheck disable=SC1091
source "${ROOT}/scripts/dsgn2_docker_common.sh"
IMAGE="${DSGN2_DOCKER_IMAGE}"
DSGN2="/workspace/DSGN2"
DSGN2_HOST="${ROOT}/external/DSGN2_awsim"
DSGN2_CONFIGS_HOST="${DSGN2_HOST}/configs"

KITTI_SUBSET="${ROOT}/dsgn/datasets/adria/kitti-subset"
DSGN2_DATA="${ROOT}/dsgn/datasets/adria/dsgn2_kitti"
SPLIT_FILE="${SPLIT_FILE:-${ROOT}/dsgn/datasets/adria/kitti_split.txt}"

# Use the real KITTI config (1242x375, standard crop MIN_REL_Y=1.0).
CFG_FILE="${CFG_FILE:-./configs/stereo/kitti_models/dsgn2.yaml}"

CKPT_DIR="${ROOT}/dsgn/checkpoints/dsgn2/kitti_pretrained"
CKPT="${CKPT:-${CKPT_DIR}/checkpoint_epoch_60.pth}"
GDRIVE_ID="${DSGN2_GDRIVE_ID:-1Z160fDx5abFZUARso1ixNJH-4UpjA4LI}"
TAG="${TAG:-kitti_subset}"
EVAL_TAG="${EVAL_TAG:-${TAG}}"
DETECTIONS_DIR="${DETECTIONS_DIR:-${ROOT}/dsgn/detections/dsgn2_kitti/${TAG}}"
DETECTIONS_2D_DIR="${DETECTIONS_2D_DIR:-${DETECTIONS_DIR}/2d}"
LOG_DIR="${ROOT}/logs"
LOG_FILE="${LOG_DIR}/dsgn2_kitti_inference_${TAG}.log"

mkdir -p "${LOG_DIR}" "${CKPT_DIR}" "${DSGN2_DATA}/ImageSets" "${DETECTIONS_DIR}" "${DETECTIONS_2D_DIR}"

if [[ ! -f "${SPLIT_FILE}" ]]; then
  echo "error: split file not found: ${SPLIT_FILE}" >&2
  exit 1
fi

if [[ ! -d "${KITTI_SUBSET}" ]]; then
  echo "error: kitti-subset not found: ${KITTI_SUBSET}" >&2
  exit 1
fi

if [[ ! -f "${DSGN2_HOST}/${CFG_FILE#./}" ]]; then
  echo "error: DSGN2 config not found: ${DSGN2_HOST}/${CFG_FILE#./}" >&2
  exit 1
fi

# --- Host: point dsgn2_kitti staging area at kitti-subset ---
# kitti-subset already has the right KITTI layout (calib, image_2, image_3, velodyne).
# Both training/ and testing/ point here; inference reads training/ (split=val path).
ln -sfn "${KITTI_SUBSET}" "${DSGN2_DATA}/training"
ln -sfn "${KITTI_SUBSET}" "${DSGN2_DATA}/testing"

cp "${SPLIT_FILE}" "${DSGN2_DATA}/ImageSets/val.txt"
head -1 "${SPLIT_FILE}" > "${DSGN2_DATA}/ImageSets/train.txt"
head -1 "${SPLIT_FILE}" > "${DSGN2_DATA}/ImageSets/test.txt"

# Invalidate cached pkl whenever the split changes — safer than stale cache.
rm -f "${DSGN2_DATA}/kitti_infos_val.pkl"

# --- Host: download pretrained KITTI checkpoint if missing ---
if [[ ! -f "${CKPT}" ]] || [[ $(stat -c%s "${CKPT}" 2>/dev/null || echo 0) -lt 1000000 ]]; then
  echo "Downloading DSGN++ pretrained checkpoint to ${CKPT} (~95 MB) ..."
  DSGN_VENV="${ROOT}/external/DSGN_custom/.venv"
  if [[ -f "${DSGN_VENV}/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${DSGN_VENV}/bin/activate"
    pip install -q gdown
    gdown "${GDRIVE_ID}" -O "${CKPT}"
  elif command -v gdown >/dev/null 2>&1; then
    gdown "${GDRIVE_ID}" -O "${CKPT}"
  else
    echo "error: need gdown to download checkpoint. Run: source ${DSGN_VENV}/bin/activate && pip install gdown" >&2
    exit 1
  fi
fi

echo "=== DSGN++ inference on real KITTI frames ===" | tee "${LOG_FILE}"
echo "Tag: ${TAG}" | tee -a "${LOG_FILE}"
echo "Data: ${KITTI_SUBSET}" | tee -a "${LOG_FILE}"
echo "Split: ${SPLIT_FILE} ($(wc -l < "${SPLIT_FILE}") frames)" | tee -a "${LOG_FILE}"
echo "Config: ${CFG_FILE}" | tee -a "${LOG_FILE}"
echo "Checkpoint: ${CKPT}" | tee -a "${LOG_FILE}"

# Mount only configs/ — full repo mount would hide image-built CUDA ops.
sudo docker run --rm "${DSGN2_GPU_FLAG[@]}" "${DSGN2_LIB_ENV[@]}" \
  -v "${ROOT}:${ROOT}" \
  -v "${DSGN2_CONFIGS_HOST}:${DSGN2}/configs" \
  -w "${DSGN2}" \
  -e HOME=/tmp \
  "${IMAGE}" \
  bash -lc "
    set -euo pipefail
    cd ${DSGN2}

    # Point OpenPCDet data root at dsgn2_kitti staging area.
    mkdir -p data
    ln -sfn ${DSGN2_DATA} data/kitti

    # Generate kitti_infos_val.pkl (label-free; no GT needed for inference).
    echo 'Generating kitti_infos_val.pkl...'
    python ${ROOT}/scripts/dsgn2_gen_val_infos.py ${DSGN2_DATA}/kitti_infos_val.pkl

    # matplotlib 3.6+ needs numpy>=1.20; image keeps numpy 1.19.5 for PT 1.7.1.
    pip install -q 'matplotlib==3.5.3'

    python ${ROOT}/scripts/dsgn2_test_wrapper.py \
      --launcher none \
      --workers 2 \
      --save_to_file \
      --cfg_file ${CFG_FILE} \
      --ckpt ${CKPT} \
      --eval_tag ${EVAL_TAG}
  " 2>&1 | tee -a "${LOG_FILE}"

# Collect outputs.
EVAL_BASE="${CKPT}.eval/eval/epoch_60/val/${EVAL_TAG}"
EVAL_3D="${EVAL_BASE}/final_result/data"
EVAL_2D="${EVAL_BASE}/final_result/data2d"

count_nonempty_txt() {
  local dir="$1"
  local n=0
  if [[ -d "${dir}" ]]; then
    local f
    for f in "${dir}"/*.txt; do
      [[ -f "${f}" && -s "${f}" ]] && n=$((n + 1))
    done
  fi
  echo "${n}"
}

if [[ -d "${EVAL_3D}" ]] && compgen -G "${EVAL_3D}/*.txt" > /dev/null; then
  cp -f "${EVAL_3D}"/*.txt "${DETECTIONS_DIR}/"
fi
if [[ -d "${EVAL_2D}" ]] && compgen -G "${EVAL_2D}/*.txt" > /dev/null; then
  cp -f "${EVAL_2D}"/*.txt "${DETECTIONS_2D_DIR}/"
fi
cp -f "${EVAL_BASE}/log_eval.txt" "${DETECTIONS_DIR}/" 2>/dev/null || true

n_3d="$(count_nonempty_txt "${EVAL_3D}")"
n_2d="$(count_nonempty_txt "${EVAL_2D}")"
echo "" | tee -a "${LOG_FILE}"
echo "=== Results ===" | tee -a "${LOG_FILE}"
echo "  frames with non-empty 3D detections: ${n_3d}" | tee -a "${LOG_FILE}"
echo "  frames with non-empty 2D detections: ${n_2d}" | tee -a "${LOG_FILE}"
if [[ "${n_3d}" -eq 0 ]]; then
  echo "  NOTE: no 3D detections written — check log for errors." | tee -a "${LOG_FILE}"
fi

echo ""
echo "3D detections: ${DETECTIONS_DIR}"
echo "2D detections: ${DETECTIONS_2D_DIR}"
echo "Full log: ${LOG_FILE}"
