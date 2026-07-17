#!/usr/bin/env bash
# Run DSGN++ inference inside PyTorch 1.7.1 Docker image on AWSIM KITTI-layout data.
# Requires: sudo docker, image built via dsgn2_build_docker.sh
#
# Data layout (NOT real KITTI — AWSIM testing_offline in KITTI folder shape):
#   dsgn/datasets/adria/dsgn2_awsim/training -> arka/dsgn_awsim/testing_offline
# Inside Docker only: /workspace/DSGN2/data/kitti -> same adria/dsgn2_awsim path.
# Do not use external/DSGN2_awsim/data/ on the host (stale symlink if present).
#
# Outputs (Autoware needs 3D KITTI txt under detections/dsgn2/):
#   dsgn/detections/dsgn2/*.txt     — 3D boxes (final_result/data)
#   dsgn/detections/dsgn2/2d/*.txt  — auxiliary 2D head only (final_result/data2d)
#
# Usage:
#   bash ~/summer26/scripts/dsgn2_run_inference.sh
#   SPLIT_FILE=~/summer26/dsgn/datasets/arka/dsgn_awsim/test_offline_debug.txt bash scripts/dsgn2_run_inference.sh
set -euo pipefail

ROOT="${HOME}/summer26"
# shellcheck disable=SC1091
source "${ROOT}/scripts/dsgn2_docker_common.sh"
IMAGE="${DSGN2_DOCKER_IMAGE}"
DSGN2="/workspace/DSGN2"
DSGN2_HOST="${ROOT}/external/DSGN2_awsim"
DSGN2_CONFIGS_HOST="${DSGN2_HOST}/configs"
CFG_FILE="${CFG_FILE:-./configs/stereo/kitti_models/dsgn2_awsim.yaml}"
ARKA_DS="${ROOT}/dsgn/datasets/arka/dsgn_awsim"
DSGN2_DATA="${ROOT}/dsgn/datasets/adria/dsgn2_awsim"
SPLIT_FILE="${SPLIT_FILE:-${ARKA_DS}/test_offline_debug.txt}"
CKPT_DIR="${ROOT}/dsgn/checkpoints/dsgn2/kitti_pretrained"
CKPT="${CKPT:-${CKPT_DIR}/checkpoint_epoch_60.pth}"
GDRIVE_ID="${DSGN2_GDRIVE_ID:-1Z160fDx5abFZUARso1ixNJH-4UpjA4LI}"
DETECTIONS_DIR="${DETECTIONS_DIR:-${ROOT}/dsgn/detections/dsgn2}"
DETECTIONS_2D_DIR="${DETECTIONS_2D_DIR:-${DETECTIONS_DIR}/2d}"
EVAL_TAG="${EVAL_TAG:-awsim_debug}"
LOG_DIR="${ROOT}/logs"
LOG_FILE="${LOG_DIR}/dsgn2_inference_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "${LOG_DIR}" "${CKPT_DIR}" "${DSGN2_DATA}/ImageSets" "${DETECTIONS_DIR}" "${DETECTIONS_2D_DIR}"

if [[ ! -f "${SPLIT_FILE}" ]]; then
  echo "error: split file not found: ${SPLIT_FILE}" >&2
  exit 1
fi

if [[ ! -f "${DSGN2_HOST}/${CFG_FILE#./}" ]]; then
  echo "error: DSGN2 config not found: ${DSGN2_HOST}/${CFG_FILE#./}" >&2
  exit 1
fi

# --- Host: lay out AWSIM testing_offline as KITTI training/ for val inference ---
ln -sfn "${ARKA_DS}/testing_offline" "${DSGN2_DATA}/training"
ln -sfn "${ARKA_DS}/testing_offline" "${DSGN2_DATA}/testing"

cp "${SPLIT_FILE}" "${DSGN2_DATA}/ImageSets/val.txt"
head -1 "${SPLIT_FILE}" > "${DSGN2_DATA}/ImageSets/train.txt"
head -1 "${SPLIT_FILE}" > "${DSGN2_DATA}/ImageSets/test.txt"

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

echo "=== Gate 4: DSGN++ inference on AWSIM frames ===" | tee "${LOG_FILE}"
echo "Split: ${SPLIT_FILE}" | tee -a "${LOG_FILE}"
echo "Config: ${CFG_FILE}" | tee -a "${LOG_FILE}"
echo "DSGN2 configs: ${DSGN2_CONFIGS_HOST}" | tee -a "${LOG_FILE}"
echo "Checkpoint: ${CKPT}" | tee -a "${LOG_FILE}"

# Mount only configs/ — full repo mount would hide image-built CUDA ops (iou3d_nms_cuda, etc.).
sudo docker run --rm "${DSGN2_GPU_FLAG[@]}" "${DSGN2_LIB_ENV[@]}" \
  -v "${ROOT}:${ROOT}" \
  -v "${DSGN2_CONFIGS_HOST}:${DSGN2}/configs" \
  -w "${DSGN2}" \
  -e HOME=/tmp \
  "${IMAGE}" \
  bash -lc "
    set -euo pipefail
    cd ${DSGN2}

    # Point OpenPCDet data root at our AWSIM KITTI layout.
    mkdir -p data
    ln -sfn ${DSGN2_DATA} data/kitti

    # Generate kitti_infos_val.pkl if not cached (label-free; AWSIM has no GT labels).
    if [[ ! -f ${DSGN2_DATA}/kitti_infos_val.pkl ]]; then
      echo 'Generating kitti_infos_val.pkl (first run only)...'
      python ${ROOT}/scripts/dsgn2_gen_val_infos.py ${DSGN2_DATA}/kitti_infos_val.pkl
    fi

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

# Collect 3D and 2D detection txt separately (never mix — Autoware needs 3D only).
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

n_3d_frames="$(count_nonempty_txt "${EVAL_3D}")"
n_2d_frames="$(count_nonempty_txt "${EVAL_2D}")"
echo "" | tee -a "${LOG_FILE}"
echo "=== Outputs: 3D -> ${DETECTIONS_DIR}  |  2D head -> ${DETECTIONS_2D_DIR} ===" | tee -a "${LOG_FILE}"
echo "  frames with non-empty 3D txt: ${n_3d_frames}" | tee -a "${LOG_FILE}"
echo "  frames with non-empty 2D txt: ${n_2d_frames}" | tee -a "${LOG_FILE}"
if [[ "${n_3d_frames}" -eq 0 ]]; then
  echo "  NOTE: no 3D detections — dsgn/detections/dsgn2/*.txt may be empty or absent." | tee -a "${LOG_FILE}"
  echo "        Check dsgn/detections/dsgn2/2d/ for auxiliary 2D-head output only." | tee -a "${LOG_FILE}"
fi

echo "" | tee -a "${LOG_FILE}"
echo "=== Detection counts vs baseline (3D in detections/dsgn2, not 2d/) ===" | tee -a "${LOG_FILE}"
BASELINE="${ROOT}/src/dsgn_offline/resource/awsim_output_offline"
for frame in 000010 000099 000105; do
  det_file="${DETECTIONS_DIR}/${frame}.txt"
  det2d_file="${DETECTIONS_2D_DIR}/${frame}.txt"
  base_file="${BASELINE}/${frame}.txt"
  det_n=0
  det2d_n=0
  base_n=0
  [[ -f "${det_file}" && -s "${det_file}" ]] && det_n=$(wc -l < "${det_file}")
  [[ -f "${det2d_file}" ]] && det2d_n=$(wc -l < "${det2d_file}")
  [[ -f "${base_file}" ]] && base_n=$(wc -l < "${base_file}")
  echo "  ${frame}: dsgn2_3d=${det_n}  dsgn2_2d=${det2d_n}  baseline=${base_n}" | tee -a "${LOG_FILE}"
done

echo ""
echo "3D detections: ${DETECTIONS_DIR}"
echo "2D detections: ${DETECTIONS_2D_DIR}"
echo "Full log: ${LOG_FILE}"
