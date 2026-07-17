#!/usr/bin/env bash

set -euo pipefail

# Usage:
#   bash scripts/visualize_dsgn_detections.sh new_config
#   bash scripts/visualize_dsgn_detections.sh finetune_60
#   FRAMES_TO_VISUALIZE=000010,000099 bash scripts/visualize_dsgn_detections.sh new_config
#
# Writes PNGs to: dsgn/detections/adria/<folder>/detection_previews/

ROOT="${HOME}/summer26"
ARKA_DS="${ROOT}/dsgn/datasets/arka/dsgn_awsim"
DATA_PATH="${DATA_PATH:-${ARKA_DS}/testing_offline}"

IMAGES_PATH="${IMAGES_PATH:-${DATA_PATH}/image_2}"
CALIB_PATH="${CALIB_PATH:-${DATA_PATH}/calib}"

DETECTIONS_DIR="${DETECTIONS_DIR:-${ROOT}/dsgn/detections/adria}"
DETECTION_FOLDER="${1:-${DETECTION_FOLDER:-finetune_60}}"
DETECTIONS_PATH="${DETECTIONS_DIR}/${DETECTION_FOLDER}"

FRAMES_TO_VISUALIZE="${FRAMES_TO_VISUALIZE:-000010,000099,000105}"
LABEL="${LABEL:-${DETECTION_FOLDER}}"
OUTPUT_PATH="${OUTPUT_PATH:-${DETECTIONS_PATH}/detection_previews}"

if [[ ! -d "${DETECTIONS_PATH}" ]]; then
  echo "error: detections folder not found: ${DETECTIONS_PATH}" >&2
  exit 1
fi

cd "${ROOT}/scripts"

python visualize_dsgn_detections.py \
  --images "${IMAGES_PATH}" \
  --calib "${CALIB_PATH}" \
  --detections "${DETECTIONS_PATH}" \
  --frames "${FRAMES_TO_VISUALIZE}" \
  --label "${LABEL}" \
  --output "${OUTPUT_PATH}"

