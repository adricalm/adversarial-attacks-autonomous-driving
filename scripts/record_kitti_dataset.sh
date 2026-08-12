#!/usr/bin/env bash
# Record a KITTI-layout stereo dataset from modded AWSIM + Autoware.
#
# Runs the recorder in its OWN short-lived container rather than inside
# autoware_full_test. That container has no dataset mount, and restarting it to
# add one would take down a working Autoware stack for no good reason. A
# separate container on the same host network / ROS_DOMAIN_ID sees the same ROS
# graph and keeps the recorder's CPU load off the stack.
#
# Output goes to dsgn/datasets/<run_id>/ (already gitignored) in the layout
# optimize_patch.py expects: image_2/, image_3/, velodyne/, pose/path.txt.
# calib/ is added afterwards via prepare_recording_datasets.py (fixed rig copy).
#
# Usage:
#   scripts/record_kitti_dataset.sh adria/awsim_stereo_001
#   scripts/record_kitti_dataset.sh adria/smoke --max-frames 20
#   scripts/record_kitti_dataset.sh adria/train_only --no-velodyne
set -euo pipefail

RUN_ID="${1:-run_$(date +%Y%m%d_%H%M%S)}"; shift || true

ROOT="$HOME/summer26"
DATASETS="$ROOT/dsgn/datasets"
OUT_HOST="$DATASETS/$RUN_ID"
DOMAIN="${ROS_DOMAIN_ID:-26}"
IMAGE="ghcr.io/autowarefoundation/autoware:universe-cuda-humble"
NAME="${RECORDER_CONTAINER:-awsim_recorder}"

if [[ ! -d "$ROOT/src/awsim_to_kitti" ]]; then
  echo "ERROR: $ROOT/src/awsim_to_kitti not found" >&2; exit 1
fi
if [[ -e "$OUT_HOST" ]]; then
  echo "ERROR: $OUT_HOST already exists (pick a new run id)" >&2; exit 1
fi

AVAIL_GB=$(df -BG --output=avail "$DATASETS" | tail -1 | tr -dc '0-9')
if (( AVAIL_GB < 25 )); then
  echo "WARNING: only ${AVAIL_GB} GB free. Stereo PNG + lidar costs ~6.2 MB/frame" >&2
  echo "         (~22 GB per 10 min at 10 Hz). Ctrl+C now if that is too tight." >&2
  sleep 5
fi

mkdir -p "$OUT_HOST"

echo "run id    : $RUN_ID"
echo "output    : $OUT_HOST"
echo "domain    : $DOMAIN"
echo "free disk : ${AVAIL_GB} GB"
echo

docker rm -f "$NAME" >/dev/null 2>&1 || true

# Interactive only when there is a terminal, so the script also works from a
# pipeline or a non-tty agent shell.
TTY_FLAGS=()
[[ -t 0 ]] && TTY_FLAGS=(-it)

exec docker run --rm "${TTY_FLAGS[@]}" \
  --name "$NAME" \
  --network host \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e ROS_DOMAIN_ID="$DOMAIN" \
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  -v "$ROOT/src:/recorder:ro" \
  -v "$OUT_HOST:/out:rw" \
  -v "$ROOT/data/autoware_data/cyclonedds_bigbuf.xml:/dds/cyclonedds.xml:ro" \
  --entrypoint /bin/bash \
  "$IMAGE" \
  -lc "source /opt/ros/humble/setup.bash
       source /opt/autoware/setup.bash
       # Use the same DDS config as the stack: lo-only and a 10 MB socket
       # receive buffer. Running on CycloneDDS defaults (2 MB) makes the
       # recorder's image/lidar traffic overflow buffers domain-wide and
       # stalls Autoware. Needs net.core.rmem_max >= 10 MB.
       export CYCLONEDDS_URI=file:///dds/cyclonedds.xml
       export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
       export ROS_DOMAIN_ID=$DOMAIN
       exec python3 /recorder/awsim_to_kitti/awsim_to_kitti_recorder.py \
         --out /out $*"
