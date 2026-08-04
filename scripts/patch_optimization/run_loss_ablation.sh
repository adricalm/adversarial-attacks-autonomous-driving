#!/usr/bin/env bash
# 2x2 loss x coverage ablation on the per-frame capacity harness.
#
# Holds fixed: frames (10, same selection), steps (300), lr (5e-2), z init
# (zeros), patch shape (rear-face aspect ratio), placement (face centre),
# parameterization (native rendered resolution).
# Varies: --loss {logit,prob} x --area-frac {0.23,0.50}.
#
# area_frac 0.23 is the area equivalent of the original square patch
# (size = 0.50 * min(face_w, face_h) -> median 0.2374 of face area).
#
# Idempotent: an arm whose summary CSV already exists is skipped, so re-running
# only fills the gaps. The GPU is shared and one DSGN forward+backward needs
# ~13 GB, so each arm waits for headroom and retries on OOM.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PY=external/DSGN_custom/.venv/bin/python
OUT=dsgn/datasets/adria/2.training_patch_optimization/ceiling/ablation
mkdir -p "$OUT"

STEPS=${STEPS:-300}
FRAMES=${FRAMES:-10}
NEED_MIB=${NEED_MIB:-12000}
SKIP_NMS=${SKIP_NMS:-1}
MAX_WAIT=${MAX_WAIT:-5400}
ATTEMPTS=${ATTEMPTS:-3}

# Fragmentation is what pushes us over the edge when the GPU is already busy.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

gpu_free_mib() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1
}

wait_for_gpu() {
  local waited=0 free
  while :; do
    free=$(gpu_free_mib)
    if [ "${free:-0}" -ge "$NEED_MIB" ]; then
      echo "GPU ok: ${free} MiB free (need ${NEED_MIB})"
      return 0
    fi
    if [ "$waited" -ge "$MAX_WAIT" ]; then
      echo "WARN: only ${free} MiB free after ${waited}s; trying anyway"
      return 0
    fi
    echo "waiting for GPU: ${free} MiB free, need ${NEED_MIB} MiB (waited ${waited}s)"
    sleep 60
    waited=$((waited + 60))
  done
}

run_arm() {
  local loss=$1 area=$2 log=$3
  local extra=()
  [ "$SKIP_NMS" = "1" ] && extra+=(--skip-nms)
  "$PY" -u scripts/patch_optimization/ceiling_test.py overfit \
    --n-frames "$FRAMES" \
    --steps "$STEPS" \
    --loss "$loss" \
    --area-frac "$area" \
    --log-every 50 \
    "${extra[@]}" \
    2>&1 | grep --line-buffered -v 'UserWarning\|upsample(' | tee -a "$log"
}

for spec in logit:0.23:023 prob:0.23:023 logit:0.50:050 prob:0.50:050; do
  loss=${spec%%:*}
  rest=${spec#*:}
  area=${rest%%:*}
  pct=${rest#*:}
  tag="${loss}_${pct}"
  summary="$OUT/summary_${tag}.csv"
  log="$OUT/log_${tag}.log"

  if [ -s "$summary" ]; then
    echo "=== arm ${tag}: already done ($summary), skipping ==="
    continue
  fi

  for attempt in $(seq 1 "$ATTEMPTS"); do
    echo "=== arm ${tag}: loss=${loss} area_frac=${area} (attempt ${attempt}/${ATTEMPTS}) ==="
    wait_for_gpu
    run_arm "$loss" "$area" "$log"
    if [ -s "$summary" ]; then
      echo "=== arm ${tag} done -> $summary ==="
      break
    fi
    echo "=== arm ${tag} FAILED on attempt ${attempt} (no summary written) ==="
    sleep 120
  done
done

echo "ALL ARMS COMPLETE"
for tag in logit_023 prob_023 logit_050 prob_050; do
  f="$OUT/summary_${tag}.csv"
  if [ -s "$f" ]; then echo "  OK      $tag"; else echo "  MISSING $tag"; fi
done
