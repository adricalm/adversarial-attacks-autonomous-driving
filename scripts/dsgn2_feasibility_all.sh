#!/usr/bin/env bash
# Run all DSGN++ L40S feasibility gates in order (host — needs sudo for docker).
#
# Usage: bash ~/summer26/scripts/dsgn2_feasibility_all.sh
set -euo pipefail

ROOT="${HOME}/summer26"

echo "=== Gate 1: Docker build ==="
bash "${ROOT}/scripts/dsgn2_build_docker.sh"

echo ""
echo "=== Gates 2-3: Smoke test (GPU + spconv) ==="
bash "${ROOT}/scripts/dsgn2_smoke_test.sh"

echo ""
echo "=== Gate 4: AWSIM inference ==="
bash "${ROOT}/scripts/dsgn2_run_inference.sh"

echo ""
echo "All gates complete. See logs/dsgn2_*.log for details."
