#!/usr/bin/env bash
# Verify bridge + motion inside Autoware Docker container (no sudo).
set -eo pipefail

# ROS setup.bash references optional env vars (e.g. AMENT_TRACE_SETUP_FILES).
set +u
source /opt/ros/humble/setup.bash
source /opt/autoware/setup.bash
set -u
unset CYCLONEDDS_URI
export ROS_DOMAIN_ID=26

RESULT="/home/aw/autoware_data/traffic_light_verify_result.json"

echo "=== bridge process ==="
pgrep -af traffic_light_green_bridge.py || echo "bridge not running"

echo "=== topic types ==="
ros2 topic type /perception/traffic_light_recognition/external/traffic_signals
ros2 topic type /perception/traffic_light_recognition/traffic_signals

echo "=== external topic info ==="
ros2 topic info /perception/traffic_light_recognition/external/traffic_signals -v

echo "=== external echo (once) ==="
EXTERNAL="$(timeout 5 ros2 topic echo --once /perception/traffic_light_recognition/external/traffic_signals 2>/dev/null || true)"

echo "=== merged traffic_signals echo (once) ==="
MERGED="$(timeout 5 ros2 topic echo --once /perception/traffic_light_recognition/traffic_signals 2>/dev/null || true)"

echo "=== autoware state ==="
STATE="$(timeout 5 ros2 topic echo --once /autoware/state 2>/dev/null || true)"

echo "=== control_cmd ==="
CONTROL="$(timeout 5 ros2 topic echo --once /control/command/control_cmd 2>/dev/null || true)"

echo "=== velocity_status ==="
VELOCITY="$(timeout 5 ros2 topic echo --once /vehicle/status/velocity_status 2>/dev/null || true)"

python3 - <<'PY' "${EXTERNAL}" "${MERGED}" "${STATE}" "${CONTROL}" "${VELOCITY}" "${RESULT}"
import json
import sys
from datetime import datetime, timezone

external, merged, state, control, velocity, result_path = sys.argv[1:7]

def has_green_groups(text: str) -> bool:
    if "traffic_light_groups:" not in text:
        return False
    if "traffic_light_groups: []" in text:
        return False
    return "color: 3" in text or "GREEN" in text

payload = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "external_non_empty": has_green_groups(external),
    "merged_non_empty": has_green_groups(merged),
    "external_sample": external[:2000],
    "merged_sample": merged[:2000],
    "autoware_state_sample": state[:500],
    "control_cmd_sample": control[:1000],
    "velocity_status_sample": velocity[:500],
}
with open(result_path, "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2)
print(json.dumps(payload, indent=2))
PY

echo "Wrote ${RESULT}"
