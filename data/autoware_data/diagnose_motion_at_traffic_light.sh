#!/usr/bin/env bash
# Compact diagnose — writes JSON only, prints summary lines.
set -eo pipefail
set +u
source /opt/ros/humble/setup.bash
source /opt/autoware/setup.bash
set -u
unset CYCLONEDDS_URI
export ROS_DOMAIN_ID=26

RESULT="/home/aw/autoware_data/traffic_light_motion_diag.json"

python3 - <<'PY' "${RESULT}"
import json
import subprocess
import sys
from datetime import datetime, timezone

result_path = sys.argv[1]

def one(cmd: str) -> str:
    try:
        return subprocess.check_output(cmd, shell=True, text=True, timeout=6).strip()
    except Exception as exc:
        return f"error: {exc}"

tl = one("timeout 5 ros2 topic echo --once /perception/traffic_light_recognition/traffic_signals 2>/dev/null || true")
pf = one("timeout 5 ros2 topic echo --once /planning/planning_factors/traffic_light 2>/dev/null || true")
ctrl = one("timeout 5 ros2 topic echo --once /control/command/control_cmd 2>/dev/null || true")
vel = one("timeout 5 ros2 topic echo --once /vehicle/status/velocity_status 2>/dev/null || true")

green = tl.count("color: 3")
payload = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "traffic_signals_green_count": green,
    "traffic_light_stop_factor_active": "STOP" in pf or "stop" in pf.lower(),
    "control_velocity_line": next((ln for ln in ctrl.splitlines() if "velocity:" in ln), ""),
    "longitudinal_velocity_line": next((ln for ln in vel.splitlines() if "longitudinal_velocity:" in ln), ""),
}
with open(result_path, "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2)

print(f"GREEN signals in traffic_signals: {green}")
print(f"traffic_light stop factor active: {payload['traffic_light_stop_factor_active']}")
print(payload["control_velocity_line"] or "control_cmd: (no velocity line)")
print(payload["longitudinal_velocity_line"] or "velocity_status: (no line)")
print(f"full JSON: {result_path}")
PY
