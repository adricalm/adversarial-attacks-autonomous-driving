#!/usr/bin/env bash
# Build a route JSON from two captured poses (start + goal).
#
# Usage (inside Docker):
#   1. Park at START, run:  bash /home/aw/scripts/capture_pose.sh --json start > /tmp/start.json
#   2. Park at GOAL,  run:  bash /home/aw/scripts/capture_pose.sh --json goal  > /tmp/goal.json
#   3. Combine:             bash /home/aw/scripts/make_route_json.sh /tmp/start.json /tmp/goal.json /tmp/my_route.json
#
# Or in one shot after moving the car twice:
#   bash /home/aw/scripts/make_route_json.sh --interactive /tmp/my_route.json
set -euo pipefail

if [[ "${1:-}" == "--interactive" ]]; then
  OUT="${2:-/tmp/my_route.json}"
  START_FRAG="$(mktemp)"
  GOAL_FRAG="$(mktemp)"
  trap 'rm -f "${START_FRAG}" "${GOAL_FRAG}"' EXIT

  echo "Park the car at START pose, press Enter..."
  read -r
  bash /home/aw/scripts/capture_pose.sh --json start > "${START_FRAG}"
  echo "Saved start."

  echo "Park the car at GOAL pose, press Enter..."
  read -r
  bash /home/aw/scripts/capture_pose.sh --json goal > "${GOAL_FRAG}"
  echo "Saved goal."

  {
    echo "{"
    cat "${START_FRAG}"
    echo ","
    cat "${GOAL_FRAG}"
    echo "}"
  } > "${OUT}"
  echo "Wrote ${OUT}"
  exit 0
fi

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <start_fragment.json> <goal_fragment.json> <output.json>" >&2
  echo "   or: $0 --interactive [output.json]" >&2
  exit 1
fi

START_FRAG="$1"
GOAL_FRAG="$2"
OUT="$3"

{
  echo "{"
  cat "${START_FRAG}"
  echo ","
  cat "${GOAL_FRAG}"
  echo "}"
} > "${OUT}"

echo "Wrote ${OUT}"
