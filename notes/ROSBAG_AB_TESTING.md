# Rosbag A/B testing (dsgn_offline ON vs OFF)

Record lightweight bags to compare baseline driving vs fake detections from `dsgn_offline`.

**Bag storage (your user, no sudo to delete):**

| Where | Path |
|-------|------|
| Host | `~/summer26/data/bags/<run_id>/` |
| Inside Docker | `/home/aw/bags/<run_id>/` |

Requires this **extra Docker mount** (add to canonical `docker run`):

```bash
-v "$HOME/summer26/data/bags:/home/aw/bags" \
```

If the container was started without it, restart Autoware with the mount before recording.

---

## Phase 0 — One-time prep

### Host — create bags dir (once)

```bash
mkdir -p ~/summer26/data/bags
```

### Host — ensure container has bags mount

Check:

```bash
sudo docker inspect autoware_full_test --format '{{range .Mounts}}{{.Source}} -> {{.Destination}}{{"\n"}}{{end}}' | grep bags
```

If missing, restart container with the mount (see [README](../README.md) + bags line above).

### Inside Docker — ROS env (every shell)

```bash
sudo docker exec -it autoware_full_test bash
```

```bash
source /opt/ros/humble/setup.bash
source /opt/autoware/setup.bash
unset CYCLONEDDS_URI
export ROS_DOMAIN_ID=26
```

For Run B (`dsgn_offline`), also:

```bash
source /home/aw/ros2_ws/install/setup.bash
```

### Inside Docker — build overlay (once)

```bash
bash /home/aw/scripts/dsgn_offline_build.sh
source /home/aw/ros2_ws/install/setup.bash
```

### Preflight

```bash
timeout 3 ros2 topic echo --once /clock
bash /home/aw/autoware_data/verify_stack_ready.sh
```

AWSIM must be **playing**.

---

## Run A — Baseline (`dsgn_offline` OFF)

You need **2 terminals**: one recording, one driving.

### A1. Confirm dsgn is off (Docker)

```bash
pgrep -af dsgn_offline || echo "OK: dsgn_offline not running"
pkill -f "ros2 run dsgn_offline dsgn_offline" 2>/dev/null || true
```

### A2. Stop if still driving (Docker)

```bash
timeout 10 ros2 service call /api/operation_mode/change_to_stop \
  autoware_adapi_v1_msgs/srv/ChangeOperationMode "{}" || true
sleep 3
```

### A3. Start recording (host — files owned by `adria`)

```bash
bash ~/summer26/scripts/record_experiment_bag_host.sh run_a_baseline_001
```

Leave running. Stop with **Ctrl+C** when the drive is done.

### A4. Init + engage (Docker — second shell)

```bash
bash /home/aw/scripts/drive_route_and_engage.sh \
  /home/aw/autoware_data/route_dsgn_ab.json
```

### A5. Stop recording + metadata

In the record terminal: **Ctrl+C**

```bash
bash ~/summer26/scripts/write_bag_metadata.sh run_a_baseline_001 baseline
```

### A6. Verify (host)

```bash
ls -lh ~/summer26/data/bags/run_a_baseline_001/
ros2 bag info ~/summer26/data/bags/run_a_baseline_001   # if ros2 on host; else inside Docker:
```

Inside Docker:

```bash
ros2 bag info /home/aw/bags/run_a_baseline_001
```

---

## Reset between A and B

Do **not** restart AWSIM.

### Docker

```bash
timeout 10 ros2 service call /api/operation_mode/change_to_stop \
  autoware_adapi_v1_msgs/srv/ChangeOperationMode "{}" || true
sleep 5
pgrep -af dsgn_offline || echo "OK"
```

Wait until velocity ≈ 0:

```bash
for i in $(seq 1 20); do
  V=$(timeout 2 ros2 topic echo --once /vehicle/status/velocity_status 2>/dev/null \
    | grep longitudinal_velocity | awk '{print $2}')
  echo "attempt $i/20 velocity=${V:-unknown}"
  python3 -c "import sys; sys.exit(0 if abs(float('${V:-999}')) < 0.05 else 1)" 2>/dev/null && break
  sleep 1
done
```

---

## Run B — With `dsgn_offline` ON

**3 terminals**: dsgn node, recording, driving.

### B1. Start dsgn_offline (Docker — keep running)

```bash
source /home/aw/ros2_ws/install/setup.bash
bash /home/aw/scripts/dsgn_offline_run.sh
```

Verify in another shell:

```bash
timeout 5 ros2 topic hz /perception/object_recognition/detection/centerpoint/validation/objects
timeout 5 ros2 topic echo --once /perception/object_recognition/detection/centerpoint/validation/objects
```

### B2. Start recording (host)

```bash
bash ~/summer26/scripts/record_experiment_bag_host.sh run_b_dsgn_on_001
```

### B3. Init + engage (Docker)

```bash
bash /home/aw/scripts/drive_route_and_engage.sh \
  /home/aw/autoware_data/route_dsgn_ab.json
```

Monitor:

```bash
bash /home/aw/autoware_data/dsgn_chain_check.sh
```

### B4. Stop recording, stop dsgn, metadata

Record terminal
```text
Ctrl+C
```

dsgn shell
```text
Ctrl+C
```

```bash
bash ~/summer26/scripts/write_bag_metadata.sh run_b_dsgn_on_001 dsgn_on
```

---

## Repeat for reproducibility

Run 3× baseline + 3× dsgn_on with different ids:

```text
run_a_baseline_001 … 003
run_b_dsgn_on_001 … 003
```

Between repeats: reset section above, then record → `drive_route_and_engage.sh`.

---

## Manage bags (host — no sudo)

```bash
ls -lh ~/summer26/data/bags/
du -sh ~/summer26/data/bags/*
rm -rf ~/summer26/data/bags/run_a_baseline_001   # example delete
```

---

## Attack variant (optional Run C)

```bash
# Docker — dsgn shell
DETECTION_FOLDER=/home/aw/ros2_ws/src/dsgn_offline/resource/awsim_output_attack_ghost \
  bash /home/aw/scripts/dsgn_offline_run.sh
```

```bash
# host
bash ~/summer26/scripts/record_experiment_bag_host.sh run_c_attack_001
bash ~/summer26/scripts/write_bag_metadata.sh run_c_attack_001 attack
```

---

## Topics recorded

See `scripts/record_experiment_bag.sh` — perception chain, planning outputs, ego state, `/clock`.

---

## Compare two bags (baseline vs dsgn ON)

After recording both runs:

```bash
# host
bash ~/summer26/scripts/compare_experiment_bags.sh run_a_baseline_001 run_b_dsgn_offline_001
```

Prints a table: speed, distance, object counts, obstacle-stop events, etc.

---

## Who receives `/perception/object_recognition/objects`?

**Rosbags do not contain subscriber info** — only messages. To see who subscribes, query the **live** graph while Autoware is running (inside Docker):

```bash
bash /home/aw/scripts/audit_object_topic_subscribers.sh
```

Or manually:

```bash
ros2 topic info -v /perception/object_recognition/objects
```

Main consumers: `map_based_prediction` (upstream), then `behavior_path_planner`, `behavior_velocity_planner`, `motion_velocity_planner` (obstacle_stop), `autonomous_emergency_braking`. See `notes/PERCEPTION_PIPELINE.md` §5.3.

---

## Related

- [`scripts/compare_experiment_bags.sh`](../scripts/compare_experiment_bags.sh) — diff baseline vs dsgn-on bags
- [`scripts/audit_object_topic_subscribers.sh`](../scripts/audit_object_topic_subscribers.sh) — live subscriber graph
- [`DSGN_OFFLINE_RUNBOOK.md`](DSGN_OFFLINE_RUNBOOK.md) — build/run `dsgn_offline`
- [`scripts/drive_route_and_engage.sh`](../scripts/drive_route_and_engage.sh) — fixed start/goal from JSON
- [`PERCEPTION_PIPELINE.md`](PERCEPTION_PIPELINE.md) — full perception → planning chain
