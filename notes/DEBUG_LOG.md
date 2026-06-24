# Debug log — Autoware + AWSIM stack

Running record of pitfalls discovered, diagnostics knowledge, and the changelog.
Not onboarding docs — see the main README for that.

---

## Debugging checklist

When something breaks, check in this order:

1. **ROS domain:** `echo $ROS_DOMAIN_ID` → `26` everywhere; `CYCLONEDDS_URI` unset.
2. **Single publishers:** `ros2 topic info -v /clock` — expect 1 publisher (AWSIM).
3. **Localization:** `ros2 topic echo /api/localization/initialization_state --once`
4. **TF:** `ros2 run tf2_ros tf2_echo map base_link` — valid transform after init.
5. **Perception:** `ros2 topic echo --once /system/component_state_monitor/component/launch/perception` → `available: true`.
6. **Routing:** `/api/routing/state` and mission planner logs if route is empty.
7. **Service types:** `ros2 interface show <srv_type>` before crafting service calls.

---

## Known pitfalls

| Pitfall | What to do instead |
|---------|-------------------|
| Fake `/clock` or vehicle status nodes | Remove them; AWSIM publishes these |
| z=0 for map poses | Use `ele` from OSM (~40.9 in Nishi-Shinjuku working area) |
| `launch_rviz_adaptors:=false` | RViz rough goals have zero subscribers; use `:=true` in canonical launch |
| RViz Fixed Frame `map` on MGRS map | Use **`viewer`**; clicks in `map` frame land far from the car |
| Default RViz **2D Goal Pose** tool | Use Autoware **2D Rough Goal Pose** → `/rviz/routing/rough_goal` |
| Relaunch without NDT bind-mount | MRM emergency, stuck at state 3, engage fails — always use canonical launch |
| Guessing route goals in RViz | Snap to lanes via rough goal + top-down view, or use `find_route_candidates.py` |
| `ros2 service call` for localization | Use `apply_route_from_osm.py` (bounded sequence + nested response) |
| Successor lanelets reversed in graph walk | Successor = lanelet whose **start** matches current lanelet's **end** |
| Agent retries `sudo docker` in a loop | It cannot authenticate; give adria the exact host command instead |
| `sudo` password typed at shell prompt | Password must go into the `sudo` prompt, not bash |
| Assuming container name `autoware_full_test` | Auto-detect via `pgrep -f e2e_simulator.launch` or `sudo docker ps` |
| Editing the diagnostic graph with `edits: { type: remove }` | **Crashes `aggregator_node` (SIGSEGV, exit -11)** — `command_mode_mappings` in `default.param.yaml` still link `/autoware/modes/*` to the removed node. No `/system/command_mode/availability` → `is_autonomous_mode_available` permanently false → MRM `emergency_stop`. Force the failing leaf to a constant `ok` instead of removing it. |
| Treating `control_command_gate` STALE as the engage blocker | It's an `or` with `vehicle_cmd_gate` (`command_gate` in `control.yaml`, `emergency_stop_operation` in `system.yaml`). STALE there is harmless if `vehicle_cmd_gate` is OK. `service_log_checker` ERROR is also unlinked from `/autoware/system`. Find the real non-OK node under `/autoware/modes/autonomous`. |
| Stale `ros2` CLI daemon after restarting the Autoware container | New `docker exec` shells may show ~24 topics + XMLRPC tracebacks on `ros2 node list`. Run `ros2 daemon stop && ros2 daemon start`, then wait for the full stack (~2–3 min) before the first query. |
| NDT scan score marginally below threshold while driving → MRM latch | Nishi-Shinjuku map yields ~2.2–2.4 likelihood. Default threshold `2.3` trips during motion, stalls NDT pose publishing (`skipping_publish_num` exhausted), delays EKF, flips `modes/autonomous` ERROR → `hazard emergency: true` → `emergency_stop`. Fix: `data/autoware_data/ndt_scan_matcher.param.yaml` sets threshold `1.8` and `skipping_publish_num: 20`. Always bind-mount this file (it's in the canonical launch command). |
| `Routing \| Arrived` with `Motion \| Stopped` after a short drive | Not a bug — the ego completed its route. `is_autonomous_mode_available: false` at this state is normal (no active route). Set a new, longer route with `recover_after_awsim_restart.sh` or `find_route_candidates.py`. |
| Safe diag graph override for a genuinely harmless STALE leaf | Use `type: const` + `level: OK` in a host-side YAML override, never `type: remove`. `type: remove` deletes the node while `command_mode_mappings` still links to it → aggregator SIGSEGV. The node must stay in the graph; only its value needs overriding. |

---

## Diagnostics → operation-mode availability chain (engage gating)

Why `change_to_autonomous` can fail with *"The target mode is not available. Please check the diagnostics."*:

```text
diagnostic_graph_aggregator (aggregator_node)
  → /diagnostics_graph/{struct,status}        (read by `dump_node`)
  → /system/command_mode/availability         (via command_mode_mappings)
      → converter_node
      → /system/operation_mode/availability    (autonomous/stop/local/... booleans)
          → operation_mode_transition_manager → /api/operation_mode/state.is_autonomous_mode_available
          → mrm_handler (falls back to emergency_stop if no mode available)
```

- If `aggregator_node` dies, **all** of the above go silent (0 publishers on availability), engage is impossible, and MRM latches `emergency_stop`. Check it: `pgrep -af aggregator_node`; logs: `sudo docker logs autoware_full_test 2>&1 | grep aggregator_node`.
- Read real availability (transient-local): `ros2 topic echo --once /system/operation_mode/availability --qos-durability transient_local --qos-reliability reliable`.
- Graph config: `config/system/diagnostics/autoware-awsim.yaml` → includes `autoware-main.yaml` → `control.yaml`, `system.yaml`, `localization.yaml`, etc. (inside the Autoware image, not on a mounted volume).

### Debugging `is_autonomous_mode_available: false`

1. **Aggregator alive?** `pgrep -af aggregator_node` — if missing, a bad diag bind-mount crashed it. Clean restart.
2. **Real availability:** `ros2 topic echo --once /system/operation_mode/availability --qos-durability transient_local --qos-reliability reliable`
3. **Graph dump:** `timeout 8 ros2 run autoware_diagnostic_graph_utils dump_node > /tmp/g.txt && grep -E "STALE|WARN|ERROR" /tmp/g.txt | sort -u`
4. **Authoritative hazard diagnosis** (names the actual blocking diagnostic): `ros2 topic echo --once /system/emergency/hazard_status` → look at `diag_single_point_fault` and `diag_latent_fault` arrays.
5. **NDT score** (if localization is suspect): `ros2 topic echo --once /diagnostics | grep -A5 "scan_matching_status"` — if "Score is below the threshold", check the NDT bind-mount.

Many ERRORs at **state 2 (WAITING_FOR_ROUTE)** are expected — routing/trajectory/control nodes report no-input until a route is set. Set a route first, then check again.

---

## Localization details

RViz "Initialize with GNSS" did **not** work (no GNSS publisher on `/sensing/gnss/pose_with_covariance`).

**Working method:** service `/localization/initialize`, type `autoware_localization_msgs/srv/InitializeLocalization`, method `DIRECT = 1`.

**Important interface detail** (do not get this wrong):

- `pose_with_covariance` is `geometry_msgs/PoseWithCovarianceStamped[<=1]` — a **bounded sequence** (list of 0–1 elements), not a bare struct.
- Response success is `response.status.success`, not `response.success`.

Prefer `scripts/apply_route_from_osm.py` over manual `ros2 service call` YAML for localization.

---

## Changelog

| Date | Milestone |
|------|-----------|
| 2026-06-16 | GUI / xrdp / RViz / AWSIM visibility fixed |
| 2026-06-16 | AWSIM moved into Docker for ROS 2 compatibility |
| 2026-06-16 | Removed duplicate fake clock/status nodes |
| 2026-06-17 | Localization via `/localization/initialize` DIRECT |
| 2026-06-17 | Routing via OSM lanelet parsing + AD API; Autoware reaches state 4 |
| 2026-06-17 | **End-to-end baseline achieved**: AWSIM + Autoware engage + drive (state 5, velocity > 0, traffic lights GREEN via bridge) |
| 2026-06-18 | Fixed MRM emergency_stop latch: removed harmful `type: remove` diag bind-mount (crashes aggregator); tuned NDT threshold 2.3→1.8 + skipping_publish_num 5→20 for Nishi-Shinjuku map; ego drives through traffic lights cleanly with no MRM activation |
| 2026-06-23 | RViz goal routing: `launch_rviz_adaptors:=true`; Fixed Frame `viewer` for MGRS map; documented NDT mount vs adaptors as separate required launch pieces |
| 2026-06-24 | Stable RViz-driven baseline: localize + set route + Auto from RViz all work. Cleaned up trial-and-error scripts. Starting DSGN pipeline. |
