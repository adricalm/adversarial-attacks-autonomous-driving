#!/usr/bin/env python3
"""Log why the ego stops, at 5 Hz, for correlating with recorder start/stop.

Run inside the Autoware container while driving, then start the recorder in
another terminal. Every line is wall-clock stamped so the moment the recorder
attaches lines up with whichever signal changes.

    python3 /home/aw/scripts/helpers/monitor_stop_cause.py
    python3 /home/aw/scripts/helpers/monitor_stop_cause.py --log /home/aw/logs/stop.log

Columns:
  vel      measured longitudinal velocity (m/s)
  cmd_v    commanded velocity from the controller (m/s)
  cmd_a    commanded acceleration (m/s^2)
  traj0    velocity of first planning trajectory point (m/s)
  mode     operation mode (2 = AUTONOMOUS)
  mrm      MRM state / behavior (1/1 = NORMAL / NONE)
  aeb      number of AEB metrics reported (non-zero => AEB active)
  avail    operation_mode/availability: autonomous flag (stop/local/remote/
           emergency_stop/comfortable_stop/pull_over logged alongside it)
  ages     seconds since each topic was last received; a stalled feed shows here

Two diagnostic sources are logged, and they are NOT the same thing:

  DIAG-*   raw /diagnostics. Each publishing node emits ONLY ITS OWN status
           in each message (there is no single "everything" snapshot on this
           topic), so this is transition-tracked per diagnostic *name*
           across messages, not per message. Useful for eyeballing which
           node is complaining, noisy by nature (a node's checks can flap
           independently of anything downstream).

  GRAPH-*  /diagnostics_graph/{struct,status}, i.e. diagnostic_graph_aggregator
           -- the actual tree that feeds /system/command_mode/availability
           (and therefore /system/operation_mode/availability + mrm_handler).
           This is the authoritative "why did autonomous become unavailable"
           source. Each graph node reports level (current), input_level (raw
           upstream level before latching) and latch_level. When latch_level
           stays nonzero after input_level has gone back to 0, the condition
           has RECOVERED but the graph is still latched -- that alone
           explains a permanent stop that outlives whatever tripped it.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy


def _try_import(module, name):
    try:
        return getattr(__import__(module, fromlist=[name]), name)
    except Exception:
        return None


VelocityReport = _try_import("autoware_vehicle_msgs.msg", "VelocityReport")
Control = _try_import("autoware_control_msgs.msg", "Control")
Trajectory = _try_import("autoware_planning_msgs.msg", "Trajectory")
OperationModeState = _try_import("autoware_adapi_v1_msgs.msg", "OperationModeState")
MrmState = _try_import("autoware_adapi_v1_msgs.msg", "MrmState")
MetricArray = _try_import("tier4_metric_msgs.msg", "MetricArray")
if MetricArray is None:
    MetricArray = _try_import("diagnostic_msgs.msg", "DiagnosticArray")
DiagnosticArray = _try_import("diagnostic_msgs.msg", "DiagnosticArray")
OperationModeAvailability = _try_import(
    "tier4_system_msgs.msg", "OperationModeAvailability"
)
DiagGraphStruct = _try_import("tier4_system_msgs.msg", "DiagGraphStruct")
DiagGraphStatus = _try_import("tier4_system_msgs.msg", "DiagGraphStatus")
PoseWithCovarianceStamped = _try_import(
    "geometry_msgs.msg", "PoseWithCovarianceStamped"
)


class Monitor(Node):
    def __init__(self, log_path=None):
        super().__init__("monitor_stop_cause")
        self.state = {}
        self.stamps = {}
        self.log = open(log_path, "a", buffering=1) if log_path else None

        # Autoware status topics are mostly transient-local / best-effort; a
        # plain default subscription silently receives nothing from some.
        volatile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        transient = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        def sub(msg_type, topic, key, extract, qos=volatile):
            if msg_type is None:
                return

            def cb(msg, key=key, extract=extract):
                try:
                    self.state[key] = extract(msg)
                except Exception:
                    self.state[key] = "err"
                self.stamps[key] = time.time()

            self.create_subscription(msg_type, topic, cb, qos)

        sub(VelocityReport, "/vehicle/status/velocity_status", "vel",
            lambda m: m.longitudinal_velocity)
        sub(Control, "/control/command/control_cmd", "cmd",
            lambda m: (m.longitudinal.velocity, m.longitudinal.acceleration))
        sub(Trajectory, "/planning/scenario_planning/trajectory", "traj",
            lambda m: m.points[0].longitudinal_velocity_mps if m.points else 0.0)
        sub(OperationModeState, "/api/operation_mode/state", "mode",
            lambda m: m.mode, transient)
        sub(MrmState, "/system/fail_safe/mrm_state", "mrm",
            lambda m: (m.state, m.behavior))
        sub(MetricArray, "/control/autonomous_emergency_braking/metrics", "aeb",
            lambda m: len(getattr(m, "metric_array", getattr(m, "status", []))))
        # MRM fires when this drops autonomous; GRAPH-* below names the leaf
        # that actually caused it.
        sub(OperationModeAvailability, "/system/operation_mode/availability", "avail",
            lambda m: (m.autonomous, m.stop, m.local, m.remote, m.emergency_stop,
                       m.comfortable_stop, m.pull_over))
        # Map position, to tell a stop that always happens at the same place on
        # the route (map / route / NPC) from one that tracks elapsed time or
        # system load.
        sub(PoseWithCovarianceStamped, "/localization/pose_with_covariance", "pos",
            lambda m: (m.pose.pose.position.x, m.pose.pose.position.y))

        # --- raw /diagnostics: transition-track PER NAME, not per message ---
        # Each diagnostic_updater publishes only its own statuses on this
        # topic, so a naive "diff against the previous message" treats every
        # name absent from the current message as "cleared", then "started"
        # again the moment that node republishes. That produces a FAIL
        # immediately followed by OK for nearly everything, even names that
        # never actually recovered. Track state per name instead.
        self._diag_level = {}
        if DiagnosticArray is not None:
            self.create_subscription(
                DiagnosticArray, "/diagnostics", self._on_diag, 10
            )

        # --- diagnostic_graph_aggregator: the real source of truth ---
        self._graph_paths = None    # list[str], node index -> path
        self._graph_diags = None    # list[str], leaf index -> name
        self._graph_node_state = {}  # path -> (level, latch_level)
        self._graph_diag_state = {}  # name -> level
        if DiagGraphStruct is not None:
            self.create_subscription(
                DiagGraphStruct, "/diagnostics_graph/struct", self._on_graph_struct,
                transient,
            )
        if DiagGraphStatus is not None:
            self.create_subscription(
                DiagGraphStatus, "/diagnostics_graph/status", self._on_graph_status,
                volatile,
            )

        self.create_timer(0.2, self._tick)

    @staticmethod
    def _level(status):
        """DiagnosticStatus.level (and graph byte levels) arrive as a single
        byte in some rclpy/DDS combos, not an int."""
        lvl = status
        if isinstance(lvl, (bytes, bytearray)):
            return int.from_bytes(lvl, "big")
        return int(lvl)

    def _stamp(self):
        return datetime.now().strftime("%H:%M:%S.%f")[:-3]

    def _emit(self, line):
        print(line, flush=True)
        if self.log:
            self.log.write(line + "\n")

    def _on_diag(self, msg):
        stamp = self._stamp()
        for s in msg.status:
            lvl = self._level(s.level)
            prev = self._diag_level.get(s.name, 0)
            if lvl != 0 and prev == 0:
                self._emit(f"{stamp}   DIAG-FAIL  level={lvl}  {s.name}")
            elif lvl == 0 and prev != 0:
                self._emit(f"{stamp}   DIAG-OK    {s.name}")
            self._diag_level[s.name] = lvl

    def _on_graph_struct(self, msg):
        # Transient-local + latched by aggregator's own lifecycle: arrives
        # once (or once per graph reconfiguration). Index order matches
        # DiagGraphStatus.nodes / .diags for the same message id.
        self._graph_paths = [n.path for n in msg.nodes]
        self._graph_diags = [d.name for d in msg.diags]

    def _on_graph_status(self, msg):
        if self._graph_paths is None or self._graph_diags is None:
            return  # struct not received yet; can't name anything usefully
        stamp = self._stamp()
        for i, node in enumerate(msg.nodes):
            path = self._graph_paths[i] if i < len(self._graph_paths) else f"#{i}"
            level = self._level(node.level)
            input_level = self._level(node.input_level)
            latch_level = self._level(node.latch_level)
            prev_level, prev_latch = self._graph_node_state.get(path, (0, 0))
            if (level, latch_level) != (prev_level, prev_latch):
                if level != 0 or latch_level != 0:
                    latched_only = " LATCHED(recovered-input)" if (
                        latch_level != 0 and input_level == 0
                    ) else ""
                    self._emit(
                        f"{stamp}   GRAPH-FAIL node={path} level={level} "
                        f"input={input_level} latch={latch_level}{latched_only}"
                    )
                else:
                    self._emit(f"{stamp}   GRAPH-OK   node={path}")
            self._graph_node_state[path] = (level, latch_level)

        for i, diag in enumerate(msg.diags):
            name = self._graph_diags[i] if i < len(self._graph_diags) else f"#{i}"
            level = self._level(diag.level)
            input_level = self._level(diag.input_level)
            prev = self._graph_diag_state.get(name, 0)
            if level != prev:
                if level != 0:
                    self._emit(
                        f"{stamp}   GRAPH-LEAF-FAIL {name} level={level} "
                        f"input={input_level} msg={diag.message!r}"
                    )
                else:
                    self._emit(f"{stamp}   GRAPH-LEAF-OK    {name}")
            self._graph_diag_state[name] = level

    def _tick(self):
        now = time.time()
        vel = self.state.get("vel")
        cmd = self.state.get("cmd")
        traj = self.state.get("traj")
        mode = self.state.get("mode")
        mrm = self.state.get("mrm")
        aeb = self.state.get("aeb")
        avail = self.state.get("avail")
        pos = self.state.get("pos")

        def age(key):
            t = self.stamps.get(key)
            return f"{now - t:.1f}" if t else "--"

        cmd_v, cmd_a = (cmd if isinstance(cmd, tuple) else (None, None))
        if isinstance(avail, tuple):
            a_auto, a_stop, a_local, a_remote, a_estop, a_comf, a_pull = avail
            avail_str = (
                f"auto={a_auto} stop={a_stop} local={a_local} remote={a_remote} "
                f"estop={a_estop} comfort={a_comf} pullover={a_pull}"
            )
        else:
            avail_str = f"{avail}"
        pos_str = (
            f"{pos[0]:.2f},{pos[1]:.2f}" if isinstance(pos, tuple) else f"{pos}"
        )
        line = (
            f"{datetime.now().strftime('%H:%M:%S.%f')[:-3]} "
            f"vel={vel if vel is None else f'{vel:6.2f}'} "
            f"cmd_v={cmd_v if cmd_v is None else f'{cmd_v:6.2f}'} "
            f"cmd_a={cmd_a if cmd_a is None else f'{cmd_a:6.2f}'} "
            f"traj0={traj if traj is None else f'{traj:6.2f}'} "
            f"mode={mode} mrm={mrm} aeb={aeb} avail=[{avail_str}] pos=({pos_str}) "
            f"| age vel={age('vel')} cmd={age('cmd')} traj={age('traj')}"
        )
        print(line, flush=True)
        if self.log:
            self.log.write(line + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", default=None, help="also append output to this file")
    args = p.parse_args()

    rclpy.init()
    node = Monitor(args.log)
    print("monitoring — start/stop the recorder and watch which column changes")
    print("(Ctrl+C to stop)\n", flush=True)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
