#!/usr/bin/env python3
"""Engage autonomous mode and verify control reaches AWSIM (run inside Docker)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import rclpy
from autoware_adapi_v1_msgs.srv import ChangeOperationMode
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

try:
    from autoware_system_msgs.msg import AutowareState
except ImportError:  # older Autoware images
    from tier4_system_msgs.msg import AutowareState

RESULT_JSON = "/home/aw/autoware_data/engage_result.json"

AUTOWARE_STATE_NAMES = {
    1: "INITIALIZING",
    2: "WAITING_FOR_ROUTE",
    3: "PLANNING",
    4: "WAITING_FOR_ENGAGE",
    5: "DRIVING",
    6: "ARRIVED_GOAL",
    7: "FINALIZING",
}


class EngageVerifier(Node):
    def __init__(self) -> None:
        super().__init__("engage_autoware")
        self.autoware_state: Optional[int] = None
        self.trajectory_velocities: List[float] = []
        self.control_cmd: Optional[Dict[str, Any]] = None
        self.velocity_status: Optional[Dict[str, Any]] = None

        self.create_subscription(
            AutowareState,
            "/autoware/state",
            self._on_autoware_state,
            qos_profile_sensor_data,
        )

    def _on_autoware_state(self, msg: AutowareState) -> None:
        self.autoware_state = int(msg.state)

    def wait_for_autoware_state(self, timeout_sec: float = 10.0) -> Optional[int]:
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.2)
            if self.autoware_state is not None:
                return self.autoware_state
        return self.autoware_state

    def sample_trajectory_velocities(self, timeout_sec: float = 5.0) -> List[float]:
        from autoware_planning_msgs.msg import Trajectory

        velocities: List[float] = []
        done = {"flag": False}

        def callback(msg: Trajectory) -> None:
            for point in msg.points:
                velocities.append(float(point.longitudinal_velocity_mps))
            done["flag"] = True

        sub = self.create_subscription(
            Trajectory,
            "/planning/scenario_planning/trajectory",
            callback,
            qos_profile_sensor_data,
        )
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline and not done["flag"]:
            rclpy.spin_once(self, timeout_sec=0.2)
        self.destroy_subscription(sub)
        self.trajectory_velocities = velocities
        return velocities

    def call_change_to_autonomous(self, timeout_sec: float = 15.0) -> Dict[str, Any]:
        client = self.create_client(ChangeOperationMode, "/api/operation_mode/change_to_autonomous")
        result: Dict[str, Any] = {"service": "/api/operation_mode/change_to_autonomous", "success": False}
        if not client.wait_for_service(timeout_sec=timeout_sec):
            result["error"] = "service unavailable"
            return result

        future = client.call_async(ChangeOperationMode.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)
        if not future.done() or future.result() is None:
            result["error"] = "service call timed out"
            return result

        response = future.result()
        status = response.status
        result.update(
            {
                "success": bool(status.success),
                "code": int(status.code),
                "message": str(status.message),
            }
        )
        return result

    def call_accept_start_if_available(self, timeout_sec: float = 5.0) -> Optional[Dict[str, Any]]:
        from autoware_adapi_v1_msgs.srv import AcceptStart

        service_name = "/api/motion/accept_start"
        client = self.create_client(AcceptStart, service_name)
        if not client.wait_for_service(timeout_sec=timeout_sec):
            return None

        future = client.call_async(AcceptStart.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)
        if not future.done() or future.result() is None:
            return {"service": service_name, "success": False, "error": "service call timed out"}

        response = future.result()
        status = response.status
        return {
            "service": service_name,
            "success": bool(status.success),
            "code": int(status.code),
            "message": str(status.message),
        }

    def sample_control_cmd(self, timeout_sec: float = 5.0) -> Optional[Dict[str, Any]]:
        from autoware_control_msgs.msg import Control

        payload: Optional[Dict[str, Any]] = None
        done = {"flag": False}

        def callback(msg: Control) -> None:
            nonlocal payload
            payload = {
                "longitudinal_velocity": float(msg.longitudinal.velocity),
                "longitudinal_acceleration": float(msg.longitudinal.acceleration),
                "lateral_steering_tire_angle": float(msg.lateral.steering_tire_angle),
            }
            done["flag"] = True

        sub = self.create_subscription(
            Control,
            "/control/command/control_cmd",
            callback,
            qos_profile_sensor_data,
        )
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline and not done["flag"]:
            rclpy.spin_once(self, timeout_sec=0.2)
        self.destroy_subscription(sub)
        self.control_cmd = payload
        return payload

    def sample_velocity_status(self, timeout_sec: float = 8.0) -> Optional[Dict[str, Any]]:
        from autoware_vehicle_msgs.msg import VelocityReport

        payload: Optional[Dict[str, Any]] = None
        samples: List[float] = []
        done = {"flag": False}

        def callback(msg: VelocityReport) -> None:
            samples.append(float(msg.longitudinal_velocity))
            nonlocal payload
            payload = {
                "longitudinal_velocity": float(msg.longitudinal_velocity),
                "lateral_velocity": float(msg.lateral_velocity),
                "heading_rate": float(msg.heading_rate),
            }
            if abs(msg.longitudinal_velocity) > 0.05:
                done["flag"] = True

        sub = self.create_subscription(
            VelocityReport,
            "/vehicle/status/velocity_status",
            callback,
            qos_profile_sensor_data,
        )
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline and not done["flag"]:
            rclpy.spin_once(self, timeout_sec=0.2)
        self.destroy_subscription(sub)
        if payload is not None:
            payload["samples"] = samples
        self.velocity_status = payload
        return payload


def summarize_velocities(velocities: List[float]) -> Dict[str, Any]:
    non_zero = [v for v in velocities if abs(v) > 1e-3]
    return {
        "count": len(velocities),
        "non_zero_count": len(non_zero),
        "max_abs_mps": max((abs(v) for v in velocities), default=0.0),
        "sample": velocities[:10],
    }


def write_result(payload: Dict[str, Any]) -> None:
    Path(RESULT_JSON).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    rclpy.init()
    node = EngageVerifier()
    result: Dict[str, Any] = {"steps": {}}

    try:
        state_before = node.wait_for_autoware_state()
        result["steps"]["autoware_state_before"] = {
            "value": state_before,
            "name": AUTOWARE_STATE_NAMES.get(state_before or -1, "UNKNOWN"),
        }
        node.get_logger().info(
            f"Autoware state before engage: {state_before} "
            f"({AUTOWARE_STATE_NAMES.get(state_before or -1, 'UNKNOWN')})"
        )

        traj_vels = node.sample_trajectory_velocities()
        traj_summary = summarize_velocities(traj_vels)
        result["steps"]["trajectory"] = traj_summary
        node.get_logger().info(f"Trajectory velocities: {traj_summary}")

        engage_result = node.call_change_to_autonomous()
        result["steps"]["change_to_autonomous"] = engage_result
        node.get_logger().info(f"change_to_autonomous -> {engage_result}")

        time.sleep(1.0)
        state_after_engage = node.wait_for_autoware_state(timeout_sec=5.0)
        result["steps"]["autoware_state_after_engage"] = {
            "value": state_after_engage,
            "name": AUTOWARE_STATE_NAMES.get(state_after_engage or -1, "UNKNOWN"),
        }

        accept_start_result = None
        if state_after_engage != 5:
            accept_start_result = node.call_accept_start_if_available()
            if accept_start_result is not None:
                result["steps"]["accept_start"] = accept_start_result
                node.get_logger().info(f"accept_start -> {accept_start_result}")
                time.sleep(1.0)
                state_after_accept = node.wait_for_autoware_state(timeout_sec=5.0)
                result["steps"]["autoware_state_after_accept_start"] = {
                    "value": state_after_accept,
                    "name": AUTOWARE_STATE_NAMES.get(state_after_accept or -1, "UNKNOWN"),
                }
                state_after_engage = state_after_accept

        control_cmd = node.sample_control_cmd()
        result["steps"]["control_cmd"] = control_cmd
        node.get_logger().info(f"control_cmd sample: {control_cmd}")

        velocity_status = node.sample_velocity_status(timeout_sec=10.0)
        result["steps"]["velocity_status"] = velocity_status
        node.get_logger().info(f"velocity_status sample: {velocity_status}")

        driving = state_after_engage == 5
        moving = bool(
            velocity_status
            and abs(float(velocity_status.get("longitudinal_velocity", 0.0))) > 0.05
        )
        result["summary"] = {
            "driving": driving,
            "vehicle_moving": moving,
            "trajectory_has_non_zero_velocity": traj_summary["non_zero_count"] > 0,
            "engage_service_success": bool(engage_result.get("success")),
        }

        write_result(result)
        if not driving:
            return 2
        if not moving:
            return 3
        return 0
    except Exception as exc:  # noqa: BLE001
        result["error"] = str(exc)
        write_result(result)
        return 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
