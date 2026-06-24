#!/usr/bin/env python3
"""Apply route candidates from JSON using Autoware ROS2 services (run inside Docker)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import rclpy
from rclpy.node import Node

from autoware_adapi_v1_msgs.srv import SetRoutePoints
from autoware_localization_msgs.srv import InitializeLocalization
from geometry_msgs.msg import PoseWithCovarianceStamped


DEFAULT_JSON = "/home/aw/autoware_data/route_candidates.json"
RESULT_JSON = "/home/aw/autoware_data/apply_route_result.json"


class RouteApplier(Node):
    def __init__(self) -> None:
        super().__init__("apply_route_from_osm")
        self.localization_client = self.create_client(
            InitializeLocalization, "/localization/initialize"
        )
        self.routing_client = self.create_client(SetRoutePoints, "/api/routing/set_route_points")

    def wait_for_service(self, client, name: str, timeout_sec: float = 30.0) -> bool:
        if client.wait_for_service(timeout_sec=timeout_sec):
            return True
        self.get_logger().error(f"Service unavailable: {name}")
        return False

    def initialize_localization(self, start: dict) -> bool:
        if not self.wait_for_service(self.localization_client, "/localization/initialize"):
            return False

        stamped = PoseWithCovarianceStamped()
        stamped.header.frame_id = "map"
        stamped.pose.pose.position.x = float(start["position"]["x"])
        stamped.pose.pose.position.y = float(start["position"]["y"])
        stamped.pose.pose.position.z = float(start["position"]["z"])
        stamped.pose.pose.orientation.x = float(start["orientation"]["x"])
        stamped.pose.pose.orientation.y = float(start["orientation"]["y"])
        stamped.pose.pose.orientation.z = float(start["orientation"]["z"])
        stamped.pose.pose.orientation.w = float(start["orientation"]["w"])

        request = InitializeLocalization.Request()
        request.pose_with_covariance = [stamped]  # bounded sequence [<=1]
        request.method = 1  # DIRECT

        future = self.localization_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=30.0)
        if not future.done() or future.result() is None:
            self.get_logger().error("Localization initialize call timed out")
            return False

        response = future.result()
        status = response.status
        self.get_logger().info(
            f"/localization/initialize -> success={status.success}, "
            f"code={status.code}, message='{status.message}'"
        )
        return bool(status.success)

    def set_route(self, goal: dict) -> bool:
        if not self.wait_for_service(self.routing_client, "/api/routing/set_route_points"):
            return False

        request = SetRoutePoints.Request()
        request.header.frame_id = "map"
        request.option.allow_goal_modification = False
        request.goal.position.x = float(goal["position"]["x"])
        request.goal.position.y = float(goal["position"]["y"])
        request.goal.position.z = float(goal["position"]["z"])
        request.goal.orientation.x = float(goal["orientation"]["x"])
        request.goal.orientation.y = float(goal["orientation"]["y"])
        request.goal.orientation.z = float(goal["orientation"]["z"])
        request.goal.orientation.w = float(goal["orientation"]["w"])

        future = self.routing_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=30.0)
        if not future.done() or future.result() is None:
            self.get_logger().error("Routing call timed out")
            return False

        response = future.result()
        status = response.status
        self.get_logger().info(
            "Routing response: "
            f"success={status.success}, code={status.code}, message='{status.message}'"
        )
        return bool(status.success)


def write_result(payload: dict) -> None:
    Path(RESULT_JSON).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    json_path = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_JSON)
    if not json_path.exists():
        print(f"Missing route JSON: {json_path}", file=sys.stderr)
        return 1

    data = json.loads(json_path.read_text(encoding="utf-8"))
    rclpy.init()
    node = RouteApplier()
    result = {"localization_success": False, "routing_success": False}
    try:
        result["localization_success"] = node.initialize_localization(data["start"])
        time.sleep(1.0)
        if result["localization_success"]:
            result["routing_success"] = node.set_route(data["goal"])
        write_result(result)
        return 0 if result["localization_success"] and result["routing_success"] else 1
    except Exception as exc:  # noqa: BLE001
        result["error"] = str(exc)
        write_result(result)
        return 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
