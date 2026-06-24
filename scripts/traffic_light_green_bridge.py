#!/usr/bin/env python3
"""Publish all map traffic lights as GREEN on the external traffic-signals topic.

Run inside the Autoware Docker container after sourcing ROS 2 + Autoware.

The traffic_light_arbiter merges ~/sub/external_traffic_signals with camera
perception and publishes merged state to /perception/traffic_light_recognition/traffic_signals,
which behavior_velocity_planner uses to release stops at intersections.
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List

import rclpy
from autoware_perception_msgs.msg import (
    TrafficLightElement,
    TrafficLightGroup,
    TrafficLightGroupArray,
)
from rclpy.node import Node
from rclpy.parameter import Parameter

DEFAULT_MAP = "/home/aw/maps/nishishinjuku_autoware_map/lanelet2_map.osm"
DEFAULT_TOPIC = "/perception/traffic_light_recognition/external/traffic_signals"
DEFAULT_OUTPUT_TOPIC = "/perception/traffic_light_recognition/traffic_signals"
DEFAULT_RATE_HZ = 10.0


def extract_traffic_light_group_ids(map_path: Path) -> List[int]:
    """Return Lanelet2 regulatory_element relation IDs for subtype=traffic_light."""
    tree = ET.parse(map_path)
    ids: List[int] = []
    for relation in tree.iterfind("relation"):
        tags = {tag.get("k"): tag.get("v") for tag in relation.findall("tag")}
        if tags.get("type") == "regulatory_element" and tags.get("subtype") == "traffic_light":
            ids.append(int(relation.get("id")))
    ids.sort()
    return ids


class TrafficLightGreenBridge(Node):
    def __init__(
        self,
        group_ids: List[int],
        topic: str,
        rate_hz: float,
        *,
        bypass_arbiter: bool = False,
    ) -> None:
        super().__init__(
            "traffic_light_green_bridge",
            parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)],
        )
        self.group_ids = group_ids
        self.bypass_arbiter = bypass_arbiter
        publish_topic = DEFAULT_OUTPUT_TOPIC if bypass_arbiter else topic
        # Match arbiter QoS depth (KEEP_LAST 1) on the merged output topic.
        qos_depth = 1 if bypass_arbiter else 10
        self.publisher = self.create_publisher(TrafficLightGroupArray, publish_topic, qos_depth)
        period_sec = 1.0 / rate_hz if rate_hz > 0.0 else 0.1
        self.timer = self.create_timer(period_sec, self._publish_green)
        mode = "bypass-arbiter" if bypass_arbiter else "external-input"
        self.get_logger().info(
            f"Publishing GREEN for {len(group_ids)} traffic_light_group_id values "
            f"to {publish_topic} ({mode}) at {rate_hz:.1f} Hz with use_sim_time=true"
        )

    def _make_green_group(self, group_id: int) -> TrafficLightGroup:
        element = TrafficLightElement()
        element.color = TrafficLightElement.GREEN
        element.shape = TrafficLightElement.CIRCLE
        element.status = TrafficLightElement.SOLID_ON
        element.confidence = 1.0

        group = TrafficLightGroup()
        group.traffic_light_group_id = group_id
        group.elements = [element]
        group.predictions = []
        return group

    def _publish_green(self) -> None:
        msg = TrafficLightGroupArray()
        msg.stamp = self.get_clock().now().to_msg()
        msg.traffic_light_groups = [self._make_green_group(group_id) for group_id in self.group_ids]
        self.publisher.publish(msg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "map_path",
        nargs="?",
        default=DEFAULT_MAP,
        help=f"Lanelet2 OSM map path (default: {DEFAULT_MAP})",
    )
    parser.add_argument(
        "--topic",
        default=DEFAULT_TOPIC,
        help=f"External traffic-signals topic (default: {DEFAULT_TOPIC})",
    )
    parser.add_argument(
        "--rate-hz",
        type=float,
        default=DEFAULT_RATE_HZ,
        help=f"Publish rate in Hz (default: {DEFAULT_RATE_HZ})",
    )
    parser.add_argument(
        "--ids",
        nargs="*",
        type=int,
        help="Optional subset of traffic_light_group_id values (default: all in map)",
    )
    parser.add_argument(
        "--bypass-arbiter",
        action="store_true",
        help="Publish directly to /perception/traffic_light_recognition/traffic_signals",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    map_path = Path(args.map_path)
    if not map_path.is_file():
        print(f"Map not found: {map_path}", file=sys.stderr)
        return 1

    group_ids = args.ids if args.ids else extract_traffic_light_group_ids(map_path)
    if not group_ids:
        print(f"No traffic_light regulatory elements found in {map_path}", file=sys.stderr)
        return 1

    rclpy.init()
    node = TrafficLightGreenBridge(
        group_ids,
        args.topic,
        args.rate_hz,
        bypass_arbiter=args.bypass_arbiter,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
