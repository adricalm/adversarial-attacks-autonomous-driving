#!/usr/bin/env python3
"""Find routable lanelet start/goal candidates near an ego pose from lanelet2_map.osm."""

from __future__ import annotations

import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

Point = Tuple[float, float, float]


@dataclass
class Lanelet:
    lanelet_id: int
    centerline: List[Point]
    heading_rad: float
    start: Point
    end: Point
    start_node_ids: Tuple[int, int]
    end_node_ids: Tuple[int, int]


def tag_value(element: ET.Element, key: str) -> Optional[str]:
    for tag in element.findall("tag"):
        if tag.get("k") == key:
            return tag.get("v")
    return None


def parse_nodes(root: ET.Element) -> Dict[int, Point]:
    nodes: Dict[int, Point] = {}
    for node in root.findall("node"):
        node_id = int(node.get("id"))
        local_x = tag_value(node, "local_x")
        local_y = tag_value(node, "local_y")
        ele = tag_value(node, "ele")
        if local_x is None or local_y is None:
            continue
        z = float(ele) if ele is not None else 0.0
        nodes[node_id] = (float(local_x), float(local_y), z)
    return nodes


def parse_ways(root: ET.Element) -> Dict[int, List[int]]:
    ways: Dict[int, List[int]] = {}
    for way in root.findall("way"):
        way_id = int(way.get("id"))
        ways[way_id] = [int(nd.get("ref")) for nd in way.findall("nd")]
    return ways


def interpolate_line(points: Sequence[Point], target_count: int) -> List[Point]:
    if not points:
        return []
    if len(points) == target_count:
        return list(points)
    if len(points) == 1:
        return [points[0]] * target_count

    cumulative = [0.0]
    for i in range(1, len(points)):
        dx = points[i][0] - points[i - 1][0]
        dy = points[i][1] - points[i - 1][1]
        dz = points[i][2] - points[i - 1][2]
        cumulative.append(cumulative[-1] + math.hypot(dx, dy) + abs(dz) * 0.01)

    total = cumulative[-1]
    if total <= 1e-9:
        return [points[0]] * target_count

    result: List[Point] = []
    for i in range(target_count):
        target_s = total * i / (target_count - 1)
        j = 0
        while j + 1 < len(cumulative) and cumulative[j + 1] < target_s:
            j += 1
        if j + 1 >= len(points):
            result.append(points[-1])
            continue
        seg_len = cumulative[j + 1] - cumulative[j]
        t = 0.0 if seg_len <= 1e-9 else (target_s - cumulative[j]) / seg_len
        x = points[j][0] + t * (points[j + 1][0] - points[j][0])
        y = points[j][1] + t * (points[j + 1][1] - points[j][1])
        z = points[j][2] + t * (points[j + 1][2] - points[j][2])
        result.append((x, y, z))
    return result


def way_points(way_id: int, ways: Dict[int, List[int]], nodes: Dict[int, Point]) -> List[Point]:
    refs = ways.get(way_id, [])
    return [nodes[ref] for ref in refs if ref in nodes]


def compute_centerline(left: List[Point], right: List[Point]) -> List[Point]:
    count = max(len(left), len(right))
    if count == 0:
        return []
    left_i = interpolate_line(left, count)
    right_i = interpolate_line(right, count)
    return [
        ((lx + rx) / 2.0, (ly + ry) / 2.0, (lz + rz) / 2.0)
        for (lx, ly, lz), (rx, ry, rz) in zip(left_i, right_i)
    ]


def heading_from_points(a: Point, b: Point) -> float:
    return math.atan2(b[1] - a[1], b[0] - a[0])


def angle_diff(a: float, b: float) -> float:
    d = (a - b + math.pi) % (2 * math.pi) - math.pi
    return abs(d)


def yaw_to_quat(yaw: float) -> Tuple[float, float, float, float]:
    half = yaw / 2.0
    return (0.0, 0.0, math.sin(half), math.cos(half))


def dist2d(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def dist3d(a: Point, b: Point) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def parse_lanelets(
    root: ET.Element,
    nodes: Dict[int, Point],
    ways: Dict[int, List[int]],
) -> List[Lanelet]:
    lanelets: List[Lanelet] = []
    for relation in root.findall("relation"):
        if tag_value(relation, "type") != "lanelet":
            continue
        if tag_value(relation, "subtype") != "road":
            continue
        if tag_value(relation, "participant:vehicle") != "yes":
            continue

        left_way_id: Optional[int] = None
        right_way_id: Optional[int] = None
        for member in relation.findall("member"):
            if member.get("type") != "way":
                continue
            role = member.get("role")
            ref = int(member.get("ref"))
            if role == "left":
                left_way_id = ref
            elif role == "right":
                right_way_id = ref

        if left_way_id is None or right_way_id is None:
            continue

        left_refs = ways.get(left_way_id, [])
        right_refs = ways.get(right_way_id, [])
        left = way_points(left_way_id, ways, nodes)
        right = way_points(right_way_id, ways, nodes)
        centerline = compute_centerline(left, right)
        if len(centerline) < 2:
            continue

        heading = heading_from_points(centerline[0], centerline[1])
        start_nodes = (left_refs[0], right_refs[0])
        end_nodes = (left_refs[-1], right_refs[-1])
        lanelets.append(
            Lanelet(
                lanelet_id=int(relation.get("id")),
                centerline=centerline,
                heading_rad=heading,
                start=centerline[0],
                end=centerline[-1],
                start_node_ids=start_nodes,
                end_node_ids=end_nodes,
            )
        )
    return lanelets


def build_successors(lanelets: Sequence[Lanelet]) -> Dict[int, List[int]]:
    # Map start_node_ids -> lanelet_ids so we can find which lanelet begins
    # where another ends (i.e., the forward successor).
    start_to_lanelets: Dict[Tuple[int, int], List[int]] = {}
    for lanelet in lanelets:
        start_to_lanelets.setdefault(lanelet.start_node_ids, []).append(lanelet.lanelet_id)

    successors: Dict[int, List[int]] = {lanelet.lanelet_id: [] for lanelet in lanelets}
    for lanelet in lanelets:
        for succ_id in start_to_lanelets.get(lanelet.end_node_ids, []):
            if succ_id != lanelet.lanelet_id:
                successors[lanelet.lanelet_id].append(succ_id)
    return successors


def nearest_point_on_centerline(lanelet: Lanelet, ego: Point) -> Tuple[Point, float, float]:
    best_point = lanelet.start
    best_dist = float("inf")
    best_forward = 0.0
    for i in range(len(lanelet.centerline) - 1):
        a = lanelet.centerline[i]
        b = lanelet.centerline[i + 1]
        ax, ay = a[0] - ego[0], a[1] - ego[1]
        bx, by = b[0] - ego[0], b[1] - ego[1]
        seg_dx, seg_dy = b[0] - a[0], b[1] - a[1]
        seg_len2 = seg_dx * seg_dx + seg_dy * seg_dy
        if seg_len2 <= 1e-9:
            t = 0.0
            px, py, pz = a
        else:
            t = max(0.0, min(1.0, (ax * seg_dx + ay * seg_dy) / seg_len2))
            px = a[0] + t * seg_dx
            py = a[1] + t * seg_dy
            pz = a[2] + t * (b[2] - a[2])
        dist = math.hypot(px - ego[0], py - ego[1])
        if dist < best_dist:
            best_dist = dist
            best_point = (px, py, pz)
            best_forward = math.hypot(px - lanelet.start[0], py - lanelet.start[1])
    return best_point, best_dist, best_forward


def collect_forward_path_points(
    start_lanelet: Lanelet,
    start_pose: Point,
    successors: Dict[int, List[int]],
    lanelet_by_id: Dict[int, Lanelet],
    max_depth: int = 20,
) -> List[Tuple[float, int, Point, float]]:
    """Return points ahead of start_pose as (arc_length, lanelet_id, point, heading)."""
    rows: List[Tuple[float, int, Point, float]] = []

    def append_lanelet_points(lanelet: Lanelet, accumulated: float, begin_at_start: bool) -> float:
        points = lanelet.centerline
        if begin_at_start:
            start_idx = 0
            for i, point in enumerate(points):
                if dist2d(point, start_pose) <= dist2d(points[start_idx], start_pose):
                    start_idx = i
            points = points[start_idx:]

        prev = start_pose if begin_at_start and points else None
        arc = accumulated
        for point in points:
            if prev is not None:
                arc += dist2d(prev, point)
            rows.append((arc, lanelet.lanelet_id, point, lanelet.heading_rad))
            prev = point
        return arc

    visited = {start_lanelet.lanelet_id}
    queue: List[Tuple[int, float, bool]] = [(start_lanelet.lanelet_id, 0.0, True)]
    depth = 0
    while queue and depth < max_depth:
        next_queue: List[Tuple[int, float, bool]] = []
        for lanelet_id, accumulated, begin_at_start in queue:
            lanelet = lanelet_by_id[lanelet_id]
            arc_end = append_lanelet_points(lanelet, accumulated, begin_at_start)
            for succ_id in successors.get(lanelet_id, []):
                if succ_id in visited:
                    continue
                visited.add(succ_id)
                next_queue.append((succ_id, arc_end, False))
        queue = next_queue
        depth += 1

    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--map",
        default="/home/adria/summer26/data/maps/nishishinjuku_autoware_map/lanelet2_map.osm",
        help="Path to lanelet2_map.osm",
    )
    parser.add_argument("--ego-x", type=float, default=81685.560)
    parser.add_argument("--ego-y", type=float, default=50307.470)
    parser.add_argument("--ego-z", type=float, default=40.924)
    parser.add_argument("--ego-yaw", type=float, default=1.848, help="Ego yaw in radians")
    parser.add_argument("--radius", type=float, default=150.0)
    parser.add_argument("--heading-tol-deg", type=float, default=45.0)
    parser.add_argument("--goal-min-forward", type=float, default=30.0)
    parser.add_argument("--goal-max-forward", type=float, default=80.0)
    parser.add_argument(
        "--json-out",
        default="",
        help="Optional path to write start/goal JSON for docker apply script",
    )
    args = parser.parse_args()

    ego = (args.ego_x, args.ego_y, args.ego_z)
    heading_tol = math.radians(args.heading_tol_deg)

    print(f"Parsing map: {args.map}")
    root = ET.parse(args.map).getroot()
    nodes = parse_nodes(root)
    ways = parse_ways(root)
    lanelets = parse_lanelets(root, nodes, ways)
    successors = build_successors(lanelets)
    lanelet_by_id = {lanelet.lanelet_id: lanelet for lanelet in lanelets}

    print(f"Loaded {len(nodes)} nodes, {len(ways)} ways, {len(lanelets)} routable lanelets")
    print()

    candidates = []
    for lanelet in lanelets:
        nearest, lateral_dist, _ = nearest_point_on_centerline(lanelet, ego)
        if lateral_dist > args.radius:
            continue
        if angle_diff(lanelet.heading_rad, args.ego_yaw) > heading_tol:
            continue
        candidates.append((lateral_dist, lanelet, nearest))

    candidates.sort(key=lambda item: item[0])
    if not candidates:
        print("No lanelet candidates found near ego pose.")
        return 1

    print("Nearby lanelet candidates (sorted by lateral distance to centerline):")
    print("lanelet_id,lateral_dist_m,heading_deg,start_x,start_y,start_z,end_x,end_y,end_z")
    for lateral_dist, lanelet, nearest in candidates[:30]:
        heading_deg = math.degrees(lanelet.heading_rad)
        print(
            f"{lanelet.lanelet_id},{lateral_dist:.2f},{heading_deg:.1f},"
            f"{lanelet.start[0]:.3f},{lanelet.start[1]:.3f},{lanelet.start[2]:.3f},"
            f"{lanelet.end[0]:.3f},{lanelet.end[1]:.3f},{lanelet.end[2]:.3f}"
        )
    print()

    start_lanelet = candidates[0][1]
    start_pose = candidates[0][2]
    start_yaw = start_lanelet.heading_rad
    qx, qy, qz, qw = yaw_to_quat(start_yaw)

    print("Recommended START (nearest compatible lanelet):")
    print(f"  lanelet_id: {start_lanelet.lanelet_id}")
    print(f"  pose: x={start_pose[0]:.6f}, y={start_pose[1]:.6f}, z={start_pose[2]:.6f}")
    print(f"  yaw_rad: {start_yaw:.6f}  yaw_deg: {math.degrees(start_yaw):.3f}")
    print(f"  orientation: x={qx:.6f}, y={qy:.6f}, z={qz:.6f}, w={qw:.6f}")
    print()

    # Search goal along connected lanelet path using arc-length distance.
    path_points = collect_forward_path_points(
        start_lanelet, start_pose, successors, lanelet_by_id
    )
    goal_candidates = []
    for arc_len, lanelet_id, point, heading in path_points:
        if args.goal_min_forward <= arc_len <= args.goal_max_forward:
            goal_candidates.append((abs(arc_len - 50.0), arc_len, lanelet_id, point, heading))

    if not goal_candidates and path_points:
        # Fallback: farthest point within max_forward on the path.
        for arc_len, lanelet_id, point, heading in path_points:
            if arc_len <= args.goal_max_forward:
                goal_candidates.append((-arc_len, arc_len, lanelet_id, point, heading))

    if not goal_candidates:
        print("No goal candidate found 30-80 m ahead on connected lanelets.")
        print(f"Successors of lanelet {start_lanelet.lanelet_id}: {successors.get(start_lanelet.lanelet_id, [])}")
        if path_points:
            print("Forward path sample (arc_len, lanelet_id, x, y, z):")
            for arc_len, lanelet_id, point, _ in path_points[:20]:
                print(f"  {arc_len:.2f}, {lanelet_id}, {point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}")
        return 1

    goal_candidates.sort(key=lambda item: item[0])
    _, forward_dist, goal_lanelet_id, goal_pose, goal_yaw = goal_candidates[0]
    gqx, gqy, gqz, gqw = yaw_to_quat(goal_yaw)

    print("Recommended GOAL:")
    print(f"  lanelet_id: {goal_lanelet_id}")
    print(f"  forward_distance_m: {forward_dist:.2f}")
    print(f"  pose: x={goal_pose[0]:.6f}, y={goal_pose[1]:.6f}, z={goal_pose[2]:.6f}")
    print(f"  yaw_rad: {goal_yaw:.6f}  yaw_deg: {math.degrees(goal_yaw):.3f}")
    print(f"  orientation: x={gqx:.6f}, y={gqy:.6f}, z={gqz:.6f}, w={gqw:.6f}")
    print()

    result = {
        "start": {
            "lanelet_id": start_lanelet.lanelet_id,
            "position": {"x": start_pose[0], "y": start_pose[1], "z": start_pose[2]},
            "orientation": {"x": qx, "y": qy, "z": qz, "w": qw},
            "yaw_rad": start_yaw,
        },
        "goal": {
            "lanelet_id": goal_lanelet_id,
            "forward_distance_m": forward_dist,
            "position": {"x": goal_pose[0], "y": goal_pose[1], "z": goal_pose[2]},
            "orientation": {"x": gqx, "y": gqy, "z": gqz, "w": gqw},
            "yaw_rad": goal_yaw,
        },
    }
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
        print(f"Wrote route candidates to {args.json_out}")

    print("ROS2 service snippets (run inside Docker):")
    print()
    print("ros2 service call /localization/initialize autoware_localization_msgs/srv/InitializeLocalization \\")
    print("  \"{pose_with_covariance: {pose: {position: {x: "
          f"{start_pose[0]:.6f}, y: {start_pose[1]:.6f}, z: {start_pose[2]:.6f}"
          "}, orientation: {x: "
          f"{qx:.6f}, y: {qy:.6f}, z: {qz:.6f}, w: {qw:.6f}"
          "}}}, method: 1}\"")
    print()
    print("ros2 service call /api/routing/set_route_points autoware_adapi_v1_msgs/srv/SetRoutePoints \\")
    print("  \"{header: {frame_id: map}, option: {allow_goal_modification: false}, goal: {position: {x: "
          f"{goal_pose[0]:.6f}, y: {goal_pose[1]:.6f}, z: {goal_pose[2]:.6f}"
          "}, orientation: {x: "
          f"{gqx:.6f}, y: {gqy:.6f}, z: {gqz:.6f}, w: {gqw:.6f}"
          "}}, waypoints: []}\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
