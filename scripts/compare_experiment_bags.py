#!/usr/bin/env python3
"""Compare two experiment rosbags (baseline vs dsgn ON).

Run inside autoware_full_test (needs Autoware message types + rosbag2_py):

  python3 /home/aw/scripts/compare_experiment_bags.py \\
    /home/aw/bags/run_a_baseline_001 /home/aw/bags/run_b_dsgn_offline_001

Host wrapper:

  bash ~/summer26/scripts/compare_experiment_bags.sh \\
    run_a_baseline_001 run_b_dsgn_offline_001
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from rosidl_runtime_py.utilities import get_message


@dataclass
class SeriesStats:
    count: int = 0
    min_val: float | None = None
    max_val: float | None = None
    sum_val: float = 0.0

    def add(self, value: float) -> None:
        self.count += 1
        self.sum_val += value
        self.min_val = value if self.min_val is None else min(self.min_val, value)
        self.max_val = value if self.max_val is None else max(self.max_val, value)

    @property
    def mean(self) -> float | None:
        return self.sum_val / self.count if self.count else None


@dataclass
class BagMetrics:
    label: str
    path: Path
    duration_s: float = 0.0
    first_stamp_ns: int | None = None
    last_stamp_ns: int | None = None
    velocity: SeriesStats = field(default_factory=SeriesStats)
    stopped_samples: int = 0
    tracked_count: SeriesStats = field(default_factory=SeriesStats)
    predicted_count: SeriesStats = field(default_factory=SeriesStats)
    detection_count: SeriesStats = field(default_factory=SeriesStats)
    frames_with_tracked: int = 0
    frames_with_predicted: int = 0
    frames_with_detection: int = 0
    obstacle_stop_msgs: int = 0
    obstacle_stop_with_factors: int = 0
    min_traj_velocity: float | None = None
    distance_m: float = 0.0
    topic_counts: dict[str, int] = field(default_factory=dict)

    def note_stamp(self, stamp_ns: int) -> None:
        if self.first_stamp_ns is None:
            self.first_stamp_ns = stamp_ns
        self.last_stamp_ns = stamp_ns

    def finalize(self) -> None:
        if self.first_stamp_ns is not None and self.last_stamp_ns is not None:
            self.duration_s = (self.last_stamp_ns - self.first_stamp_ns) / 1e9


def _stamp_ns(msg: Any) -> int | None:
    header = getattr(msg, "header", None)
    if header is not None and hasattr(header, "stamp"):
        stamp = header.stamp
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    return None


def _planning_factor_count(msg: Any) -> int:
    for attr in ("factors", "planning_factors"):
        items = getattr(msg, attr, None)
        if items is not None:
            return len(items)
    return 0


def analyze_bag(bag_dir: Path, label: str | None = None) -> BagMetrics:
    if not bag_dir.is_dir():
        raise FileNotFoundError(f"bag directory not found: {bag_dir}")

    metrics = BagMetrics(label=label or bag_dir.name, path=bag_dir)
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=str(bag_dir), storage_id="sqlite3"),
        ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        ),
    )

    type_map = {meta.name: meta.type for meta in reader.get_all_topics_and_types()}
    msg_classes: dict[str, Any] = {}

    prev_x: float | None = None
    prev_y: float | None = None

    while reader.has_next():
        topic, raw, _t = reader.read_next()
        metrics.topic_counts[topic] = metrics.topic_counts.get(topic, 0) + 1

        if topic not in msg_classes:
            msg_classes[topic] = get_message(type_map[topic])
        msg = deserialize_message(raw, msg_classes[topic])

        stamp_ns = _stamp_ns(msg)
        if stamp_ns is not None:
            metrics.note_stamp(stamp_ns)

        if topic == "/vehicle/status/velocity_status":
            v = float(msg.longitudinal_velocity)
            metrics.velocity.add(v)
            if abs(v) < 0.05:
                metrics.stopped_samples += 1

        elif topic == "/perception/object_recognition/tracking/objects":
            n = len(msg.objects)
            metrics.tracked_count.add(float(n))
            if n > 0:
                metrics.frames_with_tracked += 1

        elif topic == "/perception/object_recognition/objects":
            n = len(msg.objects)
            metrics.predicted_count.add(float(n))
            if n > 0:
                metrics.frames_with_predicted += 1

        elif topic == "/perception/object_recognition/detection/centerpoint/validation/objects":
            n = len(msg.objects)
            metrics.detection_count.add(float(n))
            if n > 0:
                metrics.frames_with_detection += 1

        elif topic == "/planning/planning_factors/obstacle_stop":
            metrics.obstacle_stop_msgs += 1
            if _planning_factor_count(msg) > 0:
                metrics.obstacle_stop_with_factors += 1

        elif topic == "/planning/scenario_planning/trajectory":
            for pt in msg.points:
                v = float(pt.longitudinal_velocity_mps)
                metrics.min_traj_velocity = (
                    v
                    if metrics.min_traj_velocity is None
                    else min(metrics.min_traj_velocity, v)
                )

        elif topic == "/localization/pose_with_covariance":
            x = float(msg.pose.pose.position.x)
            y = float(msg.pose.pose.position.y)
            if prev_x is not None and prev_y is not None:
                dx = x - prev_x
                dy = y - prev_y
                metrics.distance_m += (dx * dx + dy * dy) ** 0.5
            prev_x, prev_y = x, y

    metrics.finalize()
    return metrics


def _fmt(v: float | None, digits: int = 2) -> str:
    if v is None:
        return "-"
    return f"{v:.{digits}f}"


def _delta(a: float | None, b: float | None, digits: int = 2) -> str:
    if a is None or b is None:
        return "-"
    sign = "+" if b - a >= 0 else ""
    return f"{sign}{b - a:.{digits}f}"


def _pct(part: int, whole: int) -> str:
    if whole == 0:
        return "-"
    return f"{100.0 * part / whole:.1f}%"


def print_comparison(baseline: BagMetrics, treatment: BagMetrics) -> None:
    print("=" * 72)
    print("Experiment bag comparison")
    print("=" * 72)
    print(f"Baseline (OFF): {baseline.label}  ({baseline.path})")
    print(f"Dsgn ON:        {treatment.label}  ({treatment.path})")
    print()

    rows: list[tuple[str, str, str, str]] = [
        ("Duration (s)", _fmt(baseline.duration_s), _fmt(treatment.duration_s), _delta(baseline.duration_s, treatment.duration_s)),
        ("Distance traveled (m)", _fmt(baseline.distance_m), _fmt(treatment.distance_m), _delta(baseline.distance_m, treatment.distance_m)),
        ("Mean speed (m/s)", _fmt(baseline.velocity.mean), _fmt(treatment.velocity.mean), _delta(baseline.velocity.mean, treatment.velocity.mean)),
        ("Min speed (m/s)", _fmt(baseline.velocity.min_val), _fmt(treatment.velocity.min_val), _delta(baseline.velocity.min_val, treatment.velocity.min_val)),
        ("Max speed (m/s)", _fmt(baseline.velocity.max_val), _fmt(treatment.velocity.max_val), _delta(baseline.velocity.max_val, treatment.velocity.max_val)),
        (
            "Samples near stop (|v|<0.05)",
            str(baseline.stopped_samples),
            str(treatment.stopped_samples),
            _delta(float(baseline.stopped_samples), float(treatment.stopped_samples), 0),
        ),
        ("Max tracked objects / frame", _fmt(baseline.tracked_count.max_val, 0), _fmt(treatment.tracked_count.max_val, 0), _delta(baseline.tracked_count.max_val, treatment.tracked_count.max_val, 0)),
        ("Mean tracked objects / frame", _fmt(baseline.tracked_count.mean), _fmt(treatment.tracked_count.mean), _delta(baseline.tracked_count.mean, treatment.tracked_count.mean)),
        (
            "Frames with tracked objects",
            f"{baseline.frames_with_tracked} ({_pct(baseline.frames_with_tracked, baseline.tracked_count.count)})",
            f"{treatment.frames_with_tracked} ({_pct(treatment.frames_with_tracked, treatment.tracked_count.count)})",
            "-",
        ),
        ("Max predicted objects / frame", _fmt(baseline.predicted_count.max_val, 0), _fmt(treatment.predicted_count.max_val, 0), _delta(baseline.predicted_count.max_val, treatment.predicted_count.max_val, 0)),
        ("Mean predicted objects / frame", _fmt(baseline.predicted_count.mean), _fmt(treatment.predicted_count.mean), _delta(baseline.predicted_count.mean, treatment.predicted_count.mean)),
        (
            "Frames with predicted objects",
            f"{baseline.frames_with_predicted} ({_pct(baseline.frames_with_predicted, baseline.predicted_count.count)})",
            f"{treatment.frames_with_predicted} ({_pct(treatment.frames_with_predicted, treatment.predicted_count.count)})",
            "-",
        ),
        ("Max detections / frame (centerpoint/validation)", _fmt(baseline.detection_count.max_val, 0), _fmt(treatment.detection_count.max_val, 0), _delta(baseline.detection_count.max_val, treatment.detection_count.max_val, 0)),
        ("Mean detections / frame", _fmt(baseline.detection_count.mean), _fmt(treatment.detection_count.mean), _delta(baseline.detection_count.mean, treatment.detection_count.mean)),
        (
            "Obstacle-stop msgs (factors>0)",
            str(baseline.obstacle_stop_with_factors),
            str(treatment.obstacle_stop_with_factors),
            _delta(float(baseline.obstacle_stop_with_factors), float(treatment.obstacle_stop_with_factors), 0),
        ),
        ("Min planned traj velocity (m/s)", _fmt(baseline.min_traj_velocity), _fmt(treatment.min_traj_velocity), _delta(baseline.min_traj_velocity, treatment.min_traj_velocity)),
    ]

    print(f"{'Metric':<42} {'Baseline':>10} {'Dsgn ON':>10} {'Delta':>8}")
    print("-" * 72)
    for name, b, t, d in rows:
        print(f"{name:<42} {b:>10} {t:>10} {d:>8}")

    print()
    print("Interpretation hints:")
    print("  - More tracked/predicted objects in Dsgn ON → fake detections reached planners.")
    print("  - Higher obstacle-stop count + lower mean speed → planner reacted to objects.")
    print("  - Subscriber graph is LIVE-only; run audit_object_topic_subscribers.sh in Docker.")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare baseline vs dsgn-on experiment bags.")
    parser.add_argument("baseline", type=Path, help="baseline bag directory")
    parser.add_argument("treatment", type=Path, help="dsgn-on bag directory")
    parser.add_argument("--baseline-label", default=None)
    parser.add_argument("--treatment-label", default=None)
    args = parser.parse_args()

    print("Reading baseline bag...", file=sys.stderr)
    baseline = analyze_bag(args.baseline, args.baseline_label)
    print("Reading dsgn-on bag...", file=sys.stderr)
    treatment = analyze_bag(args.treatment, args.treatment_label)
    print_comparison(baseline, treatment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
