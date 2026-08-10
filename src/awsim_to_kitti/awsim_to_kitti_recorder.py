#!/usr/bin/env python3
"""Record a KITTI-layout stereo dataset from modded AWSIM + Autoware.

WHY THIS EXISTS
Arka's recorder (github.com/DF-Autoware-AWSIM/AWSIM_to_KITTI) established the
layout our DSGN tooling expects, but it hardcodes his paths, reads the point
cloud one point at a time, and subscribes to a left topic our stock AWSIM does
not publish. This is a rewrite that keeps his conventions exactly -- same
directory names, same %06d numbering, same pose line format -- so datasets
recorded here stay comparable with dsgn/datasets/arka/.

WHAT IT SAVES
Only what the simulator cannot reproduce later:

  image_2/%06d.png     left  (stock AWSIM camera)
  image_3/%06d.png     right (StereoMod clone, +0.54 m baseline)
  velodyne/%06d.bin    float32 x,y,z,intensity   (--no-velodyne to skip)
  pose/path.txt        "%06d x y yaw" per frame, Arka's format
  meta/run.json        topics, args, both camera_info, git commit
  meta/frames.ndjson   per-frame stamps, full ego pose, CenterPoint objects

calib/ and label_2/ are deliberately NOT written here. Both are derivable from
meta/ offline, and both have open questions (Arka's R0_rect and Tr_velo_to_cam
are copied from real KITTI and do not describe this rig). Generating them in
post means a bad choice costs a re-run of a script, not a re-drive.

GOTCHAS THIS HANDLES
- /sensing/camera/traffic_light/image_raw has TWO publishers in a full Autoware
  stack: AWSIM and traffic_light_image_decompressor. Both carry the same header
  stamp, so frames are deduplicated on (left stamp).
- AWSIM sensor topics are BEST_EFFORT. A RELIABLE subscription silently
  receives nothing.
- PNG encoding two 1920x1080 frames at 10 Hz does not fit in a ROS callback, so
  encoding happens on a worker pool behind a bounded queue. If the pool cannot
  keep up we drop frames and say so, rather than growing memory without bound.

Usage (inside a container on ROS_DOMAIN_ID=26):
  python3 awsim_to_kitti_recorder.py --out /out/run_001 --max-frames 100
"""
from __future__ import annotations

import argparse
import json
import math
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseWithCovarianceStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sensor_msgs_py import point_cloud2

try:
    from autoware_perception_msgs.msg import DetectedObjects
except ImportError:  # perception stack not sourced; objects are optional
    DetectedObjects = None

LEFT_TOPIC = "/sensing/camera/traffic_light/image_raw"
RIGHT_TOPIC = "/sensing/camera_right/traffic_light/image_raw"
LIDAR_TOPIC = "/sensing/lidar/top/pointcloud_raw"
OBJECTS_TOPIC = "/perception/object_recognition/detection/centerpoint/objects"
POSE_TOPIC = "/localization/pose_with_covariance"


def stamp_key(msg) -> int:
    s = msg.header.stamp
    return s.sec * 1_000_000_000 + s.nanosec


def yaw_from_quaternion(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def git_commit() -> str | None:
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        out = subprocess.run(
            ["git", "-C", here, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


class FrameJob:
    __slots__ = ("index", "left", "right", "lidar")

    def __init__(self, index, left, right, lidar):
        self.index = index
        self.left = left
        self.right = right
        self.lidar = lidar


class Recorder(Node):
    def __init__(self, args):
        super().__init__("awsim_to_kitti_recorder")
        self.args = args
        self.out = args.out
        self.index = args.start_index
        self.written = 0
        self.dropped = 0
        self.duplicates = 0
        self.last_left_stamp = None
        self.latest_pose = None
        self.latest_objects = None
        self.camera_info = {}
        self.t_start = time.time()
        self._last_report_t = self.t_start
        self._last_report_n = 0
        self._lock = threading.Lock()

        self.dirs = {
            "image_2": os.path.join(self.out, "image_2"),
            "image_3": os.path.join(self.out, "image_3"),
            "pose": os.path.join(self.out, "pose"),
            "meta": os.path.join(self.out, "meta"),
        }
        if not args.no_velodyne:
            self.dirs["velodyne"] = os.path.join(self.out, "velodyne")
        for d in self.dirs.values():
            os.makedirs(d, exist_ok=True)

        # Line-buffered so Ctrl+C / kill cannot leave path.txt empty while
        # frames.ndjson (which fills the 8 KiB block buffer faster) looks fine.
        self.path_txt = open(
            os.path.join(self.dirs["pose"], "path.txt"), "a", buffering=1
        )
        self.frames_ndjson = open(
            os.path.join(self.dirs["meta"], "frames.ndjson"), "a", buffering=1
        )
        self._finished = False
        self._stop = False

        self.jobs: queue.Queue[FrameJob | None] = queue.Queue(maxsize=args.queue)
        self.workers = [
            threading.Thread(target=self._writer_loop, daemon=True, name=f"w{i}")
            for i in range(args.workers)
        ]
        for w in self.workers:
            w.start()

        subs = [
            Subscriber(self, Image, args.left_topic, qos_profile=qos_profile_sensor_data),
            Subscriber(self, Image, args.right_topic, qos_profile=qos_profile_sensor_data),
        ]
        if not args.no_velodyne:
            subs.append(
                Subscriber(self, PointCloud2, args.lidar_topic, qos_profile=qos_profile_sensor_data)
            )
        self.sync = ApproximateTimeSynchronizer(subs, queue_size=args.sync_queue, slop=args.slop)
        self.sync.registerCallback(self._on_frame)

        self.create_subscription(
            PoseWithCovarianceStamped, args.pose_topic, self._on_pose, 10
        )
        if DetectedObjects is not None:
            self.create_subscription(
                DetectedObjects, args.objects_topic, self._on_objects, 10
            )
        for side, topic in (("left", args.left_topic), ("right", args.right_topic)):
            info_topic = topic.rsplit("/", 1)[0] + "/camera_info"
            self.create_subscription(
                CameraInfo, info_topic,
                lambda m, s=side, t=info_topic: self._on_camera_info(s, t, m),
                qos_profile_sensor_data,
            )

        self.create_timer(5.0, self._report)

    def _on_pose(self, msg):
        self.latest_pose = msg

    def _on_objects(self, msg):
        self.latest_objects = msg

    def _on_camera_info(self, side, topic, msg):
        if side in self.camera_info:
            return
        self.camera_info[side] = {
            "topic": topic,
            "frame_id": msg.header.frame_id,
            "width": msg.width,
            "height": msg.height,
            "K": list(msg.k),
            "P": list(msg.p),
            "R": list(msg.r),
            "D": list(msg.d),
            "distortion_model": msg.distortion_model,
        }

    def _on_frame(self, left, right, lidar=None):
        key = stamp_key(left)
        if key == self.last_left_stamp:
            self.duplicates += 1
            return
        self.last_left_stamp = key

        if self.args.max_frames and self.written + self.jobs.qsize() >= self.args.max_frames:
            return

        idx = self.index
        self.index += 1
        try:
            self.jobs.put_nowait(FrameJob(idx, left, right, lidar))
        except queue.Full:
            self.dropped += 1
            self.index -= 1
            return

        self._write_sidecar(idx, left, right, lidar)

    def _write_sidecar(self, idx, left, right, lidar):
        record = {
            "frame": f"{idx:06d}",
            "stamp_left": stamp_key(left),
            "stamp_right": stamp_key(right),
            "stamp_lidar": stamp_key(lidar) if lidar is not None else None,
            "wall_time": datetime.now(timezone.utc).isoformat(),
        }
        if self.latest_pose is not None:
            p = self.latest_pose.pose.pose.position
            o = self.latest_pose.pose.pose.orientation
            yaw = yaw_from_quaternion(o)
            record["ego"] = {
                "stamp": stamp_key(self.latest_pose),
                "position": [p.x, p.y, p.z],
                "orientation": [o.x, o.y, o.z, o.w],
                "yaw": yaw,
            }
            with self._lock:
                self.path_txt.write(f"{idx:06d} {p.x:.6f} {p.y:.6f} {yaw:.6f}\n")
                self.path_txt.flush()
        if self.latest_objects is not None:
            record["centerpoint"] = self._objects_to_json(self.latest_objects)

        with self._lock:
            self.frames_ndjson.write(json.dumps(record) + "\n")
            self.frames_ndjson.flush()

    @staticmethod
    def _objects_to_json(msg):
        out = {"stamp": stamp_key(msg), "frame_id": msg.header.frame_id, "objects": []}
        for obj in msg.objects:
            pose = obj.kinematics.pose_with_covariance.pose
            cls = obj.classification[0] if obj.classification else None
            out["objects"].append({
                "label": int(cls.label) if cls else None,
                "probability": float(cls.probability) if cls else None,
                "existence_probability": float(obj.existence_probability),
                "position": [pose.position.x, pose.position.y, pose.position.z],
                "orientation": [
                    pose.orientation.x, pose.orientation.y,
                    pose.orientation.z, pose.orientation.w,
                ],
                "dimensions": [
                    obj.shape.dimensions.x, obj.shape.dimensions.y, obj.shape.dimensions.z,
                ],
                "shape_type": int(obj.shape.type),
            })
        return out

    def _writer_loop(self):
        bridge = CvBridge()
        png_opts = [cv2.IMWRITE_PNG_COMPRESSION, self.args.png_compression]
        while True:
            job = self.jobs.get()
            if job is None:
                self.jobs.task_done()
                return
            try:
                name = f"{job.index:06d}"
                if not self.args.dry_run:
                    left = bridge.imgmsg_to_cv2(job.left, desired_encoding="bgr8")
                    right = bridge.imgmsg_to_cv2(job.right, desired_encoding="bgr8")
                    cv2.imwrite(os.path.join(self.dirs["image_2"], name + ".png"), left, png_opts)
                    cv2.imwrite(os.path.join(self.dirs["image_3"], name + ".png"), right, png_opts)
                    if job.lidar is not None:
                        self._save_lidar(
                            job.lidar, os.path.join(self.dirs["velodyne"], name + ".bin")
                        )
                with self._lock:
                    self.written += 1
            except Exception as exc:  # keep recording even if one frame is bad
                self.get_logger().error(f"frame {job.index:06d} failed: {exc}")
            finally:
                self.jobs.task_done()

    @staticmethod
    def _save_lidar(msg, path):
        arr = point_cloud2.read_points(
            msg, field_names=("x", "y", "z", "intensity"), skip_nans=True
        )
        pts = np.stack(
            [arr["x"], arr["y"], arr["z"], arr["intensity"]], axis=-1
        ).astype(np.float32)
        pts.tofile(path)

    def _report(self):
        now = time.time()
        window = now - self._last_report_t
        recent = (self.written - self._last_report_n) / window if window > 0 else 0.0
        self._last_report_t, self._last_report_n = now, self.written
        elapsed = now - self.t_start
        mean = self.written / elapsed if elapsed > 0 else 0.0
        self.get_logger().info(
            f"written={self.written} queued={self.jobs.qsize()} "
            f"dropped={self.dropped} dup_skipped={self.duplicates} "
            f"now={recent:.2f} fps mean={mean:.2f} fps"
        )
        if self.args.max_frames and self.written >= self.args.max_frames:
            self._stop = True

    def preflight(self, timeout=15.0):
        """Wait for discovery, then report publisher counts.

        ROS 2 discovery is asynchronous: querying immediately after node
        construction reports zero publishers for topics that plainly exist.
        """
        required = [("left", self.args.left_topic), ("right", self.args.right_topic)]
        if not self.args.no_velodyne:
            required.append(("lidar", self.args.lidar_topic))
        optional = [("pose", self.args.pose_topic), ("objects", self.args.objects_topic)]

        deadline = time.time() + timeout
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.2)
            if all(self.count_publishers(t) > 0 for _, t in required + optional):
                break

        print("preflight:")
        ok = True
        for label, topic in required + optional:
            n = self.count_publishers(topic)
            required_here = any(topic == t for _, t in required)
            flag = ""
            if n == 0:
                flag = "  MISSING" if required_here else "  (absent, optional)"
                if required_here:
                    ok = False
            print(f"  {label:8s} {topic}  publishers={n}{flag}")
            if n > 1:
                print(f"           note: {n} publishers, deduplicating on header stamp")
        self.t_start = time.time()
        self._last_report_t = self.t_start
        self._last_report_n = 0
        return ok

    def finish(self):
        if self._finished:
            return
        self._finished = True
        for _ in self.workers:
            self.jobs.put(None)
        for w in self.workers:
            w.join(timeout=60)
        try:
            self.path_txt.flush()
            self.path_txt.close()
        except Exception:
            pass
        try:
            self.frames_ndjson.flush()
            self.frames_ndjson.close()
        except Exception:
            pass

        elapsed = time.time() - self.t_start
        summary = {
            "created": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit(),
            "elapsed_s": round(elapsed, 2),
            "frames_written": self.written,
            "frames_dropped": self.dropped,
            "duplicates_skipped": self.duplicates,
            "mean_fps": round(self.written / elapsed, 3) if elapsed > 0 else None,
            "topics": {
                "left": self.args.left_topic,
                "right": self.args.right_topic,
                "lidar": None if self.args.no_velodyne else self.args.lidar_topic,
                "pose": self.args.pose_topic,
                "objects": self.args.objects_topic,
            },
            "args": {k: (str(v) if isinstance(v, os.PathLike) else v)
                     for k, v in vars(self.args).items()},
            "camera_info": self.camera_info,
        }
        with open(os.path.join(self.dirs["meta"], "run.json"), "w") as f:
            json.dump(summary, f, indent=2)

        print()
        print(f"wrote {self.written} frames to {self.out}")
        print(f"  dropped={self.dropped}  dup_skipped={self.duplicates}  "
              f"mean={summary['mean_fps']} fps")
        if not self.camera_info:
            print("  WARNING: no camera_info captured; calib generation will need defaults")
        elif len(self.camera_info) < 2:
            print(f"  WARNING: camera_info for only {list(self.camera_info)}")


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out", required=True, help="Dataset root directory")
    p.add_argument("--left-topic", default=LEFT_TOPIC)
    p.add_argument("--right-topic", default=RIGHT_TOPIC)
    p.add_argument("--lidar-topic", default=LIDAR_TOPIC)
    p.add_argument("--objects-topic", default=OBJECTS_TOPIC)
    p.add_argument("--pose-topic", default=POSE_TOPIC)
    p.add_argument("--no-velodyne", action="store_true",
                   help="Skip point clouds (training-only sets do not need them)")
    p.add_argument("--slop", type=float, default=0.05,
                   help="ApproximateTimeSynchronizer tolerance in seconds")
    p.add_argument("--sync-queue", type=int, default=10)
    p.add_argument("--max-frames", type=int, default=0, help="0 = until Ctrl+C")
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--workers", type=int, default=4, help="PNG encoder threads")
    p.add_argument("--queue", type=int, default=32,
                   help="Bounded job queue; frames are dropped when full")
    p.add_argument("--png-compression", type=int, default=1, choices=range(0, 10),
                   metavar="0-9", help="0 fastest/largest, 9 slowest/smallest")
    p.add_argument("--skip-preflight", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="Sync and count frames but write no images/clouds; "
                        "isolates synchroniser cost from encoding cost")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    rclpy.init()
    node = Recorder(args)
    # `docker stop` sends SIGTERM; without this the process dies before
    # finish() flushes path.txt and writes run.json.
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))

    exit_code = 0
    try:
        ok = True
        if not args.skip_preflight:
            ok = node.preflight()
            if not ok:
                print("\nerror: a required topic has no publisher; is AWSIM running?",
                      file=sys.stderr)
                exit_code = 1
        if ok:
            print(f"\nrecording to {args.out} "
                  f"(Ctrl+C or `docker stop awsim_recorder` to stop)")
            while rclpy.ok() and not node._stop:
                rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        # Always flush path.txt / frames.ndjson / run.json — including when
        # Ctrl+C lands during preflight (previous bug: empty path.txt).
        node.finish()
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
