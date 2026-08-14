#!/usr/bin/env python3
"""Check AWSIM stereo pair geometry (run inside Docker). Usage: awsim_stereo_check.py --outdir DIR"""

import argparse
import sys

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

LEFT_TOPIC = "/sensing/camera/traffic_light/image_raw"
RIGHT_TOPIC = "/sensing/camera_right/traffic_light/image_raw"

FX = 960.0          # from live camera_info, matches Arka's calib
BASELINE_M = 0.54


class Grabber(Node):
    def __init__(self):
        super().__init__("awsim_stereo_check")
        # AWSIM publishes images BEST_EFFORT; a RELIABLE subscriber silently gets nothing.
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.bridge = CvBridge()
        self.pair = None
        sync = ApproximateTimeSynchronizer(
            [Subscriber(self, __import__("sensor_msgs.msg", fromlist=["Image"]).Image, LEFT_TOPIC, qos_profile=qos),
             Subscriber(self, __import__("sensor_msgs.msg", fromlist=["Image"]).Image, RIGHT_TOPIC, qos_profile=qos)],
            queue_size=10, slop=0.02)
        sync.registerCallback(self.cb)

    def cb(self, left_msg, right_msg):
        if self.pair is not None:
            return
        dt = abs((left_msg.header.stamp.sec - right_msg.header.stamp.sec)
                 + (left_msg.header.stamp.nanosec - right_msg.header.stamp.nanosec) * 1e-9)
        self.pair = (
            self.bridge.imgmsg_to_cv2(left_msg, "bgr8"),
            self.bridge.imgmsg_to_cv2(right_msg, "bgr8"),
            dt,
        )


def epipolar_and_disparity(left, right):
    """Match features and report vertical error + disparity, in pixels."""
    gl = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    gr = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=4000)
    kl, dl = orb.detectAndCompute(gl, None)
    kr, dr = orb.detectAndCompute(gr, None)
    if dl is None or dr is None or len(kl) < 20 or len(kr) < 20:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw = matcher.knnMatch(dl, dr, k=2)
    good = [m for m, n in raw if m.distance < 0.75 * n.distance]
    if len(good) < 20:
        return None

    dx, dy = [], []
    for m in good:
        pl, pr = kl[m.queryIdx].pt, kr[m.trainIdx].pt
        dx.append(pl[0] - pr[0])   # positive if left image is shifted right => correct order
        dy.append(pl[1] - pr[1])
    dx, dy = np.array(dx), np.array(dy)

    # Keep matches consistent with a rectified pair before summarising disparity.
    keep = np.abs(dy) < 2.0
    return {
        "n_matches": len(good),
        "median_abs_dy": float(np.median(np.abs(dy))),
        "frac_dy_within_1px": float(np.mean(np.abs(dy) < 1.0)),
        "n_rectified": int(keep.sum()),
        "median_dx": float(np.median(dx[keep])) if keep.sum() else float("nan"),
        "dx_p10": float(np.percentile(dx[keep], 10)) if keep.sum() else float("nan"),
        "dx_p90": float(np.percentile(dx[keep], 90)) if keep.sum() else float("nan"),
    }


def write_visuals(left, right, outdir):
    """Anaglyph + SGBM depth, so the pair can also be eyeballed."""
    # Red-cyan anaglyph: horizontal colour fringing scales with disparity, i.e. with
    # closeness. Vertical fringing would mean the pair is not rectified.
    anaglyph = np.zeros_like(left)
    anaglyph[:, :, 2] = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)    # R <- left
    anaglyph[:, :, 0] = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)   # B <- right
    anaglyph[:, :, 1] = anaglyph[:, :, 0]                         # G <- right
    cv2.imwrite(f"{outdir}/anaglyph.png", anaglyph)

    sgbm = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=192, blockSize=7,
        P1=8 * 3 * 7 ** 2, P2=32 * 3 * 7 ** 2,
        uniquenessRatio=10, speckleWindowSize=100, speckleRange=2,
    )
    disp = sgbm.compute(left, right).astype(np.float32) / 16.0
    vis = np.clip(disp, 0, 96)
    vis = (vis / 96.0 * 255).astype(np.uint8)
    vis = cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)
    vis[disp <= 0] = 0
    cv2.imwrite(f"{outdir}/disparity.png", vis)

    valid = disp[disp > 1.0]
    if valid.size:
        depth = FX * BASELINE_M / valid
        return float(np.median(depth)), float(valid.size) / disp.size
    return float("nan"), 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="/home/aw/logs/stereo")
    ap.add_argument("--timeout", type=float, default=40.0)
    args = ap.parse_args()

    import os
    os.makedirs(args.outdir, exist_ok=True)

    rclpy.init()
    node = Grabber()
    waited = 0.0
    while rclpy.ok() and node.pair is None and waited < args.timeout:
        rclpy.spin_once(node, timeout_sec=0.2)
        waited += 0.2
    if node.pair is None:
        print("FAIL: no synchronized stereo pair received "
              f"within {args.timeout:.0f}s (are both topics publishing?)")
        return 1

    left, right, dt = node.pair
    node.destroy_node()
    rclpy.shutdown()

    cv2.imwrite(f"{args.outdir}/left.png", left)
    cv2.imwrite(f"{args.outdir}/right.png", right)

    diff = float(np.mean(np.abs(left.astype(np.int16) - right.astype(np.int16))))
    print(f"pair timestamp delta : {dt*1000:.2f} ms")
    print(f"image size           : {left.shape[1]}x{left.shape[0]}")
    print(f"mean abs pixel diff  : {diff:.2f}  ({'DIFFERENT views' if diff > 1.0 else 'IDENTICAL - BAD'})")

    stats = epipolar_and_disparity(left, right)
    if stats is None:
        print("WARN: too few feature matches to assess geometry (bland scene?)")
        return 1

    print(f"feature matches      : {stats['n_matches']}")
    print(f"median |dy|          : {stats['median_abs_dy']:.2f} px   "
          f"({'RECTIFIED' if stats['median_abs_dy'] < 1.0 else 'NOT RECTIFIED - BAD'})")
    print(f"matches |dy|<1px     : {stats['frac_dy_within_1px']*100:.1f}%")
    print(f"median disparity     : {stats['median_dx']:.2f} px "
          f"(p10 {stats['dx_p10']:.1f}, p90 {stats['dx_p90']:.1f})")

    if stats["median_dx"] > 0:
        z_med = FX * BASELINE_M / stats["median_dx"]
        z_near = FX * BASELINE_M / stats["dx_p90"] if stats["dx_p90"] > 0 else float("nan")
        z_far = FX * BASELINE_M / stats["dx_p10"] if stats["dx_p10"] > 0 else float("nan")
        print(f"implied depth        : median {z_med:.1f} m  (near {z_near:.1f} m, far {z_far:.1f} m)")
    else:
        print("median disparity is <= 0: left/right are SWAPPED or baseline is wrong")

    z_dense, coverage = write_visuals(left, right, args.outdir)
    print(f"dense SGBM depth     : median {z_dense:.1f} m, {coverage*100:.0f}% of pixels matched")

    ok = (diff > 1.0 and stats["median_abs_dy"] < 1.0 and stats["median_dx"] > 0)
    print("\nRESULT:", "stereo pair is VALID" if ok else "stereo pair is INVALID")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
