#!/usr/bin/env python3
"""Reference trajectory from the bag's raw 3D lidar, with KISS-ICP (plan/evaluation_plan_stage_1.md, E2.3).

usage: lidar_odometry.py BAG_DIR OUT_DIR [--topic /lidar_points] [--max-range 60] [--min-range 0.5]
                         [--voxel V] [--no-deskew]

The bag carries the raw scans but no lidar-SLAM trajectory, so we compute one. KISS-ICP's own
runner stamps poses with the bag *receive* time; this driver uses the scans' *header* stamps
(the sensor time, the same clock as the camera headers) and drops NaN / zero points together
with their per-point times.

Pose times are the middle of each scan's per-point time span (KISS-ICP deskews to mid-scan), the
same convention as fuse_reference.py.
Writes OUT_DIR/lidar_poses.txt: 't_ns x y z qx qy qz qw' of the LIDAR frame in the lidar odometry
world (first scan = identity), plus lidar_odometry_info.txt. KISS-ICP has no loop closure.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from kiss_icp.config import load_config
from kiss_icp.kiss_icp import KissICP
from kiss_icp.tools.point_cloud2 import read_points
from rosbags.highlevel import AnyReader
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fuse_reference import scan_points  # noqa: E402  (same mid-scan time convention as the fusion)
from rosbags.typesys import Stores, get_typestore


def R_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2
    y = np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2
    z = np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2
    x = np.copysign(x, R[2, 1] - R[1, 2])
    y = np.copysign(y, R[0, 2] - R[2, 0])
    z = np.copysign(z, R[1, 0] - R[0, 1])
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def scan(msg):
    names = [f.name for f in msg.fields]
    tf = next((n for n in ("t", "timestamp", "time") if n in names), None)
    s = read_points(msg, field_names=["x", "y", "z"] + ([tf] if tf else []))
    p = np.column_stack([s["x"], s["y"], s["z"]]).astype(np.float64)
    ok = np.all(np.isfinite(p), axis=1) & (np.abs(p).sum(1) > 1e-6)
    t = s[tf].astype(np.float64)[ok] if tf else np.array([])
    return p[ok], t, tf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bag", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--topic", default="/lidar_points")
    ap.add_argument("--max-range", type=float, default=40.0)
    ap.add_argument("--min-range", type=float, default=1.0)
    ap.add_argument("--voxel", type=float, help="map voxel size [m]; default max_range/100")
    ap.add_argument("--no-deskew", action="store_true")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    cfg = load_config(None)
    cfg.data.max_range, cfg.data.min_range = a.max_range, a.min_range
    cfg.data.deskew = not a.no_deskew
    cfg.mapping.voxel_size = a.voxel if a.voxel else a.max_range / 100.0
    odo = KissICP(cfg)

    rows, npts, t0 = [], [], time.time()
    ts = get_typestore(Stores.ROS2_HUMBLE)
    with AnyReader([a.bag], default_typestore=ts) as r:
        conn = [c for c in r.connections if c.topic == a.topic]
        total = conn[0].msgcount
        frame, tfield = None, None
        for i, (c, _, raw) in enumerate(r.messages(connections=conn)):
            m = r.deserialize(raw, c.msgtype)
            frame = frame or m.header.frame_id
            p, pt, s, _ = scan_points(m)
            tfield = next((n for n in ("t", "timestamp", "time") if n in [f.name for f in m.fields]), None)
            if not cfg.data.deskew:
                pt = np.array([])
            odo.register_frame(p, pt)
            T = odo.last_pose
            rows.append("%d %.6f %.6f %.6f %.9f %.9f %.9f %.9f" % ((s, *T[:3, 3], *R_to_quat_xyzw(T[:3, :3]))))
            npts.append(len(p))
            if i % 200 == 0:
                print("  scan %d/%d  pts %d  pos (%.2f %.2f %.2f)" % (i, total, len(p), *T[:3, 3]), flush=True)

    (a.out / "lidar_poses.txt").write_text("\n".join(rows) + "\n")
    P = np.array([[float(v) for v in l.split()[1:4]] for l in rows])
    info = ("topic: %s\nframe: %s\nscans: %d\npoints_per_scan_median: %d\nper_point_time_field: %s\ndeskew: %s\n"
            "max_range: %.1f\nmin_range: %.1f\nvoxel_size: %.3f\npath_length_m: %.2f\nend_to_start_m: %.3f\n"
            "runtime_s: %.0f\n" % (a.topic, frame, len(rows), int(np.median(npts)), tfield, cfg.data.deskew,
                                   a.max_range, a.min_range, cfg.mapping.voxel_size,
                                   np.linalg.norm(np.diff(P, axis=0), axis=1).sum(), np.linalg.norm(P[-1] - P[0]),
                                   time.time() - t0))
    (a.out / "lidar_odometry_info.txt").write_text(info)
    print(info)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
