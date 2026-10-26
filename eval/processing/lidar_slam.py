#!/usr/bin/env python3
"""Best-estimate reference trajectory from the bag's 3D lidar, with KISS-SLAM: KISS-ICP odometry +
loop closure + pose-graph optimisation (plan/evaluation_plan_stage_1.md E2.3, ablation plan R*).

usage: lidar_slam.py BAG_DIR OUT_DIR [--topic /lidar_points] [--max-range 60] [--min-range 0.5]
                     [--voxel V] [--split-dist 10] [--map-voxel 0.25] [--no-deskew]

Same scan handling as fuse_reference.py (mid-scan times, NaN-safe per-point times). The camera is
never used, so the result is independent of the ORB-SLAM3 runs it scores.

Writes OUT_DIR/lidar_poses.txt ('t_ns x y z qx qy qz qw' of the LIDAR frame, one per scan, after the
final optimisation), lidar_odometry_info.txt (read by make_gt.py) and closures.txt.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from kiss_slam.config.config import KissSLAMConfig
from kiss_slam.slam import KissSLAM
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fuse_reference import scan_points  # noqa: E402
from lidar_odometry import R_to_quat_xyzw  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bag", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--topic", default="/lidar_points")
    ap.add_argument("--max-range", type=float, default=40.0)
    ap.add_argument("--min-range", type=float, default=1.0)
    ap.add_argument("--voxel", type=float, help="odometry voxel size [m]; default max_range/100")
    ap.add_argument("--split-dist", type=float, default=10.0,
                    help="metres of travel per local map (loop candidates are checked per local map)")
    ap.add_argument("--map-voxel", type=float, default=0.25, help="local-map voxel for loop detection [m]")
    ap.add_argument("--no-deskew", action="store_true")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    cfg = KissSLAMConfig()
    pre = cfg.odometry.preprocessing
    pre.max_range, pre.min_range, pre.deskew = a.max_range, a.min_range, not a.no_deskew
    cfg.odometry.mapping.voxel_size = a.voxel if a.voxel else a.max_range / 100.0
    cfg.local_mapper.splitting_distance = a.split_dist
    cfg.local_mapper.voxel_size = a.map_voxel
    slam = KissSLAM(cfg)

    stamps, npts, t0, frame, tfield = [], [], time.time(), None, None
    ts = get_typestore(Stores.ROS2_HUMBLE)
    with AnyReader([a.bag], default_typestore=ts) as r:
        conn = [c for c in r.connections if c.topic == a.topic]
        total = conn[0].msgcount
        for i, (c, _, raw) in enumerate(r.messages(connections=conn)):
            m = r.deserialize(raw, c.msgtype)
            frame = frame or m.header.frame_id
            p, pt, s_mid, _ = scan_points(m)
            tfield = next((n for n in ("t", "timestamp", "time") if n in [f.name for f in m.fields]), None)
            slam.process_scan(p, pt if pre.deskew else np.array([]))
            stamps.append(s_mid)
            npts.append(len(p))
            if i % 200 == 0:
                print("  scan %d/%d  pts %d  closures %d" % (i, total, len(p), len(slam.closures)), flush=True)

    # Finish exactly as kiss_slam/pipeline.py does: close the last local map, then optimise every pose.
    slam.generate_new_node()
    slam.local_map_graph.erase_last_local_map()
    poses, _ = slam.fine_grained_optimization()
    poses = np.array(poses)
    if len(poses) != len(stamps):
        raise SystemExit("pose count %d != scan count %d" % (len(poses), len(stamps)))

    rows = ["%d %.6f %.6f %.6f %.9f %.9f %.9f %.9f" % (s, *T[:3, 3], *R_to_quat_xyzw(T[:3, :3]))
            for s, T in zip(stamps, poses)]
    (a.out / "lidar_poses.txt").write_text("\n".join(rows) + "\n")
    (a.out / "closures.txt").write_text("".join("%d %d\n" % c for c in slam.closures))
    P = poses[:, :3, 3]
    info = ("method: kiss-slam (loop closure + pose graph)\ntopic: %s\nframe: %s\nscans: %d\n"
            "points_per_scan_median: %d\nper_point_time_field: %s\ndeskew: %s\nmax_range: %.1f\nmin_range: %.1f\n"
            "voxel_size: %.3f\nsplit_dist_m: %.1f\nmap_voxel: %.3f\nclosures: %d\npath_length_m: %.2f\n"
            "end_to_start_m: %.3f\nz_range_m: %.3f\nruntime_s: %.0f\n"
            % (a.topic, frame, len(rows), int(np.median(npts)), tfield, pre.deskew, a.max_range, a.min_range,
               cfg.odometry.mapping.voxel_size, a.split_dist, a.map_voxel, len(slam.closures),
               np.linalg.norm(np.diff(P, axis=0), axis=1).sum(), np.linalg.norm(P[-1] - P[0]),
               P[:, 2].max() - P[:, 2].min(), time.time() - t0))
    (a.out / "lidar_odometry_info.txt").write_text(info)
    print(info)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
