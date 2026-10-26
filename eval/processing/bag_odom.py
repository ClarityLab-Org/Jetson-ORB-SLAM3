#!/usr/bin/env python3
"""Extract the robot's own odometry from the bag as a context baseline (not ground truth).

usage: bag_odom.py BAG_DIR OUT_DIR [--topics /platform/odom /platform/odom/filtered]

Writes OUT_DIR/<topic name with / replaced by _>.txt in the same 't_ns x y z qx qy qz qw' format as
lidar_poses.txt (pose of the odometry child frame, header stamps), readable by eval/benchmark/evaluate.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bag", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--topics", nargs="+", default=["/platform/odom", "/platform/odom/filtered"])
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    ts = get_typestore(Stores.ROS2_HUMBLE)
    with AnyReader([a.bag], default_typestore=ts) as r:
        for topic in a.topics:
            conn = [c for c in r.connections if c.topic == topic]
            if not conn:
                print("  missing", topic)
                continue
            rows, frames = [], None
            for c, _, raw in r.messages(connections=conn):
                m = r.deserialize(raw, c.msgtype)
                frames = frames or (m.header.frame_id, m.child_frame_id)
                p, q = m.pose.pose.position, m.pose.pose.orientation
                s = int(m.header.stamp.sec) * 1_000_000_000 + int(m.header.stamp.nanosec)
                rows.append("%d %.6f %.6f %.6f %.9f %.9f %.9f %.9f" % (s, p.x, p.y, p.z, q.x, q.y, q.z, q.w))
            dst = a.out / (topic.strip("/").replace("/", "_") + ".txt")
            dst.write_text("\n".join(rows) + "\n")
            print("  %s: %d poses, %s -> %s  -> %s" % (topic, len(rows), frames[0], frames[1], dst))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
