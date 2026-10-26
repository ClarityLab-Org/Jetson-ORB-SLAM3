#!/usr/bin/env python3
"""Reference poses of the LEFT IR optical frame at every image time (plan/evaluation_plan_stage_1.md, E2.3).

usage: make_gt.py SEQ_DIR LIDAR_POSES [--lidar-frame F] [--cam-frame F] [--time-offset-ms D] [--no-lever-arm]

  SEQ_DIR      output of bag_to_euroc.py (times.txt, tf_static.json, calib.json)
  LIDAR_POSES  output of lidar_odometry.py ('t_ns x y z qx qy qz qw' of the lidar frame)

T_world_cam(t) = T_world_lidar(t) * T_lidar_cam, with T_lidar_cam chained from /tf_static.
Positions are interpolated linearly, rotations by SLERP, at each image timestamp (+ offset); no
extrapolation, and image times more than 0.2 s from a lidar pose on either side are skipped.
Writes SEQ_DIR/gt.csv in the EuRoC ground-truth format eval/benchmark/evaluate.py reads
('#timestamp[ns],p_x,p_y,p_z,q_w,q_x,q_y,q_z') and appends the transform used to source_info.txt.
"""
from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np


def q2R(q):  # xyzw
    x, y, z, w = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def R2q(R):  # -> xyzw
    w = np.sqrt(max(0.0, 1 + np.trace(R))) / 2
    x = np.copysign(np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def T(t, q):
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = q2R(q), t
    return M


def slerp(q0, q1, u):
    q0, q1 = q0 / np.linalg.norm(q0), q1 / np.linalg.norm(q1)
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + u * (q1 - q0)
        return q / np.linalg.norm(q)
    th = np.arccos(d)
    return (np.sin((1 - u) * th) * q0 + np.sin(u * th) * q1) / np.sin(th)


def chain(tf: list, a: str, b: str) -> np.ndarray:
    """T_a_b (pose of frame b in frame a) through the static TF graph, walking edges either way."""
    adj = {}
    for e in tf:
        M = T(e["t"], e["q_xyzw"])                       # T_parent_child
        adj.setdefault(e["parent"], []).append((e["child"], M))
        adj.setdefault(e["child"], []).append((e["parent"], np.linalg.inv(M)))
    q, seen = deque([(a, np.eye(4), [a])]), {a}
    while q:
        f, M, path = q.popleft()
        if f == b:
            print("  TF chain: " + " -> ".join(path))
            return M
        for g, E in adj.get(f, []):
            if g not in seen:
                seen.add(g)
                q.append((g, M @ E, path + [g]))
    raise SystemExit("no static TF path %s -> %s. Frames known: %s" % (a, b, sorted(adj)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("seq", type=Path)
    ap.add_argument("lidar_poses", type=Path)
    ap.add_argument("--lidar-frame", help="default: from lidar_odometry_info.txt next to LIDAR_POSES")
    ap.add_argument("--cam-frame", help="default: the infra1 camera_info frame from calib.json")
    ap.add_argument("--time-offset-ms", type=float, default=0.0, help="added to image times before lookup")
    ap.add_argument("--no-lever-arm", action="store_true", help="diagnostic E7.2: use lidar poses as-is")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    info = a.lidar_poses.parent / "lidar_odometry_info.txt"
    lidar_frame = a.lidar_frame or dict(l.split(": ", 1) for l in info.read_text().splitlines())["frame"]
    cam_frame = a.cam_frame or json.loads((a.seq / "calib.json").read_text())["left"]["frame"]
    tf = json.loads((a.seq / "tf_static.json").read_text())
    T_lc = np.eye(4) if a.no_lever_arm else chain(tf, lidar_frame, cam_frame)
    print("  T_%s_%s: t=%s m  |t|=%.3f m" % (lidar_frame, cam_frame, np.round(T_lc[:3, 3], 4), np.linalg.norm(T_lc[:3, 3])))

    L = np.loadtxt(a.lidar_poses)
    lt = L[:, 0].astype(np.int64)
    lp, lq = L[:, 1:4], L[:, 4:8]
    times = np.loadtxt(a.seq / "times.txt", dtype=np.int64, ndmin=1)
    off = int(round(a.time_offset_ms * 1e6))
    out, skipped = [], 0
    for s in times:
        q = s + off
        i = np.searchsorted(lt, q)
        if i == 0 or i >= len(lt) or (q - lt[i - 1]) > 2e8 or (lt[i] - q) > 2e8:
            skipped += 1
            continue
        u = (q - lt[i - 1]) / (lt[i] - lt[i - 1])
        Twl = T((1 - u) * lp[i - 1] + u * lp[i], slerp(lq[i - 1], lq[i], u))
        Twc = Twl @ T_lc
        qx, qy, qz, qw = R2q(Twc[:3, :3])
        out.append("%d,%.6f,%.6f,%.6f,%.9f,%.9f,%.9f,%.9f" % (s, *Twc[:3, 3], qw, qx, qy, qz))
    dst = a.out or (a.seq / "gt.csv")
    dst.write_text("#timestamp[ns],p_x,p_y,p_z,q_w,q_x,q_y,q_z\n" + "\n".join(out) + "\n")
    msg = ("gt_file: %s\ngt_poses: %d\ngt_skipped_images: %d\ngt_coverage_pct: %.1f\ngt_time_offset_ms: %.1f\n"
           "gt_lever_arm: %s\nT_lidar_cam_t: %s\n" % (dst.name, len(out), skipped, 100 * len(out) / len(times),
                                                      a.time_offset_ms, not a.no_lever_arm,
                                                      np.round(T_lc[:3, 3], 4).tolist()))
    if a.out is None:
        with open(a.seq / "source_info.txt", "a") as f:
            f.write(msg)
    print(msg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
