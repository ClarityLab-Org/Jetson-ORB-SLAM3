#!/usr/bin/env python3
"""Read-only inventory of an external ROS 2 bag (plan/evaluation_plan_stage_1.md, E1.3).

usage: inspect_bag.py BAG_DIR [--samples OUT_DIR]

BAG_DIR is the folder holding metadata.yaml and the .db3. Nothing in it is written: the SQLite file
is opened by rosbags in read-only mode. With --samples, a few left/right IR frames are written to
OUT_DIR (as PNG) for the projector / rectification check (E3.3).

Needs: rosbags, numpy, opencv-python-headless  (venv: ~/venvs/ext).
"""
from __future__ import annotations

import argparse
import collections
import math
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

IR_L, IR_R = "/camera/infra1/image_rect_raw", "/camera/infra2/image_rect_raw"
INFO_L, INFO_R = "/camera/infra1/camera_info", "/camera/infra2/camera_info"


def stamp_ns(h) -> int:
    return int(h.stamp.sec) * 1_000_000_000 + int(h.stamp.nanosec)


def image_array(msg) -> np.ndarray:
    a = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
    if msg.encoding in ("mono8", "8UC1"):
        return a[:, :msg.width]
    if msg.encoding in ("mono16", "16UC1"):
        return a.view(np.uint16)[:, :msg.width]
    raise ValueError("unsupported encoding " + msg.encoding)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bag", type=Path)
    ap.add_argument("--samples", type=Path)
    a = ap.parse_args()
    ts = get_typestore(Stores.ROS2_HUMBLE)
    with AnyReader([a.bag], default_typestore=ts) as r:
        print("== bag: %s  duration %.1f s  messages %d" % (a.bag, r.duration / 1e9, r.message_count))
        for c in sorted(r.connections, key=lambda c: c.topic):
            print("   %-48s %-40s %7d" % (c.topic, c.msgtype, c.msgcount))
        by = {c.topic: c for c in r.connections}

        def first(topic):
            for conn, _, raw in r.messages(connections=[by[topic]]):
                return r.deserialize(raw, conn.msgtype)

        # --- camera_info / calibration
        print("\n== IR calibration")
        cal = {}
        for t in (INFO_L, INFO_R):
            if t not in by:
                print("   MISSING", t)
                continue
            m = first(t)
            k, p = np.array(m.k), np.array(m.p)
            cal[t] = p
            print("   %s frame=%s %dx%d model=%s" % (t, m.header.frame_id, m.width, m.height, m.distortion_model))
            print("     K fx=%.4f fy=%.4f cx=%.4f cy=%.4f   D=%s" % (k[0], k[4], k[2], k[5], np.round(np.array(m.d), 6).tolist()))
            print("     P row0=%s" % np.round(p[:4], 6).tolist())
        if INFO_R in cal:
            p = cal[INFO_R]
            print("   baseline = -P_right[0,3]/fx = %.5f m" % (-p[3] / p[0]))

        # --- images: encodings, stamps, pairing
        print("\n== IR images")
        stamps = {}
        for t in (IR_L, IR_R):
            if t not in by:
                print("   MISSING", t)
                continue
            s, enc = [], None
            for conn, _, raw in r.messages(connections=[by[t]]):
                m = r.deserialize(raw, conn.msgtype)
                s.append(stamp_ns(m.header))
                enc = enc or (m.encoding, m.width, m.height, m.header.frame_id)
            s = np.array(s)
            stamps[t] = s
            d = np.diff(np.sort(s)) / 1e6
            print("   %s enc=%s %dx%d frame=%s  n=%d  span=%.1f s  dt median=%.2f ms  max=%.1f ms  gaps>2x=%d"
                  % (t, enc[0], enc[1], enc[2], enc[3], len(s), (s.max() - s.min()) / 1e9, np.median(d), d.max(),
                     int((d > 2 * np.median(d)).sum())))
        if len(stamps) == 2:
            common = np.intersect1d(stamps[IR_L], stamps[IR_R])
            print("   identical left/right stamps: %d of %d / %d" % (len(common), len(stamps[IR_L]), len(stamps[IR_R])))

        # --- TF
        print("\n== /tf_static")
        if "/tf_static" in by:
            for conn, _, raw in r.messages(connections=[by["/tf_static"]]):
                m = r.deserialize(raw, conn.msgtype)
                for tr in m.transforms:
                    q, v = tr.transform.rotation, tr.transform.translation
                    print("   %-32s -> %-36s t=(%.4f %.4f %.4f) q=(%.4f %.4f %.4f %.4f)"
                          % (tr.header.frame_id, tr.child_frame_id, v.x, v.y, v.z, q.x, q.y, q.z, q.w))
        print("\n== /tf (dynamic) parent -> child : count")
        if "/tf" in by:
            cnt = collections.Counter()
            for conn, _, raw in r.messages(connections=[by["/tf"]]):
                for tr in r.deserialize(raw, conn.msgtype).transforms:
                    cnt[(tr.header.frame_id, tr.child_frame_id)] += 1
            for (p_, c_), n in cnt.most_common():
                print("   %-32s -> %-32s %d" % (p_, c_, n))

        # --- lidar
        for t in ("/lidar_points",):
            if t not in by:
                continue
            print("\n== %s" % t)
            n = 0
            st = []
            for conn, _, raw in r.messages(connections=[by[t]]):
                m = r.deserialize(raw, conn.msgtype)
                st.append(stamp_ns(m.header))
                if n == 0:
                    print("   frame=%s  %dx%d  point_step=%d  fields=%s" % (m.header.frame_id, m.height, m.width,
                          m.point_step, [(f.name, f.offset, f.datatype) for f in m.fields]))
                n += 1
            st = np.array(st)
            print("   n=%d span=%.1f s rate=%.2f Hz" % (n, (st[-1] - st[0]) / 1e9, (n - 1) / ((st[-1] - st[0]) / 1e9)))

        for t in ("/platform/odom", "/platform/odom/filtered"):
            if t in by:
                m = first(t)
                print("\n== %s frame=%s child=%s  pose_cov_diag=%s twist_cov_diag=%s" % (
                    t, m.header.frame_id, m.child_frame_id, np.round(np.array(m.pose.covariance)[::7], 6).tolist(),
                    np.round(np.array(m.twist.covariance)[::7], 6).tolist()))

        # --- IMU: frame, units, gravity, orientation, rest statistics over the first 3 s
        if "/imu/data" in by:
            acc, gyr, ori, st = [], [], [], []
            for conn, _, raw in r.messages(connections=[by["/imu/data"]]):
                m = r.deserialize(raw, conn.msgtype)
                s = stamp_ns(m.header)
                st.append(s)
                if len(st) == 1:
                    print("\n== /imu/data frame=%s  orient_cov=%s gyro_cov=%s acc_cov=%s" % (
                        m.header.frame_id, np.round(np.array(m.orientation_covariance)[::4], 8).tolist(),
                        np.round(np.array(m.angular_velocity_covariance)[::4], 8).tolist(),
                        np.round(np.array(m.linear_acceleration_covariance)[::4], 8).tolist()))
                acc.append([m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z])
                gyr.append([m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z])
                ori.append([m.orientation.x, m.orientation.y, m.orientation.z, m.orientation.w])
            acc, gyr, st = np.array(acc), np.array(gyr), np.array(st)
            d = np.diff(st) / 1e9
            print("   n=%d rate=%.1f Hz dt max=%.1f ms  orientation present=%s"
                  % (len(st), 1 / np.median(d), d.max() * 1e3, bool(np.abs(np.array(ori)).sum() > 0)))
            w = (st - st[0]) < 3e9
            print("   first 3 s: |acc| mean=%.4f std=%.4f  acc mean=%s  gyro mean=%s  gyro std=%s"
                  % (np.linalg.norm(acc[w], axis=1).mean(), np.linalg.norm(acc[w], axis=1).std(),
                     np.round(acc[w].mean(0), 4).tolist(), np.round(gyr[w].mean(0), 5).tolist(),
                     np.round(gyr[w].std(0), 5).tolist()))
            print("   whole bag: |acc| mean=%.4f  |gyro| max=%.3f rad/s  |acc| max=%.2f"
                  % (np.linalg.norm(acc, axis=1).mean(), np.linalg.norm(gyr, axis=1).max(),
                     np.linalg.norm(acc, axis=1).max()))
        if "/imu/mag" in by:
            mg = []
            for conn, _, raw in r.messages(connections=[by["/imu/mag"]]):
                m = r.deserialize(raw, conn.msgtype)
                mg.append([m.magnetic_field.x, m.magnetic_field.y, m.magnetic_field.z])
            mg = np.linalg.norm(np.array(mg), axis=1)
            print("\n== /imu/mag |B| mean=%.3e std=%.3e (std/mean %.1f%%: large = indoor disturbances)"
                  % (mg.mean(), mg.std(), 100 * mg.std() / mg.mean()))

        # --- sample frames
        if a.samples and IR_L in by and IR_R in by:
            a.samples.mkdir(parents=True, exist_ok=True)
            import cv2
            n = by[IR_L].msgcount
            want = {int(n * f) for f in (0.05, 0.3, 0.55, 0.8)}
            left = {}
            for i, (conn, _, raw) in enumerate(r.messages(connections=[by[IR_L]])):
                if i in want:
                    m = r.deserialize(raw, conn.msgtype)
                    left[stamp_ns(m.header)] = image_array(m)
            for conn, _, raw in r.messages(connections=[by[IR_R]]):
                m = r.deserialize(raw, conn.msgtype)
                s = stamp_ns(m.header)
                if s in left:
                    im = np.hstack([left[s], image_array(m)])
                    for y in range(0, im.shape[0], 40):     # rectification check: features on the same row
                        im[y, :] = 255
                    cv2.imwrite(str(a.samples / ("pair_%d.png" % s)), im)
            print("\n== wrote %d sample pairs to %s" % (len(list(a.samples.glob('pair_*.png'))), a.samples))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
