#!/usr/bin/env python3
"""Convert an external ROS 2 bag's IR stereo to the EuRoC layout stereo_euroc replays
(plan/evaluation_plan_stage_1.md, E2.1 + E2.2).

usage: bag_to_euroc.py BAG_DIR OUT_DIR [--left T] [--right T] [--left-info T] [--right-info T]

Writes into OUT_DIR (the bag is only read):
  mav0/cam0/data/<t_ns>.png, mav0/cam1/data/<t_ns>.png   left / right IR, lossless
  times.txt                                                one <t_ns> per stereo pair
  settings.yaml                                            Examples/Stereo/RealSense_D435.yaml with the
                                                           camera block replaced from camera_info
  calib.json, tf_static.json, source_info.txt              what was used, for the record
Pairs are matched on identical header stamps (D435 IR frames of one exposure share the stamp).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

REPO = Path(__file__).resolve().parents[2]
TEMPLATE = REPO / "Examples/Stereo/RealSense_D435.yaml"


def stamp_ns(h) -> int:
    return int(h.stamp.sec) * 1_000_000_000 + int(h.stamp.nanosec)


def mono(msg) -> np.ndarray:
    if msg.encoding not in ("mono8", "8UC1"):
        raise SystemExit("unsupported IR encoding %s (expected mono8)" % msg.encoding)
    return np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, :msg.width]


def settings_yaml(kl, pr, w, h, fps) -> str:
    fx, fy, cx, cy = kl
    b = -pr[3] / pr[0]
    if not 0.045 <= b <= 0.055:
        raise SystemExit("baseline %.4f m is not a D435 baseline (0.045-0.055): check the camera_info topics" % b)
    s = TEMPLATE.read_text()
    rep = {"Camera1.fx": "%.6f" % fx, "Camera1.fy": "%.6f" % fy, "Camera1.cx": "%.6f" % cx,
           "Camera1.cy": "%.6f" % cy, "Stereo.b": "%.6f" % b, "Camera.width": str(w),
           "Camera.height": str(h), "Camera.fps": str(int(round(fps)))}
    for k, v in rep.items():
        s, n = re.subn(r"^(%s:\s*).*$" % re.escape(k), r"\g<1>" + v, s, flags=re.M)
        if n != 1:
            raise SystemExit("template has no unique key " + k)
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bag", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--left", default="/camera/infra1/image_rect_raw")
    ap.add_argument("--right", default="/camera/infra2/image_rect_raw")
    ap.add_argument("--left-info", default="/camera/infra1/camera_info")
    ap.add_argument("--right-info", default="/camera/infra2/camera_info")
    a = ap.parse_args()
    c0, c1 = a.out / "mav0/cam0/data", a.out / "mav0/cam1/data"
    c0.mkdir(parents=True, exist_ok=True)
    c1.mkdir(parents=True, exist_ok=True)

    ts = get_typestore(Stores.ROS2_HUMBLE)
    with AnyReader([a.bag], default_typestore=ts) as r:
        by = {c.topic: c for c in r.connections}

        def first(t):
            for conn, _, raw in r.messages(connections=[by[t]]):
                return r.deserialize(raw, conn.msgtype)

        il, ir = first(a.left_info), first(a.right_info)
        if any(abs(x) > 1e-9 for x in list(il.d) + list(ir.d)):
            raise SystemExit("non-zero distortion in camera_info: images are not rectified")
        kl = (il.k[0], il.k[4], il.k[2], il.k[5])
        calib = dict(left=dict(frame=il.header.frame_id, w=il.width, h=il.height, K=list(il.k), P=list(il.p)),
                     right=dict(frame=ir.header.frame_id, w=ir.width, h=ir.height, K=list(ir.k), P=list(ir.p)),
                     baseline_m=-ir.p[3] / ir.p[0])

        tf = []
        if "/tf_static" in by:
            for conn, _, raw in r.messages(connections=[by["/tf_static"]]):
                for t in r.deserialize(raw, conn.msgtype).transforms:
                    v, q = t.transform.translation, t.transform.rotation
                    tf.append(dict(parent=t.header.frame_id, child=t.child_frame_id,
                                   t=[v.x, v.y, v.z], q_xyzw=[q.x, q.y, q.z, q.w]))

        # Images: the two topics are interleaved in the bag; hold unmatched frames until the partner arrives.
        pending = {a.left: {}, a.right: {}}
        written, png = [], [cv2.IMWRITE_PNG_COMPRESSION, 1]
        for conn, _, raw in r.messages(connections=[by[a.left], by[a.right]]):
            m = r.deserialize(raw, conn.msgtype)
            s = stamp_ns(m.header)
            me, other = (a.left, a.right) if conn.topic == a.left else (a.right, a.left)
            if s in pending[other]:
                im_o = pending[other].pop(s)
                im_l, im_r = (mono(m), im_o) if me == a.left else (im_o, mono(m))
                cv2.imwrite(str(c0 / ("%d.png" % s)), im_l, png)
                cv2.imwrite(str(c1 / ("%d.png" % s)), im_r, png)
                written.append(s)
                if len(written) % 500 == 0:
                    print("  %d pairs" % len(written), flush=True)
            else:
                pending[me][s] = mono(m)
                for k in [k for k in pending[me] if k < s - 2_000_000_000]:   # partner never came
                    del pending[me][k]

    written.sort()
    unl, unr = len(pending[a.left]), len(pending[a.right])
    (a.out / "times.txt").write_text("".join("%d\n" % s for s in written))
    for cam in ("cam0", "cam1"):
        (a.out / "mav0" / cam / "data.csv").write_text(
            "#timestamp [ns],filename\n" + "".join("%d,%d.png\n" % (s, s) for s in written))
    d = np.diff(np.array(written)) / 1e9
    fps = 1.0 / np.median(d)
    (a.out / "settings.yaml").write_text(settings_yaml(kl, ir.p, il.width, il.height, fps))
    (a.out / "calib.json").write_text(json.dumps(calib, indent=1))
    (a.out / "tf_static.json").write_text(json.dumps(tf, indent=1))
    info = ("source_bag: %s\nleft: %s\nright: %s\npairs: %d\nunmatched_left_at_end: %d\nunmatched_right_at_end: %d\n"
            "duration_s: %.3f\nfps_median: %.3f\nmax_gap_s: %.3f\ngaps_over_2x: %d\nbaseline_m: %.5f\nresolution: %dx%d\n"
            % (a.bag, a.left, a.right, len(written), unl, unr, (written[-1] - written[0]) / 1e9, fps, d.max(),
               int((d > 2 / fps).sum()), calib["baseline_m"], il.width, il.height))
    (a.out / "source_info.txt").write_text(info)
    print(info)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
