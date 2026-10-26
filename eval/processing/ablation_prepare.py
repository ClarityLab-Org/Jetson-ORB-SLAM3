#!/usr/bin/env python3
"""Build every ablation variant of the Husky sequence (plan/ablation_plan_husky.md, step P-A).
Runs on the PC, which has the bag; the Jetson only receives the result.

usage: ablation_prepare.py [--bag DIR] [--base SEQ_DIR] [--reference DIR] [--stage1-run DIR]
                           [--imu-offset-ms D] [--no-rgbd]

  --bag         ~/datasets/external_raw/husky_20261023          (raw bag: IMU, depth, colour)
  --base        ~/datasets/ext/husky_20261023                    (stage-1 IR stereo sequence)
  --reference   eval/results/husky/reference          (fused reference, repository copy)
  --stage1-run  a stage-1 ORB-SLAM3 run folder with f_run.txt, used to measure the camera-IMU time
                offset (default: the first gpu run under eval/results/husky/repro_results_ext)

Adds to the base sequence (shared by all variants, nothing existing is changed):
  mav0/imu0/data.csv   Husky IMU in EuRoC format ('#timestamp [ns],w_x,w_y,w_z,a_x,a_y,a_z'), IMU frame,
                       timestamps shifted by the measured camera-IMU offset
  depth/<t>.png        16-bit depth (mm) in the IR-left frame, same stamps as the IR images
  color/<t>.png, depth_c/<t>.png   colour (RGB) + depth aligned to colour
Creates ~/datasets/ext/husky_20261023__<variant>/ for each variant below: symlinks to the base data,
its own settings.yaml, times.txt, gt.csv (in that mode's output frame), mode.txt, and for RGB-D
assoc.txt. Writes ablation_prepare.json (calibration values used) into the base folder.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "eval" / "benchmark"))
from make_gt import chain, q2R  # noqa: E402

# variant -> (mode, gt frame, description). The gt frame is the frame the program's trajectory is in.
VARIANTS = {
    "mono":       ("ext_m", "cam", "A1 mono IR (left camera only)"),
    "mi":         ("ext_mi", "imu", "A3 mono IR + IMU"),
    "si":         ("ext_si", "imu", "A4 stereo IR + IMU"),
    "ird":        ("ext_rgbd", "cam", "A5 RGB-D mode: IR left + depth"),
    "rgbd":       ("ext_rgbd", "color", "A6 RGB-D mode: colour + aligned depth"),
    "nolc":       ("ext_s", "cam", "C1 stereo, loop closing off"),
    "si_nolc":    ("ext_si", "imu", "C1 stereo + IMU, loop closing off"),
    "f800":       ("ext_s", "cam", "D2 stereo, 800 features"),
    "si_f800":    ("ext_si", "imu", "D2 stereo + IMU, 800 features"),
    "7p5":        ("ext_s", "cam", "D1 stereo, 7.5 FPS (every 2nd frame)"),
    "si_7p5":     ("ext_si", "imu", "D1 stereo + IMU, 7.5 FPS"),
    "bscale":     ("ext_s", "cam", "D3 stereo, baseline x1.068 (tests the stage-1 scale finding)"),
}


def stamp_ns(h):
    return int(h.stamp.sec) * 1_000_000_000 + int(h.stamp.nanosec)


def set_key(text, key, value):
    """Set `key: value` in an ORB-SLAM3 YAML (replace if present, else append)."""
    pat = r"^(%s:\s*).*$" % re.escape(key)
    if re.search(pat, text, flags=re.M):
        return re.sub(pat, r"\g<1>" + value, text, count=1, flags=re.M)
    return text.rstrip("\n") + "\n%s: %s\n" % (key, value)


def get_key(text, key):
    m = re.search(r"^%s:\s*([^\s#]+)" % re.escape(key), text, flags=re.M)
    return m.group(1) if m else None


def imu_block(T_bc, noise):
    d = ", ".join("%.9f" % v for v in T_bc.ravel())
    return ("\n# --- IMU (Husky IMU, imu_0_link): written by eval/processing/ablation_prepare.py ---\n"
            "IMU.T_b_c1: !!opencv-matrix\n   rows: 4\n   cols: 4\n   dt: f\n   data: [%s]\n"
            "IMU.InsertKFsWhenLost: 0\nIMU.NoiseGyro: %.3e\nIMU.NoiseAcc: %.3e\nIMU.GyroWalk: %.3e\n"
            "IMU.AccWalk: %.3e\nIMU.Frequency: %.1f\n" % (d, noise["gyro"], noise["acc"], noise["gyro_walk"],
                                                        noise["acc_walk"], noise["rate_hz"]))


def pinhole(text, fx, fy, cx, cy):
    text = set_key(text, "Camera.type", '"PinHole"')
    for k, v in (("Camera1.fx", fx), ("Camera1.fy", fy), ("Camera1.cx", cx), ("Camera1.cy", cy),
                 ("Camera1.k1", 0.0), ("Camera1.k2", 0.0), ("Camera1.p1", 0.0), ("Camera1.p2", 0.0)):
        text = set_key(text, k, "%.6f" % v)
    return text


def angular_rate_from_traj(t, T):
    """|omega| [rad/s] from consecutive rotations of a trajectory (frame-independent)."""
    w = []
    for i in range(1, len(t)):
        dR = T[i - 1][:3, :3].T @ T[i][:3, :3]
        ang = math.acos(max(-1.0, min(1.0, (np.trace(dR) - 1) / 2)))
        w.append(ang / max(t[i] - t[i - 1], 1e-3))
    return 0.5 * (t[1:] + t[:-1]), np.array(w)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", type=Path, default=Path.home() / "datasets/external_raw/husky_20261023")
    ap.add_argument("--base", type=Path, default=Path.home() / "datasets/ext/husky_20261023")
    ap.add_argument("--reference", type=Path, default=REPO / "eval/results/husky/reference")
    ap.add_argument("--stage1-run", type=Path,
                    default=REPO / "eval/results/husky/stage1/runs/ext/ext_s/husky_20261023/gpu/run1")
    ap.add_argument("--imu-offset-ms", type=float, help="override the measured camera-IMU offset")
    ap.add_argument("--imu-noise-inflate", type=float, default=5.0)
    ap.add_argument("--no-rgbd", action="store_true", help="skip depth/colour extraction and A5/A6")
    a = ap.parse_args()
    import cv2
    from evaluate import load_tum
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore

    base = a.base
    tf = json.loads((base / "tf_static.json").read_text())
    calib = json.loads((base / "calib.json").read_text())
    cam_frame = calib["left"]["frame"]
    times = np.loadtxt(base / "times.txt", dtype=np.int64)
    base_settings = (base / "settings.yaml").read_text()
    info = {}

    with AnyReader([a.bag], default_typestore=get_typestore(Stores.ROS2_HUMBLE)) as r:
        by = {c.topic: c for c in r.connections}
        # ---------------------------------------------------------------- IMU
        print("==> IMU")
        it, g, ac, imu_frame = [], [], [], None
        for c, _, raw in r.messages(connections=[by["/imu/data"]]):
            m = r.deserialize(raw, c.msgtype)
            imu_frame = imu_frame or m.header.frame_id
            it.append(stamp_ns(m.header))
            g.append([m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z])
            ac.append([m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z])
        o = np.argsort(it, kind="stable")
        it, g, ac = np.array(it)[o], np.array(g)[o], np.array(ac)[o]
        keep = np.concatenate([[True], np.diff(it) > 0])       # drop duplicate stamps
        it, g, ac = it[keep], g[keep], ac[keep]
        rate = 1e9 / float(np.median(np.diff(it)))
        rest = (it - it[0]) < 2.4e9                             # robot at rest for the first ~2.5 s
        dt = 1.0 / rate
        s = a.imu_noise_inflate
        noise = dict(gyro=s * max(float(g[rest].std(0).max()), 1e-4) * math.sqrt(dt),
                     acc=s * max(float(ac[rest].std(0).max()), 1e-3) * math.sqrt(dt),
                     gyro_walk=1e-5, acc_walk=1e-3, rate_hz=rate)
        T_ic = chain(tf, imu_frame, cam_frame)                  # IMU.T_b_c1: camera pose in the IMU frame
        info.update(imu_frame=imu_frame, imu_rate_hz=rate, imu_noise=noise, T_imu_cam=T_ic.tolist(),
                    acc_rest_norm=float(np.linalg.norm(ac[rest].mean(0))))
        print("    %d samples at %.1f Hz; noise gyro %.2e acc %.2e (x%.0f); |acc| at rest %.3f" % (
            len(it), rate, noise["gyro"], noise["acc"], s, info["acc_rest_norm"]))

        # camera-IMU time offset: gyro |omega| vs the stage-1 camera trajectory's |omega|
        off_ms = a.imu_offset_ms
        if off_ms is None and (a.stage1_run / "f_run.txt").is_file():
            te, Te = load_tum(a.stage1_run / "f_run.txt")
            tc, wc = angular_rate_from_traj(te, Te)
            ti = (it - it[0]) / 1e9 + it[0] / 1e9
            wi = np.linalg.norm(g - g[rest].mean(0), axis=1)
            grid = np.arange(tc[0] + 1, tc[-1] - 1, 0.01)
            cam = np.interp(grid, tc, wc)
            best = None
            for lag in np.arange(-0.15, 0.1501, 0.002):         # IMU stamp + lag = camera time
                imu = np.interp(grid, ti + lag, wi)
                cc = float(np.corrcoef(cam, imu)[0, 1])
                if best is None or cc > best[1]:
                    best = (lag, cc)
            off_ms = round(best[0] * 1000, 1)
            info.update(imu_offset_corr=best[1])
            print("    camera-IMU offset from %s: %+.1f ms (correlation %.2f)" % (a.stage1_run.name, off_ms, best[1]))
        off_ms = off_ms or 0.0
        if abs(off_ms) < 5:
            print("    |offset| < 5 ms: not applied")
            applied = 0.0
        else:
            applied = off_ms
            print("    applying %+.1f ms to the IMU stamps" % applied)
        info["imu_offset_ms_applied"] = applied
        d = base / "mav0/imu0"
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "data.csv", "w") as f:
            f.write("#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],"
                    "a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n")
            for s_, gg, aa in zip(it + int(round(applied * 1e6)), g, ac):
                f.write("%d,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f\n" % (s_, *gg, *aa))

        # ---------------------------------------------------------------- depth + colour
        assoc_ird, assoc_rgbd, color_times = [], [], []
        if not a.no_rgbd:
            print("==> depth + colour images")
            for sub in ("depth", "color", "depth_c"):
                (base / sub).mkdir(exist_ok=True)
            png = [cv2.IMWRITE_PNG_COMPRESSION, 1]
            for topic, sub in (("/camera/depth/image_rect_raw", "depth"), ("/camera/color/image_raw", "color"),
                               ("/camera/aligned_depth_to_color/image_raw", "depth_c")):
                n = 0
                for c, _, raw in r.messages(connections=[by[topic]]):
                    m = r.deserialize(raw, c.msgtype)
                    st = stamp_ns(m.header)
                    out = base / sub / ("%d.png" % st)
                    if not out.is_file():
                        if m.encoding in ("16UC1", "mono16"):
                            im = np.frombuffer(m.data, np.uint16).reshape(m.height, m.step // 2)[:, :m.width]
                        elif m.encoding == "rgb8":
                            im = np.frombuffer(m.data, np.uint8).reshape(m.height, m.step)[:, :m.width * 3]
                            im = im.reshape(m.height, m.width, 3)          # stored as is; Camera.RGB: 1
                        else:
                            raise SystemExit("unexpected encoding %s on %s" % (m.encoding, topic))
                        cv2.imwrite(str(out), im, png)
                    n += 1
                    if sub == "color":
                        color_times.append(st)
                print("    %s: %d images" % (sub, n))
            dts = np.array(sorted(int(p.stem) for p in (base / "depth").glob("*.png")))
            dcs = np.array(sorted(int(p.stem) for p in (base / "depth_c").glob("*.png")))
            for s_ in times:                                    # IR left + depth (same exposure)
                j = int(np.argmin(np.abs(dts - s_)))
                if abs(int(dts[j]) - int(s_)) < 5_000_000:
                    assoc_ird.append("%.9f mav0/cam0/data/%d.png %.9f depth/%d.png" % (s_ / 1e9, s_, dts[j] / 1e9, dts[j]))
            color_times = np.array(sorted(color_times))
            for s_ in color_times:                              # colour + aligned depth
                j = int(np.argmin(np.abs(dcs - s_)))
                if abs(int(dcs[j]) - int(s_)) < 20_000_000:
                    assoc_rgbd.append("%.9f color/%d.png %.9f depth_c/%d.png" % (s_ / 1e9, s_, dcs[j] / 1e9, dcs[j]))
            print("    associations: IR+depth %d, colour+depth %d" % (len(assoc_ird), len(assoc_rgbd)))

        cinfo = {}
        if "/camera/color/camera_info" in by:
            for c, _, raw in r.messages(connections=[by["/camera/color/camera_info"]]):
                m = r.deserialize(raw, c.msgtype)
                cinfo = dict(frame=m.header.frame_id, K=list(m.k), D=list(m.d))
                break

    # -------------------------------------------------------------------- variants
    print("==> variants")
    kl = calib["left"]["K"]
    fused = a.reference / "fusion" / "fused_poses.txt"
    mono_s = pinhole(base_settings, kl[0], kl[4], kl[2], kl[5])
    imu_txt = imu_block(T_ic, noise)
    made = []
    for v, (mode, gtf, desc) in VARIANTS.items():
        if a.no_rgbd and v in ("ird", "rgbd"):
            continue
        d = base.parent / ("%s__%s" % (base.name, v))
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        (d / "mav0").symlink_to(Path("..") / base.name / "mav0")
        for f in ("tf_static.json", "calib.json"):
            shutil.copy(base / f, d / f)
        st = base_settings
        t_out = times
        if v == "mono":
            st = mono_s
        elif v == "mi":
            st = mono_s + imu_txt
        elif v.startswith("si"):
            st = base_settings + imu_txt
        if v in ("nolc", "si_nolc"):
            st = set_key(st, "loopClosing", "0")
        if v in ("f800", "si_f800"):
            st = set_key(st, "ORBextractor.nFeatures", "800")
        if v in ("7p5", "si_7p5"):
            st = set_key(st, "Camera.fps", "8")
            t_out = times[::2]
        if v == "bscale":
            st = set_key(st, "Stereo.b", "%.6f" % (float(get_key(base_settings, "Stereo.b")) * 1.068))
        if v == "ird":                                          # RGB-D mode needs a PinHole camera
            st = set_key(mono_s, "RGBD.DepthMapFactor", "1000.0")
            (d / "depth").symlink_to(Path("..") / base.name / "depth")
            (d / "assoc.txt").write_text("\n".join(assoc_ird) + "\n")
        if v == "rgbd":
            K = cinfo["K"]
            st = pinhole(base_settings, K[0], K[4], K[2], K[5])
            st = set_key(st, "Camera.RGB", "1")
            st = set_key(st, "RGBD.DepthMapFactor", "1000.0")
            for sub in ("color", "depth_c"):
                (d / sub).symlink_to(Path("..") / base.name / sub)
            (d / "assoc.txt").write_text("\n".join(assoc_rgbd) + "\n")
            t_out = color_times
        (d / "settings.yaml").write_text(st)
        (d / "times.txt").write_text("".join("%d\n" % x for x in t_out))
        (d / "mode.txt").write_text(mode + "\n")
        (d / "variant.txt").write_text(desc + "\n")
        frame = {"cam": cam_frame, "imu": info["imu_frame"], "color": cinfo.get("frame")}[gtf]
        subprocess.run([sys.executable, "-I", str(HERE / "make_gt.py"), str(d), str(fused), "--cam-frame", frame,
                        "--out", str(d / "gt.csv")], check=True, stdout=subprocess.DEVNULL)
        n_gt = sum(1 for _ in open(d / "gt.csv")) - 1
        made.append((d.name, mode, frame, len(t_out), n_gt, desc))
        print("    %-26s %-9s gt frame %-28s images %5d  gt %5d  %s" % (d.name, mode, frame, len(t_out), n_gt, desc))
    info["variants"] = [dict(name=m[0], mode=m[1], gt_frame=m[2], images=m[3], gt_poses=m[4], desc=m[5]) for m in made]
    (base / "ablation_prepare.json").write_text(json.dumps(info, indent=1))
    print("==> done: %d variants; calibration in %s" % (len(made), base / "ablation_prepare.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
