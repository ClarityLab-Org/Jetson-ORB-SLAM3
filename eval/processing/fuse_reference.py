#!/usr/bin/env python3
"""Best-estimate reference trajectory from EVERY non-camera sensor in the Husky bag
(plan/husky_stage1_runbook.md, step 3).

usage: fuse_reference.py BAG_DIR OUT_DIR [options]      (see --help)

How each non-camera stream is used
  /lidar_points            KISS-ICP lidar odometry (deskewed with per-point times) -> between factors,
                           and lidar loop closures (keyframe-to-submap ICP) -> loop factors
  /imu/data                gyro + accelerometer -> IMU preintegration factors (GTSAM CombinedImuFactor,
                           bias estimated); gyro also predicts each scan's rotation for ICP; the
                           driver's orientation -> weak roll/pitch factors, only if its gravity
                           direction agrees with the accelerometer at rest (checked automatically)
  /platform/odom           wheel odometry -> between factors (planar, slip-tolerant noise, scale
                           calibrated against the lidar); also predicts each scan's translation for ICP
  /tf_static               lidar / IMU / base_link / camera extrinsics
  /imu/mag                 heading, only if it passes a consistency check (indoor fields usually fail)
  /platform/odom/filtered, /tf odom->base_link, /imu/rpy, /platform/joint_states
                           derived from the streams above (EKF of wheels + IMU, its TF, the IMU
                           driver's Euler angles, the wheel encoders behind /platform/odom). Fusing
                           them again would double-count the same measurements, so they are used for
                           VALIDATION only (reported against the fused result)
  camera topics            never read: the reference must be independent of the ORB-SLAM3 runs

Graph (GTSAM, batch Levenberg-Marquardt): one state per lidar scan (10 Hz): pose of base_link,
velocity, IMU bias. Lidar, wheel and loop factors carry robust kernels (Huber / Cauchy), so a bad
ICP result, wheel slip or a wrong loop is down-weighted instead of bending the trajectory.

Outputs in OUT_DIR
  fused_poses.txt          't_ns x y z qx qy qz qw' of base_link (the reference)
  frontend_poses.txt       the lidar odometry alone (with IMU/wheel prediction), base_link
  lidar_odometry_info.txt  'frame: base_link' etc. (read by make_gt.py)
  loops.csv, factors.txt, biases.txt, fusion_report.txt, fusion_*.png
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_gt import R2q, T, chain, q2R, slerp  # noqa: E402

CAMERA_PREFIX = "/camera/"


# ------------------------------------------------------------------------------ small SO(3)/SE(3)
def expSO3(w):
    th = float(np.linalg.norm(w))
    K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    if th < 1e-9:
        return np.eye(3) + K
    return np.eye(3) + math.sin(th) / th * K + (1 - math.cos(th)) / th ** 2 * K @ K


def rot_angle(R):
    return math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2)))


def inv(M):
    Mi = np.eye(4)
    Mi[:3, :3] = M[:3, :3].T
    Mi[:3, 3] = -M[:3, :3].T @ M[:3, 3]
    return Mi


def rot_between(a, b):
    """Rotation R with R @ a_hat = b_hat."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3) if c > 0 else expSO3(np.array([np.pi, 0, 0]))
    return expSO3(v / np.linalg.norm(v) * math.atan2(np.linalg.norm(v), c))


def interp_pose(ts, P, Q, t):
    """Pose at time t from arrays (ts int ns, positions, quaternions xyzw); None outside the range."""
    i = int(np.searchsorted(ts, t))
    if i == 0 or i >= len(ts):
        return None
    u = (t - ts[i - 1]) / (ts[i] - ts[i - 1])
    return T((1 - u) * P[i - 1] + u * P[i], slerp(Q[i - 1], Q[i], u))


# ------------------------------------------------------------------------------ bag reading
def stamp_ns(h):
    return int(h.stamp.sec) * 1_000_000_000 + int(h.stamp.nanosec)


def read_aux(reader, by, a):
    """IMU, wheel odometry, mag, validation odometry, static TF: everything except lidar and camera."""
    out = dict(imu_t=[], gyr=[], acc=[], ori=[], odom_t=[], odom_p=[], odom_q=[], filt_t=[], filt_p=[],
               filt_q=[], mag_t=[], mag=[], tf=[], frames={})
    want = [t for t in ("/imu/data", a.wheel_topic, "/platform/odom/filtered", "/imu/mag", "/tf_static") if t in by]
    for conn, _, raw in reader.messages(connections=[by[t] for t in want]):
        m = reader.deserialize(raw, conn.msgtype)
        t = conn.topic
        if t == "/imu/data":
            out["frames"].setdefault("imu", m.header.frame_id)
            out["imu_t"].append(stamp_ns(m.header))
            out["gyr"].append([m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z])
            out["acc"].append([m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z])
            out["ori"].append([m.orientation.x, m.orientation.y, m.orientation.z, m.orientation.w])
        elif t in (a.wheel_topic, "/platform/odom/filtered"):
            k = "odom" if t == a.wheel_topic else "filt"
            out["frames"].setdefault(k, (m.header.frame_id, m.child_frame_id))
            p, q = m.pose.pose.position, m.pose.pose.orientation
            out[k + "_t"].append(stamp_ns(m.header))
            out[k + "_p"].append([p.x, p.y, p.z])
            out[k + "_q"].append([q.x, q.y, q.z, q.w])
        elif t == "/imu/mag":
            out["mag_t"].append(stamp_ns(m.header))
            out["mag"].append([m.magnetic_field.x, m.magnetic_field.y, m.magnetic_field.z])
        elif t == "/tf_static":
            for tr in m.transforms:
                v, q = tr.transform.translation, tr.transform.rotation
                out["tf"].append(dict(parent=tr.header.frame_id, child=tr.child_frame_id,
                                      t=[v.x, v.y, v.z], q_xyzw=[q.x, q.y, q.z, q.w]))
    for k in list(out):
        if k not in ("tf", "frames"):
            out[k] = np.array(out[k], dtype=np.int64 if k.endswith("_t") else float)
    for k in ("imu", "odom", "filt", "mag"):                # sort by stamp (bags store receive order)
        if len(out[k + "_t"]):
            o = np.argsort(out[k + "_t"], kind="stable")
            for f in [f for f in out if f.startswith(k + "_") or (k == "imu" and f in ("gyr", "acc", "ori"))
                      or (k == "mag" and f == "mag")]:
                out[f] = out[f][o]
    return out


def scan_points(msg):
    """xyz + per-point time of one PointCloud2 (NaN/zero points dropped with their times), and the
    scan's reference time: the middle of its per-point time span (KISS-ICP deskews to mid-scan)."""
    from kiss_icp.tools.point_cloud2 import read_points
    names = [f.name for f in msg.fields]
    tf = next((n for n in ("t", "timestamp", "time") if n in names), None)
    s = read_points(msg, field_names=["x", "y", "z"] + ([tf] if tf else []))
    p = np.column_stack([s["x"], s["y"], s["z"]]).astype(np.float64)
    ok = np.all(np.isfinite(p), axis=1) & (np.abs(p).sum(1) > 1e-6)
    hdr = stamp_ns(msg.header)
    if not tf:
        return p[ok], np.array([]), hdr, 0.0
    pt = s[tf].astype(np.float64)[ok]
    lo, hi = float(pt.min()), float(pt.max())
    if lo > 1e9:                                        # absolute epoch seconds (Hesai driver)
        mid = int(round(0.5 * (lo + hi) * 1e9))
    elif hi > 1e6:                                      # relative nanoseconds
        mid = hdr + int(round(0.5 * (lo + hi)))
    else:                                               # relative seconds
        mid = hdr + int(round(0.5 * (lo + hi) * 1e9))
    return p[ok], pt, mid, (mid - hdr) / 1e6


# ------------------------------------------------------------------------------ IMU helpers
class Imu:
    def __init__(self, t_ns, gyr, acc, t0):
        self.t = (t_ns - t0) / 1e9
        self.g, self.a = gyr, acc

    def segments(self, ta, tb):
        """(acc, gyro, dt) pieces covering [ta, tb] (sample i held until sample i+1)."""
        i0 = max(int(np.searchsorted(self.t, ta, side="right")) - 1, 0)
        out = []
        i = i0
        while i < len(self.t) and self.t[i] < tb:
            s = max(self.t[i], ta)
            e = min(self.t[i + 1] if i + 1 < len(self.t) else tb, tb)
            if e > s:
                out.append((self.a[i], self.g[i], e - s))
            i += 1
        return out

    def delta_R(self, ta, tb, bg):
        R = np.eye(3)
        for _, g, dt in self.segments(ta, tb):
            R = R @ expSO3((g - bg) * dt)
        return R


def rest_window(imu: Imu, wheel_t, wheel_p, t0, max_s=10.0):
    """Initial stationary period: wheels not moving and gyro quiet. Returns its end time (s)."""
    wt = (wheel_t - t0) / 1e9
    moving = wt[1:][np.linalg.norm(np.diff(wheel_p, axis=0), axis=1) > 1e-3]
    end = min(float(moving[0]) if len(moving) else max_s, max_s)
    gq = imu.t < end
    if gq.sum() < 20:
        return 0.0
    return end


# ------------------------------------------------------------------------------ extrinsic checks
def Rz(a):
    c, s_ = math.cos(a), math.sin(a)
    return np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1.0]])


def check_lidar_yaw(reader, lidar_conn, cfg, T_BL, wt, wp, wq, travel_m=3.0, max_scans=600):
    """Yaw of the lidar on the robot, from the data: KISS-ICP alone (no prediction) on the first scans
    until the wheels report `travel_m` of forward travel, then compare the lidar's direction of travel
    (rotated to base_link by the TF) with the wheels'. Returns (yaw correction rad, details)."""
    from kiss_icp.kiss_icp import KissICP
    odo = KissICP(cfg)
    W0, P0, dB, i = None, None, np.zeros(3), 0
    for i, (c, _, raw) in enumerate(reader.messages(connections=lidar_conn)):
        m = reader.deserialize(raw, c.msgtype)
        pts, ptimes, t_mid, _ = scan_points(m)
        odo.register_frame(pts, ptimes)
        Wk = interp_pose(wt, wp, wq, t_mid)
        if Wk is None:
            continue
        if W0 is None:                                      # first scan with wheel data: the start
            W0, P0 = Wk, odo.last_pose.copy()
        dB = (inv(W0) @ Wk)[:3, 3]
        if np.linalg.norm(dB[:2]) > travel_m or i >= max_scans:
            break
    dL = (inv(P0) @ odo.last_pose)[:3, 3] if P0 is not None else np.zeros(3)
    dL_in_B = T_BL[:3, :3] @ dL                             # lidar travel, in base axes per the TF
    a_l, a_w = math.atan2(dL_in_B[1], dL_in_B[0]), math.atan2(dB[1], dB[0])
    corr = (a_w - a_l + math.pi) % (2 * math.pi) - math.pi
    return corr, dict(lidar_travel_m=float(np.linalg.norm(dL_in_B[:2])), wheel_travel_m=float(np.linalg.norm(dB[:2])),
                      yaw_correction_deg=math.degrees(corr), scans_used=i + 1)


def check_imu_axes(imu_B, wt, wp, wq, t0):
    """Sign checks of the IMU axes (in base axes per the TF) against the wheels: yaw rate and forward
    acceleration, both smoothed over 1 s. Returns dict of correlations."""
    tw = (wt - t0) / 1e9
    grid = np.arange(tw[0] + 1, tw[-1] - 1, 0.1)
    yaw = np.unwrap([math.atan2(q2R(q)[1, 0], q2R(q)[0, 0]) for q in wq])
    yaw_g = np.interp(grid, tw, yaw)
    wz_wheel = np.gradient(yaw_g, grid)
    gz = np.interp(grid, imu_B.t, imu_B.g[:, 2])
    k = np.ones(10) / 10
    corr_w = float(np.corrcoef(np.convolve(wz_wheel, k, "same"), np.convolve(gz, k, "same"))[0, 1])
    # forward speed from wheels (body frame) -> acceleration; IMU forward specific force minus its mean
    v = []
    for t in grid:
        A = interp_pose(wt, wp, wq, int(t0 + (t - 0.05) * 1e9))
        Bm = interp_pose(wt, wp, wq, int(t0 + (t + 0.05) * 1e9))
        v.append((inv(A) @ Bm)[0, 3] / 0.1 if A is not None and Bm is not None else 0.0)
    acc_wheel = np.gradient(np.convolve(v, k, "same"), grid)
    ax = np.interp(grid, imu_B.t, imu_B.a[:, 0])
    ax = np.convolve(ax - ax.mean(), k, "same")
    corr_a = float(np.corrcoef(acc_wheel, ax)[0, 1])
    return dict(yaw_rate_corr=corr_w, fwd_acc_corr=corr_a)


# ------------------------------------------------------------------------------ loop closure
def register(points, submap_pts, init, cfg_reg, d_max, kernel):
    from kiss_icp.mapping import VoxelHashMap
    from kiss_icp.registration import Registration
    vm = VoxelHashMap(voxel_size=0.3, max_distance=1e4, max_points_per_voxel=20)
    vm.add_points(submap_pts)
    reg = Registration(max_num_iterations=cfg_reg["iters"], convergence_criterion=1e-4, max_num_threads=0)
    return reg.align_points_to_map(points=points, voxel_map=vm, initial_guess=init,
                                   max_correspondance_distance=d_max, kernel=kernel)


def fitness(points, M, tree, inlier):
    q = points @ M[:3, :3].T + M[:3, 3]
    d, _ = tree.query(q, k=1, distance_upper_bound=1.0)
    ok = d < inlier
    return float(ok.mean()), float(np.sqrt(np.mean(d[ok] ** 2))) if ok.any() else float("inf")


# ------------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--lidar-topic", default="/lidar_points")
    ap.add_argument("--wheel-topic", default="/platform/odom")
    ap.add_argument("--base-frame", default="base_link")
    ap.add_argument("--max-range", type=float, default=40.0)
    ap.add_argument("--min-range", type=float, default=1.0, help="drops returns from the robot itself")
    ap.add_argument("--voxel", type=float, help="KISS-ICP voxel [m]; default max_range/100")
    ap.add_argument("--kf-dist", type=float, default=0.5, help="keyframe spacing [m] for loop search")
    ap.add_argument("--kf-angle", type=float, default=10.0, help="keyframe spacing [deg]")
    ap.add_argument("--loop-min-gap", type=float, default=30.0, help="min time between loop ends [s]")
    ap.add_argument("--loop-radius", type=float, default=3.0, help="base search radius [m] (+3%% of travel)")
    ap.add_argument("--loop-min-fitness", type=float, default=0.35,
                    help="share of query points within 20 cm of the submap after ICP (16-beam lidar: ~0.3-0.6)")
    ap.add_argument("--loop-max-rmse", type=float, default=0.12, help="inlier RMSE after ICP [m]")
    ap.add_argument("--gravity", type=float, default=9.81)
    ap.add_argument("--imu-noise-inflate", type=float, default=5.0)
    ap.add_argument("--no-imu", action="store_true", help="ablation: drop IMU factors")
    ap.add_argument("--no-wheel", action="store_true", help="ablation: drop wheel factors")
    ap.add_argument("--no-loops", action="store_true", help="ablation: drop loop factors")
    ap.add_argument("--no-attitude", action="store_true", help="ablation: drop IMU-orientation factors")
    ap.add_argument("--no-wheel-scale", action="store_true", help="use wheel odometry without scale calibration")
    ap.add_argument("--trust-tf", action="store_true", help="skip the extrinsic checks against the data")
    ap.add_argument("--no-predict", action="store_true",
                    help="front end: KISS-ICP's own constant-velocity prediction instead of IMU + wheels")
    ap.add_argument("--max-scans", type=int, default=0, help="debug: stop after N scans")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    log_lines = []

    def log(s=""):
        print(s, flush=True)
        log_lines.append(s)

    import gtsam
    from gtsam.symbol_shorthand import B, V, X
    from kiss_icp.config import load_config
    from kiss_icp.kiss_icp import KissICP
    from kiss_icp.voxelization import voxel_down_sample
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore
    from scipy.spatial import cKDTree

    ts = get_typestore(Stores.ROS2_HUMBLE)
    with AnyReader([a.bag], default_typestore=ts) as reader:
        by = {c.topic: c for c in reader.connections}
        assert not any(t.startswith(CAMERA_PREFIX) for t in (a.lidar_topic, a.wheel_topic))
        log("== reading IMU, wheel odometry, magnetometer, validation odometry, static TF")
        aux = read_aux(reader, by, a)
        lidar_conn = [by[a.lidar_topic]]
        lidar_frame = None
        for c, _, raw in reader.messages(connections=lidar_conn):
            lidar_frame = reader.deserialize(raw, c.msgtype).header.frame_id
            break

        imu_frame = aux["frames"]["imu"]
        T_BL = chain(aux["tf"], a.base_frame, lidar_frame)
        T_BI = chain(aux["tf"], a.base_frame, imu_frame)
        T_LB = inv(T_BL)
        wheel_child = aux["frames"]["odom"][1]
        if wheel_child != a.base_frame:
            raise SystemExit("wheel odometry child frame is %s, not %s" % (wheel_child, a.base_frame))
        log("   lidar frame %s  T_base_lidar t=%s" % (lidar_frame, np.round(T_BL[:3, 3], 4).tolist()))
        log("   imu frame %s    T_base_imu t=%s" % (imu_frame, np.round(T_BI[:3, 3], 4).tolist()))
        log("   imu %d samples, wheel %d, mag %d, filtered %d" % (len(aux["imu_t"]), len(aux["odom_t"]),
                                                                   len(aux["mag_t"]), len(aux["filt_t"])))

        t0 = int(min(aux["imu_t"][0], aux["odom_t"][0]))
        R_BI = T_BI[:3, :3]
        gyr_B = aux["gyr"] @ R_BI.T            # IMU vectors expressed in base_link axes (for prediction)
        imu_B = Imu(aux["imu_t"], gyr_B, aux["acc"] @ R_BI.T, t0)
        imu_I = Imu(aux["imu_t"], aux["gyr"], aux["acc"], t0)      # raw, for preintegration

        # ---- initial rest: gyro bias, gravity direction, accelerometer scale
        rest_end = rest_window(imu_I, aux["odom_t"], aux["odom_p"], t0)
        w = imu_I.t < rest_end
        if w.sum() < 20:
            log("   WARNING: no stationary start found; gyro bias prior = 0")
            bg_I, acc_rest_I = np.zeros(3), aux["acc"][:50].mean(0)
            g_std, a_std = np.full(3, 3e-3), np.full(3, 3e-2)
        else:
            bg_I, acc_rest_I = aux["gyr"][w].mean(0), aux["acc"][w].mean(0)
            g_std, a_std = aux["gyr"][w].std(0), aux["acc"][w].std(0)
        dt_imu = float(np.median(np.diff(imu_I.t)))
        g_dens = max(float(g_std.max()), 1e-4) * math.sqrt(dt_imu)
        a_dens = max(float(a_std.max()), 1e-3) * math.sqrt(dt_imu)
        has_gravity = bool(abs(np.linalg.norm(acc_rest_I) - a.gravity) < 1.5)
        log("   rest 0-%.1f s: gyro bias %s rad/s, |acc| %.3f (gravity %s), noise density gyro %.2e acc %.2e"
            % (rest_end, np.round(bg_I, 5).tolist(), np.linalg.norm(acc_rest_I),
               "present" if has_gravity else "MISSING", g_dens, a_dens))
        use_imu = not a.no_imu and has_gravity
        if not has_gravity and not a.no_imu:
            log("   IMU factors disabled: accelerometer has no gravity (driver output is gravity-compensated)")

        # Gravity-aligned world: base_link at t0 with roll/pitch from the resting accelerometer, yaw 0.
        R_W_B0 = rot_between(R_BI @ acc_rest_I, np.array([0, 0, 1.0]))

        # ---- magnetometer consistency (indoors the field is usually distorted -> excluded)
        mag_ok, mag_note = False, "absent"
        if len(aux["mag"]):
            n = np.linalg.norm(aux["mag"], axis=1)
            cv = float(n.std() / n.mean())
            mag_note = "|B| varies %.0f%% over the run" % (100 * cv)
            mag_ok = bool(cv < 0.05)
            log("   magnetometer: %s -> %s" % (mag_note, "used" if mag_ok else "EXCLUDED (distorted field)"))

        # ---- front end: KISS-ICP with IMU-rotation + wheel-translation prediction
        cfg = load_config(None)
        cfg.data.max_range, cfg.data.min_range, cfg.data.deskew = a.max_range, a.min_range, True
        cfg.mapping.voxel_size = a.voxel or a.max_range / 100.0
        wt, wp, wq = aux["odom_t"], aux["odom_p"], aux["odom_q"]

        # ---- extrinsic checks against the data (the bag's TF is not trusted blindly)
        ext = {}
        if not a.trust_tf:
            corr, det = check_lidar_yaw(reader, lidar_conn, cfg, T_BL, wt, wp, wq)
            ext["lidar_yaw"] = det
            msg = "   lidar yaw check: lidar %.2f m vs wheels %.2f m of travel, TF yaw off by %.1f deg" % (
                det["lidar_travel_m"], det["wheel_travel_m"], det["yaw_correction_deg"])
            if abs(det["yaw_correction_deg"]) > 5 and det["wheel_travel_m"] > 1.0:
                T_BL[:3, :3] = Rz(corr) @ T_BL[:3, :3]
                T_LB = inv(T_BL)
                msg += " -> CORRECTED (yaw applied about base z; TF translation kept)"
                ext["lidar_yaw"]["applied"] = True
            else:
                ext["lidar_yaw"]["applied"] = False
            log(msg)
            ext["imu_axes"] = check_imu_axes(imu_B, wt, wp, wq, t0)
            log("   IMU axis check vs wheels: yaw-rate corr %.2f, forward-acc corr %.2f (both should be > 0)"
                % (ext["imu_axes"]["yaw_rate_corr"], ext["imu_axes"]["fwd_acc_corr"]))
            if ext["imu_axes"]["yaw_rate_corr"] < 0:
                log("   WARNING: IMU yaw axis disagrees with the wheels; check T_base_imu")
        odo = KissICP(cfg)
        scan_t, P_L, kf, kf_pts, offs, npts = [], [], [], [], [], []
        last_kf_T, travel, travel_at = None, 0.0, []
        n_total = lidar_conn[0].msgcount
        log("== lidar front end (%d scans, voxel %.2f m, range %.1f-%.1f m)"
            % (n_total, cfg.mapping.voxel_size, a.min_range, a.max_range))
        for i, (c, _, raw) in enumerate(reader.messages(connections=lidar_conn)):
            if a.max_scans and i >= a.max_scans:
                break
            m = reader.deserialize(raw, c.msgtype)
            pts, ptimes, t_mid, off_ms = scan_points(m)
            offs.append(off_ms)
            if scan_t and not a.no_predict:
                ta, tb = (scan_t[-1] - t0) / 1e9, (t_mid - t0) / 1e9
                dB = np.eye(4)
                dB[:3, :3] = imu_B.delta_R(ta, tb, R_BI @ bg_I)
                Wa, Wb = interp_pose(wt, wp, wq, scan_t[-1]), interp_pose(wt, wp, wq, t_mid)
                if Wa is not None and Wb is not None:
                    dB[:3, 3] = (inv(Wa) @ Wb)[:3, 3]
                odo.last_delta = T_LB @ dB @ T_BL            # prediction in the lidar frame
            frame, _ = odo.register_frame(pts, ptimes)
            Pk = odo.last_pose.copy()
            if P_L:
                travel += float(np.linalg.norm(Pk[:3, 3] - P_L[-1][:3, 3]))
            scan_t.append(t_mid)
            P_L.append(Pk)
            travel_at.append(travel)
            npts.append(len(pts))
            if last_kf_T is None or np.linalg.norm((inv(last_kf_T) @ Pk)[:3, 3]) > a.kf_dist or \
                    math.degrees(rot_angle((inv(last_kf_T) @ Pk)[:3, :3])) > a.kf_angle:
                kf.append(i)
                kf_pts.append(voxel_down_sample(frame, 0.2).astype(np.float32))
                last_kf_T = Pk
            if i % 250 == 0:
                log("   scan %d/%d  pts %d  travel %.1f m  keyframes %d" % (i, n_total, len(pts), travel, len(kf)))
    N = len(scan_t)
    scan_t = np.array(scan_t, dtype=np.int64)
    log("   %d scans, %d keyframes, lidar travel %.1f m; scan time - header stamp: median %.1f ms"
        % (N, len(kf), travel, float(np.median(offs))))

    # ---- shared setup for the graph
    Diag = gtsam.noiseModel.Diagonal.Sigmas
    huber = lambda s: gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(1.345), Diag(s))
    cauchy = lambda s: gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Cauchy.Create(1.0), Diag(s))
    W_B0 = np.eye(4)
    W_B0[:3, :3] = R_W_B0
    X_init = [W_B0 @ T_BL @ inv(P_L[0]) @ P_L[k] @ T_LB for k in range(N)]   # base_link poses
    tsec = (scan_t - t0) / 1e9

    # Wheel odometry scale (tyre radius / skid): calibrated against the lidar's horizontal travel over
    # 2 s windows while moving. The lidar measures distance geometrically; the wheels do not.
    wheel_scale, ws = 1.0, []
    step = 20
    for k in range(step, N, step):
        Wa, Wb = interp_pose(wt, wp, wq, scan_t[k - step]), interp_pose(wt, wp, wq, scan_t[k])
        if Wa is None or Wb is None:
            continue
        dw = np.linalg.norm((inv(Wa) @ Wb)[:2, 3])
        dl = np.linalg.norm((inv(X_init[k - step]) @ X_init[k])[:2, 3])
        if dw > 0.2:
            ws.append(dl / dw)
    if len(ws) > 10 and not a.no_wheel_scale:
        wheel_scale = float(np.median(ws))
    log("   wheel scale vs lidar: median %.4f over %d windows (IQR %.4f-%.4f) -> %s" % (
        float(np.median(ws)) if ws else float("nan"), len(ws),
        *(np.percentile(ws, [25, 75]) if ws else (float("nan"),) * 2),
        "applied %.4f" % wheel_scale if wheel_scale != 1.0 else "not applied"))

    # The driver's orientation is only usable if its gravity direction agrees with the accelerometer
    # at rest (some drivers publish it in another convention or with permuted components).
    AttFactor = getattr(gtsam, "AttitudeFactorPose3", None) or getattr(gtsam, "Pose3AttitudeFactor", None)
    att_note, att_ok = "absent", False
    if np.abs(aux["ori"]).sum() > 0:
        up_acc = acc_rest_I / np.linalg.norm(acc_rest_I)
        j0 = int(np.searchsorted(imu_I.t, 0.5 * rest_end)) if rest_end > 0 else 0
        Rq = q2R(aux["ori"][j0])
        best = min(math.degrees(math.acos(np.clip(np.dot(v, up_acc), -1, 1)))
                   for v in (Rq.T @ [0, 0, 1.0], Rq @ [0, 0, 1.0], Rq.T @ [0, 0, -1.0], Rq @ [0, 0, -1.0]))
        att_ok = best < 5.0 and AttFactor is not None
        att_note = "driver orientation vs accelerometer gravity at rest: %.1f deg -> %s" % (
            best, "used" if att_ok else "EXCLUDED (inconsistent convention)")
        log("   " + att_note)

    if use_imu:
        bias0 = gtsam.imuBias.ConstantBias(np.zeros(3), bg_I)
        pp = gtsam.PreintegrationCombinedParams.MakeSharedU(a.gravity)
        s_ = a.imu_noise_inflate
        pp.setGyroscopeCovariance((s_ * g_dens) ** 2 * np.eye(3))
        pp.setAccelerometerCovariance((s_ * a_dens) ** 2 * np.eye(3))
        pp.setIntegrationCovariance(1e-8 * np.eye(3))
        pp.setBiasOmegaCovariance((s_ * 1e-4) ** 2 * np.eye(3))
        pp.setBiasAccCovariance((s_ * 1e-3) ** 2 * np.eye(3))
        for name in ("setBiasAccOmegaInit", "setBiasAccOmegaInt"):
            if hasattr(pp, name):
                getattr(pp, name)(1e-5 * np.eye(6))
        pp.setBodyPSensor(gtsam.Pose3(T_BI))
        pims = []
        for k in range(1, N):
            pim = gtsam.PreintegratedCombinedMeasurements(pp, bias0)
            for acc_, gyr_, dt in imu_I.segments(tsec[k - 1], tsec[k]):
                pim.integrateMeasurement(acc_, gyr_, dt)
            pims.append(pim)

    def solve(loop_list, start=None, tag=""):
        """Build the graph with the given loop closures and optimise it. `start`: initial Values."""
        graph, init_v, kinds = gtsam.NonlinearFactorGraph(), gtsam.Values(), {}
        cnt = dict(wheel=0, attitude=0, loop=0)

        def add(kind, factor):
            kinds.setdefault(kind, []).append(graph.size())
            graph.add(factor)
        add("prior", gtsam.PriorFactorPose3(X(0), gtsam.Pose3(X_init[0]), Diag(np.array([1e-3] * 6))))
        for k in range(N):
            init_v.insert(X(k), start.atPose3(X(k)) if start else gtsam.Pose3(X_init[k]))
        if use_imu:
            add("prior", gtsam.PriorFactorVector(V(0), np.zeros(3), Diag(np.array([0.05] * 3))))
            add("prior", gtsam.PriorFactorConstantBias(B(0), bias0, Diag(np.array([0.5] * 3 + [0.01] * 3))))
            for k in range(N):
                if start:
                    init_v.insert(V(k), start.atVector(V(k)))
                    init_v.insert(B(k), start.atConstantBias(B(k)))
                else:
                    vk = (X_init[min(k + 1, N - 1)][:3, 3] - X_init[max(k - 1, 0)][:3, 3]) / max(
                        tsec[min(k + 1, N - 1)] - tsec[max(k - 1, 0)], 1e-3)
                    init_v.insert(V(k), vk)
                    init_v.insert(B(k), bias0)
            for k in range(1, N):
                pim = pims[k - 1]
                if pim.deltaTij() > 0:
                    add("imu", gtsam.CombinedImuFactor(X(k - 1), V(k - 1), X(k), V(k), B(k - 1), B(k), pim))
                else:                                    # no IMU sample in this interval: keep V and B linked
                    add("bias_tie", gtsam.BetweenFactorConstantBias(B(k - 1), B(k), gtsam.imuBias.ConstantBias(),
                                                                   Diag(np.array([1e-3] * 6))))
                    add("bias_tie", gtsam.BetweenFactorVector(V(k - 1), V(k), np.zeros(3), Diag(np.array([0.5] * 3))))
        # lidar odometry between factors (in base_link)
        for k in range(1, N):
            dL = inv(P_L[k - 1]) @ P_L[k]
            add("lidar", gtsam.BetweenFactorPose3(X(k - 1), X(k), gtsam.Pose3(T_BL @ dL @ T_LB),
                                                  huber(np.array([0.005] * 3 + [0.03] * 3))))
        # wheel odometry: planar, yaw slips on a skid-steer, no z / roll / pitch information
        if not a.no_wheel:
            for k in range(1, N):
                Wa, Wb = interp_pose(wt, wp, wq, scan_t[k - 1]), interp_pose(wt, wp, wq, scan_t[k])
                if Wa is None or Wb is None:
                    continue
                dW = inv(Wa) @ Wb
                dW[:3, 3] *= wheel_scale
                # sigmas (rx ry rz tx ty tz) per 0.1 s: roll/pitch/z unobserved by wheels, yaw slips,
                # forward travel looser than the lidar's 3 cm, sideways tight (skid-steer cannot slide)
                add("wheel", gtsam.BetweenFactorPose3(X(k - 1), X(k), gtsam.Pose3(dW),
                                                      cauchy(np.array([0.2, 0.2, 0.02, 0.05, 0.01, 0.1]))))
                cnt["wheel"] += 1
        # IMU driver orientation: gravity direction only (roll/pitch), every 10th scan, weak
        if not a.no_attitude and att_ok:
            it, iq = aux["imu_t"], aux["ori"]
            for k in range(0, N, 10):
                j = int(np.clip(np.searchsorted(it, scan_t[k]), 0, len(it) - 1))
                down_B = R_BI @ (q2R(iq[j]).T @ np.array([0, 0, -1.0]))
                add("attitude", AttFactor(X(k), gtsam.Unit3(np.array([0, 0, -1.0])),
                                          gtsam.noiseModel.Isotropic.Sigma(2, 0.05), gtsam.Unit3(down_B)))
                cnt["attitude"] += 1
        for l in loop_list:
            add("loop", gtsam.BetweenFactorPose3(X(l["kj"]), X(l["ki"]), gtsam.Pose3(T_BL @ l["M"] @ T_LB),
                                                 cauchy(np.array([0.01] * 3 + [0.03] * 3))))
            cnt["loop"] += 1
        log("== factor graph%s: states %d; factors %s" % (tag, N, {k: len(v) for k, v in kinds.items()}))
        params = gtsam.LevenbergMarquardtParams()
        params.setMaxIterations(100)
        e0 = graph.error(init_v)
        for kind, idx in kinds.items():                     # where the initial guess disagrees
            e = np.array([graph.at(i).error(init_v) for i in idx])
            log("   initial %-9s sum %.1f  median %.3f  max %.3f" % (kind, e.sum(), np.median(e), e.max()))
        r = gtsam.LevenbergMarquardtOptimizer(graph, init_v, params).optimize()
        e1 = graph.error(r)
        log("   total error %.1f -> %.1f" % (e0, e1))
        return r, graph, kinds, e0, e1, cnt

    # ---- loop closures, searched on a trajectory already corrected by IMU + wheels (pass 1)
    trees = {}

    def search_loops(Xb):
        """Keyframe-window (5 KFs) to submap (9 KFs around an earlier keyframe) ICP, initialised from
        the base poses Xb. Returns every tested candidate (accepted flag set)."""
        rel = lambda j, i: T_LB @ inv(Xb[j]) @ Xb[i] @ T_BL     # lidar pose of scan i in lidar frame j
        kf_arr = np.array(kf)
        kf_pos = np.array([Xb[k][:3, 3] for k in kf])
        kf_time = (scan_t[kf_arr] - t0) / 1e9
        out = []
        for qi in range(len(kf)):
            ki = kf[qi]
            old = np.where(kf_time < kf_time[qi] - a.loop_min_gap)[0]
            if not len(old):
                continue
            d = np.linalg.norm(kf_pos[old] - kf_pos[qi], axis=1)
            radius = a.loop_radius + 0.03 * (travel_at[ki] - np.array([travel_at[kf[j]] for j in old]))
            sel = d < radius
            if not sel.any():
                continue
            q = np.concatenate([kf_pts[n].astype(np.float64) @ rel(ki, kf[n])[:3, :3].T + rel(ki, kf[n])[:3, 3]
                                for n in range(max(qi - 2, 0), min(qi + 3, len(kf)))])
            best = None
            for cj in old[sel][np.argsort(d[sel])][:3]:
                kj = kf[cj]
                if cj not in trees:
                    sub = np.concatenate([kf_pts[n].astype(np.float64) @ rel(kj, kf[n])[:3, :3].T
                                          + rel(kj, kf[n])[:3, 3] for n in range(max(cj - 4, 0), min(cj + 5, len(kf)))])
                    trees[cj] = (cKDTree(sub), sub)
                tree, sub = trees[cj]
                init = rel(kj, ki)
                f0, _ = fitness(q, init, tree, 0.2)
                M = register(q, sub, init, dict(iters=100), 1.0, 0.3)
                M = register(q, sub, M, dict(iters=100), 0.4, 0.1)
                f1, rmse = fitness(q, M, tree, 0.2)
                corr = inv(init) @ M
                rad = float(radius[list(old).index(cj)])
                ok = bool(f1 > a.loop_min_fitness and f1 >= f0 and rmse < a.loop_max_rmse
                          and np.linalg.norm(corr[:3, 3]) < max(rad, 0.5)
                          and math.degrees(rot_angle(corr[:3, :3])) < 10 and len(q) > 1000)
                rec = dict(kj=kj, ki=ki, fit_before=f0, fit_after=f1, rmse=rmse,
                           corr_m=float(np.linalg.norm(corr[:3, 3])),
                           corr_deg=math.degrees(rot_angle(corr[:3, :3])), accepted=False, M=M)
                out.append(rec)
                if ok and (best is None or f1 > best["fit_after"]):
                    best = rec
            if best:
                best["accepted"] = True
        return out

    if a.no_loops:
        res, graph, kinds, err0, err1, counts = solve([])
        loops = []
    else:
        res1, *_ = solve([], tag=" pass 1 (no loops)")
        Xb1 = [res1.atPose3(X(k)).matrix() for k in range(N)]
        log("== loop closure search on the pass-1 trajectory")
        loops = search_loops(Xb1)
        acc = [l for l in loops if l["accepted"]]
        log("   candidates tested %d, accepted %d" % (len(loops), len(acc)))
        with open(a.out / "loops.csv", "w") as f:
            f.write("scan_j,scan_i,t_j_s,t_i_s,fitness_before,fitness_after,rmse_m,correction_m,correction_deg,accepted\n")
            for l in loops:
                f.write("%d,%d,%.2f,%.2f,%.3f,%.3f,%.4f,%.3f,%.2f,%d\n" % (
                    l["kj"], l["ki"], (scan_t[l["kj"]] - t0) / 1e9, (scan_t[l["ki"]] - t0) / 1e9,
                    l["fit_before"], l["fit_after"], l["rmse"], l["corr_m"], l["corr_deg"], l["accepted"]))
        res, graph, kinds, err0, err1, counts = solve(acc, start=res1, tag=" pass 2 (with loops)")
    n_wheel, n_att, n_loop = counts["wheel"], counts["attitude"], counts["loop"]

    # ---- outputs
    def write_traj(path, mats):
        with open(path, "w") as f:
            for s, M in zip(scan_t, mats):
                f.write("%d %.6f %.6f %.6f %.9f %.9f %.9f %.9f\n" % (s, *M[:3, 3], *R2q(M[:3, :3])))
    Xf = [res.atPose3(X(k)).matrix() for k in range(N)]
    write_traj(a.out / "fused_poses.txt", Xf)
    write_traj(a.out / "frontend_poses.txt", X_init)
    lines = []
    for kind, idx in kinds.items():
        e = np.array([graph.at(i).error(res) for i in idx])
        lines.append("%-9s n=%5d  error mean %.3f  median %.3f  p95 %.3f  max %.3f"
                     % (kind, len(e), e.mean(), np.median(e), np.percentile(e, 95), e.max()))
    (a.out / "factors.txt").write_text("\n".join(lines) + "\n")
    for s in lines:
        log("   " + s)
    if use_imu:
        with open(a.out / "biases.txt", "w") as f:
            f.write("# t_s acc_bias_xyz gyro_bias_xyz\n")
            for k in range(0, N, 10):
                b = res.atConstantBias(B(k))
                f.write("%.2f %s %s\n" % (tsec[k], " ".join("%.5f" % v for v in b.accelerometer()),
                                          " ".join("%.6f" % v for v in b.gyroscope())))

    # consistency of accepted loops with the fused trajectory (inlier fit at the fused relative pose)
    loop_fit = []
    for l in [l for l in loops if l["accepted"]]:
        M = T_LB @ inv(Xf[l["kj"]]) @ Xf[l["ki"]] @ T_BL
        cj = kf.index(l["kj"])
        if cj in trees:
            loop_fit.append(fitness(kf_pts[kf.index(l["ki"])].astype(np.float64), M, trees[cj][0], 0.2))

    Pf = np.array([M[:3, 3] for M in Xf])
    P0 = np.array([M[:3, 3] for M in X_init])
    shift = np.linalg.norm(Pf - (P0 - P0[0] + Pf[0]), axis=1)
    info = {
        "method": "GTSAM factor graph: KISS-ICP lidar odometry + lidar loop closures + IMU preintegration"
                  " + wheel odometry + IMU roll/pitch",
        "frame": a.base_frame, "scans": N, "keyframes": len(kf),
        "imu_factors": use_imu, "wheel_factors": n_wheel, "attitude_factors": n_att, "loop_factors": n_loop,
        "magnetometer": ("used" if mag_ok else "excluded: " + mag_note),
        "imu_driver_orientation": att_note,
        "wheel_scale_applied": wheel_scale,
        "extrinsic_checks": ext,
        "T_base_lidar_used": T_BL.tolist(), "T_base_imu_used": T_BI.tolist(),
        "validation_only": "/platform/odom/filtered, /tf odom->base_link, /imu/rpy, /platform/joint_states",
        "camera_topics_read": "none",
        "path_length_m": float(np.linalg.norm(np.diff(Pf, axis=0), axis=1).sum()),
        "end_to_start_m": float(np.linalg.norm(Pf[-1] - Pf[0])),
        "z_range_m": float(Pf[:, 2].max() - Pf[:, 2].min()),
        "fused_vs_frontend_max_shift_m": float(shift.max()),
        "loop_fitness_after_fusion_median": float(np.median([f for f, _ in loop_fit])) if loop_fit else None,
        "loop_rmse_after_fusion_median_m": float(np.median([r for _, r in loop_fit])) if loop_fit else None,
        "graph_error_initial": float(err0), "graph_error_final": float(err1),
        "rest_end_s": rest_end, "gyro_bias_rest": bg_I.tolist(), "acc_rest_norm": float(np.linalg.norm(acc_rest_I)),
        "scan_time_minus_header_ms_median": float(np.median(offs)),
        "kiss_voxel_m": cfg.mapping.voxel_size, "range_m": [a.min_range, a.max_range],
        "runtime_s": time.time() - t_start,
    }
    (a.out / "fusion_info.json").write_text(json.dumps(info, indent=1))
    (a.out / "lidar_odometry_info.txt").write_text(
        "frame: %s\nmethod: %s\nscans: %d\npath_length_m: %.2f\n" % (a.base_frame, info["method"], N,
                                                                    info["path_length_m"]))
    for k, v in info.items():
        log("   %s: %s" % (k, v))
    (a.out / "fusion_report.txt").write_text("\n".join(log_lines) + "\n")

    # ---- plots (validation: robot's own odometry vs our fused estimate)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def to_world(t_arr, Pw, Qw):
            ok = [interp_pose(t_arr, Pw, Qw, s) for s in scan_t]
            i0 = next(i for i, x in enumerate(ok) if x is not None)
            A = Xf[i0] @ inv(ok[i0])                      # align at the first common time
            return np.array([(A @ x)[:3, 3] if x is not None else [np.nan] * 3 for x in ok])
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(*P0[:, :2].T, lw=0.8, label="lidar front end (no graph)")
        ax.plot(*to_world(wt, wp, wq)[:, :2].T, lw=0.8, label="/platform/odom (wheels)")
        if len(aux["filt_t"]):
            ax.plot(*to_world(aux["filt_t"], aux["filt_p"], aux["filt_q"])[:, :2].T, lw=0.8,
                    label="/platform/odom/filtered (robot EKF)")
        ax.plot(*Pf[:, :2].T, "k", lw=1.5, label="fused reference")
        for l in [l for l in loops if l["accepted"]]:
            ax.plot([Pf[l["kj"], 0], Pf[l["ki"], 0]], [Pf[l["kj"], 1], Pf[l["ki"], 1]], "r-", lw=0.6)
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.legend(loc="best", fontsize=8)
        ax.set_title("Husky reference (red: accepted loop closures)")
        fig.savefig(a.out / "fusion_topview.png", dpi=150, bbox_inches="tight")
        fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        ax[0].plot(tsec, Pf[:, 2], label="fused z")
        ax[0].plot(tsec, P0[:, 2] - P0[0, 2] + Pf[0, 2], label="front-end z")
        ax[0].set_ylabel("z [m]")
        ax[0].legend()
        ax[1].plot(tsec, shift)
        ax[1].set_ylabel("fused - front end [m]")
        ax[1].set_xlabel("t [s]")
        fig.savefig(a.out / "fusion_timeseries.png", dpi=120, bbox_inches="tight")
    except Exception as e:                                  # plots are a convenience only
        log("   (plots skipped: %r)" % e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
