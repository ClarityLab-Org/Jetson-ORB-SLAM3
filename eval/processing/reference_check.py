#!/usr/bin/env python3
"""How good is the fused reference? Compare it with every other trajectory we can build without the
camera (plan/husky_stage1_runbook.md, step 4).

usage: reference_check.py REF_DIR SEQ_DIR [--kiss-slam DIR] [--kiss-icp DIR] [--odom DIR] [--ablations DIR ...]

  REF_DIR    fuse_reference.py output (fused_poses.txt, frontend_poses.txt, fusion_info.json)
  SEQ_DIR    bag_to_euroc.py output (tf_static.json: to bring lidar-frame trajectories to base_link)
  --kiss-slam, --kiss-icp   lidar_slam.py / lidar_odometry.py outputs (lidar frame)
  --odom     bag_odom.py output (platform_odom.txt, platform_odom_filtered.txt; base_link)
  --ablations  fuse_reference.py outputs run with --no-imu / --no-wheel / --no-loops (sensitivity)

For each alternative: SE(3)-aligned RMS / max distance to the reference at the reference times
(alternative interpolated, no extrapolation), and path length / end-to-start gap.
Writes REF_DIR/reference_check.md and reference_check.png.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "benchmark"))
from evaluate import umeyama  # noqa: E402
from make_gt import T, chain, slerp  # noqa: E402


def load(path):
    a = np.loadtxt(path, ndmin=2)
    return a[:, 0].astype(np.int64), a[:, 1:4], a[:, 4:8]


def at_times(t, P, Q, ts, T_post=np.eye(4)):
    """Positions of (pose @ T_post) interpolated at times ts; NaN outside the covered span."""
    out = np.full((len(ts), 3), np.nan)
    for n, s in enumerate(ts):
        i = int(np.searchsorted(t, s))
        if 0 < i < len(t) and (t[i] - t[i - 1]) < 5e8:
            u = (s - t[i - 1]) / (t[i] - t[i - 1])
            out[n] = (T((1 - u) * P[i - 1] + u * P[i], slerp(Q[i - 1], Q[i], u)) @ T_post)[:3, 3]
    return out


def compare(ref_p, alt_p):
    ok = ~np.isnan(alt_p).any(1)
    _, R, tr = umeyama(alt_p[ok], ref_p[ok], False)
    e = np.linalg.norm(alt_p[ok] @ R.T + tr - ref_p[ok], axis=1)
    pl = np.linalg.norm(np.diff(alt_p[ok], axis=0), axis=1).sum()
    return dict(n=int(ok.sum()), rms=float(np.sqrt(np.mean(e ** 2))), max=float(e.max()),
                path=float(pl), end_gap=float(np.linalg.norm(alt_p[ok][-1] - alt_p[ok][0])), aligned=alt_p[ok] @ R.T + tr,
                ok=ok)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ref", type=Path)
    ap.add_argument("seq", type=Path)
    ap.add_argument("--kiss-slam", type=Path)
    ap.add_argument("--kiss-icp", type=Path)
    ap.add_argument("--odom", type=Path)
    ap.add_argument("--ablations", type=Path, nargs="*", default=[])
    a = ap.parse_args()

    tf = json.loads((a.seq / "tf_static.json").read_text())
    info = json.loads((a.ref / "fusion_info.json").read_text())
    rt, rp, rq = load(a.ref / "fused_poses.txt")
    ref = rp
    cands = [("lidar front end (fusion input, no graph)", a.ref / "frontend_poses.txt", np.eye(4))]

    def lidar_to_base(lidar_frame):
        """T_lidar_base: the extrinsic the fusion verified (and corrected if needed), not the raw TF."""
        if "T_base_lidar_used" in info:
            return np.linalg.inv(np.array(info["T_base_lidar_used"]))
        return chain(tf, lidar_frame, info["frame"])
    if a.kiss_slam:
        lf = dict(l.split(": ", 1) for l in (a.kiss_slam / "lidar_odometry_info.txt").read_text().splitlines())["frame"]
        cands.append(("KISS-SLAM (lidar only, own loop closure)", a.kiss_slam / "lidar_poses.txt",
                      lidar_to_base(lf)))
    if a.kiss_icp:
        lf = dict(l.split(": ", 1) for l in (a.kiss_icp / "lidar_odometry_info.txt").read_text().splitlines())["frame"]
        cands.append(("KISS-ICP (lidar only, no loop closure)", a.kiss_icp / "lidar_poses.txt",
                      lidar_to_base(lf)))
    if a.odom:
        cands.append(("/platform/odom (wheels)", a.odom / "platform_odom.txt", np.eye(4)))
        cands.append(("/platform/odom/filtered (robot EKF: wheels + IMU)", a.odom / "platform_odom_filtered.txt",
                      np.eye(4)))
    for d in a.ablations:
        fi = json.loads((d / "fusion_info.json").read_text())
        tag = []
        if not fi.get("imu_factors"):
            tag.append("no IMU")
        if not fi.get("wheel_factors"):
            tag.append("no wheels")
        if not fi.get("loop_factors"):
            tag.append("no loops")
        cands.append(("fusion ablation: %s" % (", ".join(tag) or d.name), d / "fused_poses.txt", np.eye(4)))

    rows, curves = [], []
    for name, path, Tpost in cands:
        if not path.is_file():
            rows.append("| %s | missing (%s) | | | | |" % (name, path))
            continue
        t, P, Q = load(path)
        r = compare(ref, at_times(t, P, Q, rt, Tpost))
        rows.append("| %s | %d | %.3f | %.3f | %.1f | %.2f |" % (name, r["n"], r["rms"], r["max"], r["path"], r["end_gap"]))
        curves.append((name, r))
    ref_path = float(np.linalg.norm(np.diff(ref, axis=0), axis=1).sum())
    md = ["# Reference trajectory check", "",
          "Reference: `%s` (%s)." % (a.ref / "fused_poses.txt", info["method"]), "",
          "| Fusion input | Value |", "|---|---|"]
    for k in ("imu_factors", "wheel_factors", "attitude_factors", "loop_factors", "magnetometer",
              "validation_only", "camera_topics_read", "loop_fitness_after_fusion_median",
              "loop_rmse_after_fusion_median_m", "path_length_m", "end_to_start_m", "z_range_m",
              "fused_vs_frontend_max_shift_m", "imu_driver_orientation", "extrinsic_checks", "acc_rest_norm", "scan_time_minus_header_ms_median"):
        md.append("| %s | %s |" % (k, info.get(k)))
    md += ["", "Reference path length %.1f m, end-to-start %.2f m." % (ref_path, np.linalg.norm(ref[-1] - ref[0])), "",
           "Distance of each alternative to the reference after SE(3) alignment (at the reference times):", "",
           "| Trajectory | Poses | RMS [m] | Max [m] | Path [m] | End-start [m] |", "|---|---|---|---|---|---|"]
    md += rows
    md += ["", "Reading the table:",
           "- Lidar-only estimates (front end, KISS-ICP, KISS-SLAM) are not independent of the reference (they share",
           "  the lidar). Agreement shows the fusion did not distort the lidar geometry; the difference to the front",
           "  end is what the IMU, wheels and loop closures corrected.",
           "- The robot's own odometry is independent of the lidar; it drifts, so expect decimetres to metres.",
           "- The ablation rows show how sensitive the reference is to each input. Their RMS is a practical",
           "  uncertainty bound for the reference: ORB-SLAM3 differences smaller than it are not resolvable."]
    (a.ref / "reference_check.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(*ref[:, :2].T, "k", lw=1.6, label="fused reference")
        for name, r in curves:
            ax.plot(*r["aligned"][:, :2].T, lw=0.8, label="%s (RMS %.2f m)" % (name, r["rms"]))
        ax.set_aspect("equal")
        ax.legend(fontsize=7)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        fig.savefig(a.ref / "reference_check.png", dpi=150, bbox_inches="tight")
    except Exception as e:                                      # noqa: BLE001
        print("(plot skipped: %r)" % e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
