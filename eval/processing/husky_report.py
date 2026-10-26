#!/usr/bin/env python3
"""Our Jetson ORB-SLAM3 (camera only) vs the fused non-camera reference: comparison report
(plan/husky_stage1_runbook.md, step 6).

usage: husky_report.py --results DIR --seq SEQ_DIR --reference REF_DIR --out OUT_DIR [--timing DIR]

  --results  run_matrix.py ext results (…/ext/ext_s/<seq>/<arm>/run<k>/), scored by evaluate.py
  --timing   optional: the same suite run with the REGISTER_TIMES build (per-stage ms, true FPS)
  --seq      ~/datasets/ext/<seq>  (times.txt, gt.csv, tf_static.json, calib.json)
  --reference  the reference folder: eval/results/husky/reference (fusion/, crosschecks/odom/)
             or a prepare_husky.sh work folder ~/datasets/ext_work/<seq> (fused/, odom/)
  --out      report folder: HUSKY_REPORT.md + PNG plots + runs.json

Accuracy is scored on the full-frame trajectory (f_run.txt) and the keyframe trajectory (kf_run.txt,
the paper's Table 1 metric) against gt.csv (reference in the left IR optical frame at each image time).
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics as st
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "benchmark"))
from evaluate import associate, ate_rmse, load_gt_csv, load_tum, relative_errors, umeyama  # noqa: E402
from make_gt import T, chain, slerp  # noqa: E402

ARMS = [("gpu", "GPU ORB (default)"), ("cpu", "CPU ORB (CPU_ORB=1)"), ("gpu_pipe", "Pipelined (PIPELINE_FE=1)")]
TEGRA = re.compile(r"RAM (\d+)/(\d+)MB.*?CPU \[([^\]]*)\].*?GR3D_FREQ (\d+)%.*?tj@([\d.]+)C.*?VDD_IN (\d+)mW"
                   r".*?VDD_CPU_GPU_CV (\d+)mW.*?VDD_SOC (\d+)mW")


def med(v):
    v = [x for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return st.median(v) if v else None


def rng(v):
    v = [x for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return (min(v), max(v)) if v else (None, None)


def f(v, nd=2):
    return "–" if v is None else ("%.*f" % (nd, v))


def fr(vals, nd=2):
    m, (lo, hi) = med(vals), rng(vals)
    return "–" if m is None else "%s (%s–%s)" % (f(m, nd), f(lo, nd), f(hi, nd))


def crash_class(txt):
    if "SO3::exp failed" in txt:
        return "Sophus NaN (mid-run)"
    if "pure virtual" in txt:
        return "pure virtual (teardown)"
    if "gdk" in txt or "X Window" in txt:
        return "X error (teardown)"
    if "Bus error" in txt:
        return "bus error (teardown)"
    if "Segmentation fault" in txt:
        return "segfault (teardown)"
    return "clean"


def tegrastats(path):
    rows = []
    if not path.is_file():
        return {}
    for line in path.read_text(errors="replace").splitlines():
        m = TEGRA.search(line)
        if not m:
            continue
        cpu = [float(c.split("%")[0]) if c.split("%")[0].replace(".", "").isdigit() else 0.0
               for c in m.group(3).split(",")]
        rows.append(dict(ram=int(m.group(1)), cpu=cpu, gr3d=int(m.group(4)), tj=float(m.group(5)),
                         vin=int(m.group(6)), vcg=int(m.group(7)), vsoc=int(m.group(8))))
    busy = [i for i, r in enumerate(rows) if sum(u > 25 for u in r["cpu"]) >= 2 or r["gr3d"] > 5]
    if not busy:
        return {}
    W = rows[busy[0]:busy[-1] + 1]
    a = lambda k: np.array([r[k] for r in W], float)
    return dict(window_s=len(W), vin_w=a("vin").mean() / 1e3, vin_peak_w=a("vin").max() / 1e3,
                vcg_w=a("vcg").mean() / 1e3, vsoc_w=a("vsoc").mean() / 1e3, energy_j=a("vin").sum() / 1e3,
                cpu_pct=float(np.mean([sum(r["cpu"]) for r in W])), gr3d_pct=a("gr3d").mean(),
                ram_peak_mb=max(r["ram"] for r in rows), ram_start_mb=W[0]["ram"], tj_max=max(r["tj"] for r in rows),
                pre_vin_w=float(np.mean([r["vin"] for r in rows[:max(busy[0], 1)]])) / 1e3)


def rpe(T_est, T_gt, t_gt, lengths):
    """Translation error per distance travelled (%), rotation (deg/m), on index-aligned poses."""
    out = {}
    for L in lengths:
        tr, rr, n = relative_errors(T_est, T_gt, [L], 1)
        out[L] = (tr, rr, n)
    return out


def load_pose_file(path):
    a = np.loadtxt(path, ndmin=2)
    return a[:, 0].astype(np.int64), a[:, 1:4], a[:, 4:8]


def odom_in_camera(path, T_base_cam, t_img_ns):
    """Robot odometry (base_link) as camera poses at the image times (for the context baseline)."""
    t, P, Q = load_pose_file(path)
    ts, Ts = [], []
    for s in t_img_ns:
        i = int(np.searchsorted(t, s))
        if 0 < i < len(t) and (t[i] - t[i - 1]) < 5e8:
            u = (s - t[i - 1]) / (t[i] - t[i - 1])
            ts.append(s / 1e9)
            Ts.append(T((1 - u) * P[i - 1] + u * P[i], slerp(Q[i - 1], Q[i], u)) @ T_base_cam)
    return np.array(ts), np.array(Ts)


def score(t_est, T_est, t_gt, T_gt):
    ie, ig = associate(t_est, t_gt, 0.02)
    if len(ie) < 10:
        return None
    pe, pg = T_est[ie, :3, 3], T_gt[ig, :3, 3]
    se3, _ = ate_rmse(pe, pg, False)
    sim3, scale = ate_rmse(pe, pg, True)
    s, R, tr = umeyama(pe, pg, False)
    err = np.linalg.norm(pe @ R.T + tr - pg, axis=1)
    rp = rpe(T_est[ie], T_gt[ig], t_gt[ig], [1, 5, 10])
    # Horizontal / vertical split of the SE(3)-aligned error. The reference world is gravity-aligned
    # (z up); ORB-SLAM3's world is its first camera frame, so the split is done after alignment.
    res_ = pe @ R.T + tr - pg
    e2 = np.linalg.norm(res_[:, :2], axis=1)
    ez = res_[:, 2]
    # scale check without any alignment: straight-line distance ratios over ~10 s windows
    seg = []
    for k in range(0, len(pe) - 150, 150):
        de = np.linalg.norm(pe[k + 150] - pe[k])
        if de > 0.8:
            seg.append(np.linalg.norm(pg[k + 150] - pg[k]) / de)
    return dict(n=len(ie), ate_se3_cm=se3 * 100, ate_sim3_cm=sim3 * 100, scale=scale, max_cm=float(err.max() * 100),
                ate_xy_cm=float(np.sqrt(np.mean(e2 ** 2)) * 100), z_rms_cm=float(np.sqrt(np.mean(ez ** 2)) * 100),
                seg_scale=float(np.median(seg)) if seg else None,
                err=err, t=t_gt[ig], aligned=pe @ R.T + tr, rpe=rp)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--timing", type=Path)
    ap.add_argument("--seq", type=Path, required=True)
    ap.add_argument("--reference", "--work", dest="work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    name = a.seq.name
    # two layouts: the repository copy (fusion/, crosschecks/odom/) or the PC work folder (fused/, odom/)
    fdir = a.work / "fusion" if (a.work / "fusion").is_dir() else a.work / "fused"
    odir = a.work / "crosschecks" / "odom" if (a.work / "crosschecks" / "odom").is_dir() else a.work / "odom"
    if not (fdir / "fusion_info.json").is_file():
        raise SystemExit("no fusion_info.json under %s" % a.work)
    times = np.loadtxt(a.seq / "times.txt", dtype=np.int64)
    n_img = len(times)
    dur = (times[-1] - times[0]) / 1e9
    t_gt, T_gt = load_gt_csv(a.seq / "gt.csv")
    ref_info = json.loads((fdir / "fusion_info.json").read_text())
    tf = json.loads((a.seq / "tf_static.json").read_text())
    cam_frame = json.loads((a.seq / "calib.json").read_text())["left"]["frame"]
    T_base_cam = chain(tf, ref_info["frame"], cam_frame)

    # ---- per run
    runs = {k: [] for k, _ in ARMS}
    base = a.results / "ext" / "ext_s" / name
    for arm, _ in ARMS:
        for rd in sorted((base / arm).glob("run*")):
            meta = json.loads((rd / "meta.json").read_text()) if (rd / "meta.json").is_file() else {}
            con = (rd / "console.log").read_text(errors="replace") if (rd / "console.log").is_file() else ""
            r = dict(run=rd.name, exit=meta.get("exit_code"), wall_s=meta.get("wall_s"), crash=crash_class(con),
                     loops=con.count("*Loop detected"), merges=con.count("*Merge detected"),
                     maps=(int(re.findall(r"There are (\d+) maps", con)[-1]) if re.findall(r"There are (\d+) maps", con) else None),
                     lost=con.count("Fail to track local map"), perf=tegrastats(rd / "tegrastats.log"))
            for kind, fn in (("frames", "f_run.txt"), ("kf", "kf_run.txt")):
                if (rd / fn).is_file() and (rd / fn).stat().st_size > 0:
                    te, Te = load_tum(rd / fn)
                    r[kind] = score(te, Te, t_gt, T_gt)
                    r[kind + "_total"] = len(te)
                    if kind == "frames":
                        r["coverage"] = 100.0 * len(te) / n_img
                        r["init_s"] = float(te[0] - times[0] / 1e9)
                        r["_traj"] = (te, Te)
            runs[arm].append(r)

    # ---- timing runs (instrumented build)
    timing = {}
    if a.timing:
        for arm, _ in ARMS:
            vals = {}
            for rd in sorted((a.timing / "ext" / "ext_s" / name / arm).glob("run*")):
                ev = rd / "eval.json"
                if ev.is_file():
                    for k, v in (json.loads(ev.read_text()).get("timing") or {}).items():
                        vals.setdefault(k, []).append(v["mean_ms"])
            timing[arm] = {k: med(v) for k, v in vals.items()}

    # ---- baselines: robot odometry vs reference (same metric)
    base_rows = []
    for label, fn in (("/platform/odom (wheels only)", "platform_odom.txt"),
                      ("/platform/odom/filtered (robot EKF: wheels + IMU)", "platform_odom_filtered.txt")):
        p = odir / fn
        if p.is_file():
            to, To = odom_in_camera(p, T_base_cam, times)
            s = score(to, To, t_gt, T_gt)
            if s:
                base_rows.append((label, s))

    # ---- markdown
    L = []
    w = L.append
    w("# Jetson ORB-SLAM3 vs fused reference: Husky dataset `%s`" % name)
    w("")
    w("Our camera-only pipeline (stereo IR, `Examples/Stereo/stereo_euroc`, the R6 ORB settings) on the other team's"
      " Husky recording, scored against a reference built from **every non-camera sensor** in the bag.")
    w("")
    w("## 1. Data and reference")
    w("")
    w("| | |")
    w("|---|---|")
    w("| Sequence | %d stereo pairs, %.1f s, %.1f FPS, 640×480 IR, projector off |" % (n_img, dur, (n_img - 1) / dur))
    w("| Reference | %s |" % ref_info["method"])
    w("| Reference path | %.1f m, end-to-start %.2f m, z range %.2f m |" % (
        ref_info["path_length_m"], ref_info["end_to_start_m"], ref_info["z_range_m"]))
    w("| Factors | IMU %s, wheel %s, roll/pitch %s, loop closures %s |" % (
        ref_info["imu_factors"], ref_info["wheel_factors"], ref_info["attitude_factors"], ref_info["loop_factors"]))
    w("| Magnetometer | %s |" % ref_info["magnetometer"])
    w("| Validation only | %s |" % ref_info["validation_only"])
    w("| Camera topics used by the reference | %s |" % ref_info["camera_topics_read"])
    w("| Reference in camera frame | `gt.csv`: %d poses (T_base_cam from /tf_static) |" % len(t_gt))
    rc = fdir / "reference_check.md"
    unc, parts = None, []
    if rc.is_file():
        for l in rc.read_text().splitlines():
            if l.startswith("| fusion ablation: no loops |"):
                parts.append(("RMS change without loop closure", float(l.split("|")[3]) * 100))
    lc = fdir / "loops.csv"
    if lc.is_file():
        acc = [l.split(",") for l in lc.read_text().splitlines()[1:] if l.strip().endswith(",1")]
        if acc:
            parts.append(("half the loop correction on the pass-1 trajectory", max(float(x[7]) for x in acc) * 50))
    if parts:
        unc = max(v for _, v in parts)
        w("| Reference uncertainty (estimate) | **~%s cm**: max of %s. Differences below this are not resolvable;"
          " details in `%s` |" % (f(unc, 0), "; ".join("%s %.1f cm" % (n, v) for n, v in parts), rc))
    w("")

    w("## 2. Accuracy (median over runs, min–max in brackets)")
    w("")
    hdr = ["Metric"] + [lab for _, lab in ARMS]
    rows = []

    def row(label, getter, nd=2):
        rows.append([label] + [fr([getter(r) for r in runs[k]], nd) for k, _ in ARMS])
    g = lambda kind, key: (lambda r: (r.get(kind) or {}).get(key))
    row("**ATE SE(3), keyframes [cm]** (paper Table 1 metric)", g("kf", "ate_se3_cm"))
    row("ATE SE(3), all frames [cm]", g("frames", "ate_se3_cm"))
    row("ATE horizontal part (after SE(3) alignment), all frames [cm]", g("frames", "ate_xy_cm"))
    row("ATE vertical part (after SE(3) alignment), all frames [cm]", g("frames", "z_rms_cm"))
    row("ATE Sim(3), all frames [cm]", g("frames", "ate_sim3_cm"))
    row("Scale (Sim(3)), all frames", g("frames", "scale"), 4)
    row("Scale from 10 s distance ratios (reference / ours), no alignment", g("frames", "seg_scale"), 4)
    row("Max position error [cm]", g("frames", "max_cm"), 1)
    for Ld in (1, 5, 10):
        row("RPE translation over %d m [%%]" % Ld, lambda r, Ld=Ld: (r.get("frames") or {}).get("rpe", {}).get(Ld, (None,))[0])
        row("RPE rotation over %d m [deg/100 m]" % Ld,
            lambda r, Ld=Ld: (r.get("frames") or {}).get("rpe", {}).get(Ld, (None, None))[1])
    row("Coverage (images with a pose) [%]", lambda r: r.get("coverage"), 1)
    row("Time to first pose [s]", lambda r: r.get("init_s"), 2)
    row("Loops closed", lambda r: r.get("loops"), 0)
    row("Maps at the end", lambda r: r.get("maps"), 0)
    row("'Fail to track local map' events", lambda r: r.get("lost"), 0)
    w("| " + " | ".join(hdr) + " |")
    w("|" + "---|" * len(hdr))
    for rr in rows:
        w("| " + " | ".join(rr) + " |")
    w("")
    w("Runs per arm: %s." % ", ".join("%s %d (%d with a trajectory)" % (k, len(runs[k]), sum(1 for r in runs[k] if r.get("frames")))
                                      for k, _ in ARMS))
    w("")

    if base_rows:
        w("### Context: the robot's own odometry against the same reference")
        w("")
        w("| Trajectory | ATE SE(3) [cm] | Max [cm] | RPE 10 m [%] |")
        w("|---|---|---|---|")
        for label, s in base_rows:
            w("| %s | %s | %s | %s |" % (label, f(s["ate_se3_cm"]), f(s["max_cm"], 1), f(s["rpe"][10][0])))
        w("| **Our ORB-SLAM3, GPU (median)** | %s | | |" % f(med([g("frames", "ate_se3_cm")(r) for r in runs["gpu"]])))
        w("")

    # equivalence
    w("## 3. GPU ≡ CPU (and pipelining) against real reference poses")
    w("")
    kf = {k: [g("kf", "ate_se3_cm")(r) for r in runs[k]] for k, _ in ARMS}
    fa = {k: [g("frames", "ate_se3_cm")(r) for r in runs[k]] for k, _ in ARMS}
    spread = lambda v: (rng(v)[1] - rng(v)[0]) if rng(v)[0] is not None else None
    w("| Comparison | Δ median ATE, frames [cm] | Spread A [cm] | Spread B [cm] | Verdict |")
    w("|---|---|---|---|---|")
    for A, B_ in (("gpu", "cpu"), ("gpu", "gpu_pipe"), ("cpu", "gpu_pipe")):
        ma, mb = med(fa[A]), med(fa[B_])
        if ma is None or mb is None:
            continue
        d, sa, sb = abs(ma - mb), spread(fa[A]), spread(fa[B_])
        if d <= max(sa or 0, sb or 0):
            verdict = "equivalent (within run-to-run spread)"
        elif unc is not None and d < unc:
            verdict = "not resolvable (below the reference uncertainty)"
        else:
            verdict = "**different**"
        w("| %s vs %s | %s | %s | %s | %s |" % (A, B_, f(d), f(sa), f(sb), verdict))
    w("")
    w("A difference counts only if it is larger than the run-to-run spread of both arms and than the reference"
      " uncertainty (§1).")
    w("")
    # reference-free equivalence (as R6): distance between trajectories of different arms vs run-to-run
    trajs = {k: [r["_traj"] for r in runs[k] if r.get("_traj")] for k, _ in ARMS}

    def pair_rms(Xs, Ys, same):
        out = []
        for i, (ta, Ta) in enumerate(Xs):
            for j, (tb, Tb) in enumerate(Ys):
                if same and j <= i:
                    continue
                ia, ib = associate(ta, tb, 0.001)
                if len(ia) >= 10:
                    out.append(ate_rmse(Ta[ia, :3, 3], Tb[ib, :3, 3], False)[0] * 100)
        return med(out), len(out)
    w("Reference-free check (no ground truth involved, as in R6): median RMS distance between SE(3)-aligned"
      " trajectories [cm]. Cross-arm distances should not exceed the run-to-run distances.")
    w("")
    w("| Pair | Distance [cm] | Pairs |")
    w("|---|---|---|")
    for A, B_ in (("gpu", "cpu"), ("gpu", "gpu"), ("cpu", "cpu"), ("gpu", "gpu_pipe"), ("gpu_pipe", "gpu_pipe")):
        m_, n_ = pair_rms(trajs[A], trajs[B_], A == B_)
        w("| %s vs %s | %s | %d |" % (A, B_, f(m_), n_))
    w("")

    # performance
    w("## 4. Compute (tegrastats over the tracking window, median over runs)")
    w("")
    w("Replay is paced to the camera (15 FPS): an arm that keeps up spends the same wall time, so CPU/GPU load"
      " and energy per frame are the efficiency comparison. True per-frame cost comes from the instrumented runs.")
    w("")
    P = lambda key: (lambda r: (r.get("perf") or {}).get(key))
    prow = []
    for label, key, nd in (("VDD_IN mean [W]", "vin_w", 2), ("VDD_IN peak [W]", "vin_peak_w", 2),
                           ("VDD_CPU_GPU_CV [W]", "vcg_w", 2), ("VDD_SOC [W]", "vsoc_w", 2),
                           ("CPU load, sum of 6 cores [%]", "cpu_pct", 0), ("GPU load GR3D [%]", "gr3d_pct", 1),
                           ("RAM peak [MB]", "ram_peak_mb", 0), ("Tj max [°C]", "tj_max", 1),
                           ("Busy window incl. vocabulary load + shutdown [s]", "window_s", 0)):
        prow.append([label] + [f(med([P(key)(r) for r in runs[k]]), nd) for k, _ in ARMS])
    prow.append(["Energy per frame [J]"] + [f(med([P("energy_j")(r) / n_img if P("energy_j")(r) else None
                                                   for r in runs[k]]), 3) for k, _ in ARMS])
    budget = 1000.0 * dur / max(n_img - 1, 1)                 # ms per frame at the camera rate

    def realtime(k):
        tt = (timing.get(k) or {}).get("Total Tracking") if timing else None
        if tt:
            return ("yes (%.1f ms < %.1f ms)" % (tt, budget)) if tt < budget else ("**no** (%.1f ms > %.1f ms)" % (tt, budget))
        return "n/a (needs the timing runs)"
    prow.append(["Real-time: mean tracking time vs the camera's %.1f ms frame budget" % budget] +
                [realtime(k) for k, _ in ARMS])
    w("| Metric | " + " | ".join(lab for _, lab in ARMS) + " |")
    w("|" + "---|" * (len(ARMS) + 1))
    for rr in prow:
        w("| " + " | ".join(rr) + " |")
    w("")
    if timing:
        w("### Per-stage timing (REGISTER_TIMES build) [ms/frame]")
        w("")
        stages = ["ORB Extraction", "Stereo Matching", "Pose Prediction", "LM Track", "New KF decision", "Total Tracking",
                  "Total Local Mapping"]
        w("| Stage | " + " | ".join(lab for _, lab in ARMS) + " |")
        w("|" + "---|" * (len(ARMS) + 1))
        for sname in stages:
            w("| %s | %s |" % (sname, " | ".join(f((timing.get(k) or {}).get(sname), 1) for k, _ in ARMS)))
        w("| Throughput 1000/Total Tracking [FPS] | %s |" % " | ".join(
            f(1000 / timing[k]["Total Tracking"] if (timing.get(k) or {}).get("Total Tracking") else None, 1) for k, _ in ARMS))
        w("| Budget at 15 FPS | 66.7 ms |" + " |" * (len(ARMS) - 1))
        w("")

    w("## 5. Run health")
    w("")
    w("| Arm | Run | Exit | Outcome | Coverage [%] | ATE frames [cm] |")
    w("|---|---|---|---|---|---|")
    for k, _ in ARMS:
        for r in runs[k]:
            w("| %s | %s | %s | %s | %s | %s |" % (k, r["run"], r["exit"], r["crash"], f(r.get("coverage"), 1),
                                                   f(g("frames", "ate_se3_cm")(r))))
    w("")

    # ---- plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(T_gt[:, 0, 3], T_gt[:, 1, 3], "k", lw=2, label="reference (fused, camera frame)")
        for k, lab in ARMS:
            ok = [r for r in runs[k] if r.get("frames")]
            if ok:
                rm = sorted(ok, key=lambda r: r["frames"]["ate_se3_cm"])[len(ok) // 2]
                ax.plot(rm["frames"]["aligned"][:, 0], rm["frames"]["aligned"][:, 1], lw=0.9,
                        label="%s, median run (%.1f cm)" % (lab, rm["frames"]["ate_se3_cm"]))
        for label, s in base_rows:
            ax.plot(s["aligned"][:, 0], s["aligned"][:, 1], "--", lw=0.7, label="%s (%.0f cm)" % (label, s["ate_se3_cm"]))
        ax.set_aspect("equal")
        ax.legend(fontsize=7)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title("Top view, SE(3)-aligned to the reference")
        fig.savefig(a.out / "topview.png", dpi=150, bbox_inches="tight")
        fig, ax = plt.subplots(figsize=(10, 4))
        for k, lab in ARMS:
            for r in runs[k]:
                if r.get("frames"):
                    ax.plot(r["frames"]["t"] - t_gt[0], r["frames"]["err"] * 100, lw=0.6,
                            color={"gpu": "C0", "cpu": "C1", "gpu_pipe": "C2"}[k],
                            label=lab if r is next(x for x in runs[k] if x.get("frames")) else None)
        if unc:
            ax.axhline(unc, color="k", ls=":", lw=0.8, label="reference uncertainty")
        ax.set_xlabel("t [s]")
        ax.set_ylabel("position error [cm]")
        ax.legend(fontsize=7)
        fig.savefig(a.out / "error_vs_time.png", dpi=130, bbox_inches="tight")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.boxplot([[x for x in fa[k] if x is not None] or [np.nan] for k, _ in ARMS], tick_labels=[k for k, _ in ARMS])
        ax.set_ylabel("ATE SE(3), all frames [cm]")
        fig.savefig(a.out / "ate_boxplot.png", dpi=130, bbox_inches="tight")
        w("## 6. Plots")
        w("")
        w("![top view](topview.png)")
        w("")
        w("![error over time](error_vs_time.png)")
        w("")
        w("![ATE per arm](ate_boxplot.png)")
        w("")
        w("Reference construction and its cross-checks: `%s` (fusion_topview.png, reference_check.png)." % (fdir))
    except Exception as e:                                          # noqa: BLE001
        w("(plots skipped: %r)" % e)

    (a.out / "HUSKY_REPORT.md").write_text("\n".join(L) + "\n")
    slim = {k: [{kk: vv for kk, vv in r.items() if not kk.startswith("_") and kk not in ("frames", "kf")}
                | {kind: {kk: vv for kk, vv in (r.get(kind) or {}).items() if kk not in ("err", "t", "aligned", "rpe")}
                   for kind in ("frames", "kf")} for r in runs[k]] for k, _ in ARMS}
    (a.out / "runs.json").write_text(json.dumps(slim, indent=1, default=float))
    print("wrote", a.out / "HUSKY_REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
