#!/usr/bin/env python3
"""Ablation report: every cell of plan/ablation_plan_husky.md against the fused reference.

usage: ablation_report.py --stage1 DIR --abl DIR --data DIR --reference DIR --out DIR

  --stage1     stage-1 results root (run_matrix.py ext): cells B0, B1, B2, C2 baseline arms
  --abl        ablation results root (run_matrix.py abl): all other cells
  --data       ~/datasets/ext (the base sequence and its __variant folders, each with its own gt.csv)
  --reference  eval/results/husky/reference (for the non-visual baselines E1-E3)
  --out        report folder: ABLATION_REPORT.md, ablation_cells.json, PNG plots

Every run is scored against its own variant's gt.csv (camera, IMU-body or colour-camera frame, at
that variant's image times), with the same metrics as husky_report.py.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "benchmark"))
from evaluate import load_gt_csv, load_tum  # noqa: E402
from husky_report import crash_class, f, fr, med, odom_in_camera, rng, score, tegrastats  # noqa: E402
from make_gt import chain  # noqa: E402

SEQ = "husky_20261023"
# (id, group, configuration, sequence suffix, mode folder, arm, results root key)
CELLS = [
    ("B0", "baseline", "stereo IR, GPU ORB, DBoW2 (stage 1)", "", "ext_s", "gpu", "stage1"),
    ("A1", "A sensors", "mono IR", "__mono", "ext_m", "gpu", "abl"),
    ("A3", "A sensors", "mono IR + IMU", "__mi", "ext_mi", "gpu", "abl"),
    ("A4", "A sensors", "stereo IR + IMU", "__si", "ext_si", "gpu", "abl"),
    ("A5", "A sensors", "RGB-D mode: IR + depth", "__ird", "ext_rgbd", "gpu", "abl"),
    ("A6", "A sensors", "RGB-D mode: colour + depth", "__rgbd", "ext_rgbd", "gpu", "abl"),
    ("B1", "B compute", "stereo, CPU ORB (stage 1)", "", "ext_s", "cpu", "stage1"),
    ("B2", "B compute", "stereo, PIPELINE_FE (stage 1)", "", "ext_s", "gpu_pipe", "stage1"),
    ("B1i", "B compute", "stereo + IMU, CPU ORB", "__si", "ext_si", "cpu", "abl"),
    ("B2i", "B compute", "stereo + IMU, PIPELINE_FE", "__si", "ext_si", "gpu_pipe", "abl"),
    ("B3", "B compute", "stereo + IMU, ADAPTIVE_FE", "__si", "ext_si", "adaptive_fe", "abl"),
    ("C1", "C loop closure", "stereo, no loop closing", "__nolc", "ext_s", "gpu", "abl"),
    ("C1i", "C loop closure", "stereo + IMU, no loop closing", "__si_nolc", "ext_si", "gpu", "abl"),
    ("C2", "C loop closure", "stereo, DBoW2 + CNN", "", "ext_s", "cnn", "abl"),
    ("C2i", "C loop closure", "stereo + IMU, DBoW2 + CNN", "__si", "ext_si", "cnn", "abl"),
    ("C3", "C loop closure", "stereo + IMU, ADAPTIVE_LC", "__si", "ext_si", "adaptive_lc", "abl"),
    ("D1", "D stress", "stereo, 7.5 FPS", "__7p5", "ext_s", "gpu", "abl"),
    ("D1i", "D stress", "stereo + IMU, 7.5 FPS", "__si_7p5", "ext_si", "gpu", "abl"),
    ("D2", "D stress", "stereo, 800 features", "__f800", "ext_s", "gpu", "abl"),
    ("D2i", "D stress", "stereo + IMU, 800 features", "__si_f800", "ext_si", "gpu", "abl"),
    ("D3", "D stress", "stereo, baseline x1.068 (calibration test)", "__bscale", "ext_s", "gpu", "abl"),
]


def score_cell(root: Path, mode: str, seq_dir: Path, arm: str):
    times = np.loadtxt(seq_dir / "times.txt", dtype=np.int64)
    t_gt, T_gt = load_gt_csv(seq_dir / "gt.csv")
    runs = []
    for rd in sorted((root / mode / seq_dir.name / arm).glob("run*")):
        con = (rd / "console.log").read_text(errors="replace") if (rd / "console.log").is_file() else ""
        maps = re.findall(r"There are (\d+) maps", con)
        r = dict(run=rd.name, crash=crash_class(con), loops=con.count("*Loop detected"),
                 maps=int(maps[-1]) if maps else None, lost=con.count("Fail to track local map"),
                 imu_init="end VIBA 1" in con,            # src/LocalMapping.cc: first visual-inertial BA done
                 imu_resets=con.count("Not enough motion for initializing"),
                 perf=tegrastats(rd / "tegrastats.log"))
        f_ = next((rd / n for n in ("f_run.txt", "CameraTrajectory.txt") if (rd / n).is_file()
                   and (rd / n).stat().st_size > 0), None)
        if f_:
            te, Te = load_tum(f_)
            r["frames"] = score(te, Te, t_gt, T_gt)
            r["coverage"] = 100.0 * len(te) / len(times)
            r["init_s"] = float(te[0] - times[0] / 1e9)
            r["n_frames"] = len(times)
        runs.append(r)
    return runs


def exec_mean(path: Path) -> dict:
    """ExecMean.txt (REGISTER_TIMES): 'Label: mean$\\pm$std' and plain 'Label: value' counters."""
    out = {}
    if not path.is_file():
        return out
    for line in path.read_text(errors="replace").splitlines():
        m = re.match(r"\s*([A-Za-z .]+):\s*([0-9.eE+-]+|nan)(?:\$\\pm\$)?", line)
        if m and m.group(2) != "nan" and not m.group(1).strip().startswith("OpenCV"):
            try:
                out[m.group(1).strip()] = float(m.group(2))
            except ValueError:
                pass
    return out


def summarize(runs, use_sim3=False):
    ok = [r for r in runs if r.get("frames")]
    g = lambda k: [(r.get("frames") or {}).get(k) for r in runs]
    epf = [r["perf"]["energy_j"] / r["n_frames"] if r.get("perf") and r.get("n_frames") else None for r in ok]
    key = "ate_sim3_cm" if use_sim3 else "ate_se3_cm"
    return dict(n=len(runs), ok=len(ok), ate=g(key), ate_se3=g("ate_se3_cm"), ate_sim3=g("ate_sim3_cm"),
                scale=g("scale"), xy=g("ate_xy_cm"),
                rpe10=[(r.get("frames") or {}).get("rpe", {}).get(10, (None,))[0] for r in runs],
                coverage=[r.get("coverage") for r in runs], init=[r.get("init_s") for r in runs],
                loops=[r["loops"] for r in runs], maps=[r["maps"] for r in runs],
                crashes=sum(1 for r in runs if r["crash"] == "Sophus NaN (mid-run)"),
                cpu=[(r.get("perf") or {}).get("cpu_pct") for r in ok], gpu=[(r.get("perf") or {}).get("gr3d_pct") for r in ok],
                vin=[(r.get("perf") or {}).get("vin_w") for r in ok], epf=epf,
                imu_init=sum(1 for r in runs if r.get("imu_init")),
                imu_resets=sum(r.get("imu_resets", 0) for r in runs))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", type=Path, required=True)
    ap.add_argument("--abl", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path.home() / "datasets/ext")
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stage1-timing", type=Path, help="stage-1 timing results root (REGISTER_TIMES build)")
    ap.add_argument("--abl-timing", type=Path, help="ablation timing results root (REGISTER_TIMES build)")
    ap.add_argument("--idle", type=Path, help="tegrastats log of the idle board (same power mode)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    roots = {"stage1": a.stage1 / "ext", "abl": a.abl / "abl"}

    cells = []
    for cid, grp, label, suf, mode, arm, rk in CELLS:
        seq_dir = a.data / (SEQ + suf)
        if not (seq_dir / "gt.csv").is_file():
            cells.append(dict(id=cid, group=grp, label=label, missing="no sequence %s" % seq_dir.name))
            continue
        runs = score_cell(roots[rk], mode, seq_dir, arm)
        sm = summarize(runs, use_sim3=(mode in ("ext_m",)))
        cells.append(dict(id=cid, group=grp, label=label, mode=mode, arm=arm, seq=seq_dir.name,
                          sim3_primary=(mode == "ext_m"), **sm))

    troots = {"stage1": a.stage1_timing / "ext" if a.stage1_timing else None,
              "abl": a.abl_timing / "abl" if a.abl_timing else None}
    rk_of = {c[0]: c[6] for c in CELLS}
    for c in cells:
        if c.get("missing"):
            continue
        tr = troots[rk_of[c["id"]]]
        c["timing"] = exec_mean(tr / c["mode"] / c["seq"] / c["arm"] / "run1" / "ExecMean.txt") if tr else {}
    idle_w = None
    if a.idle and a.idle.is_file():
        v = [int(x) for x in re.findall(r"VDD_IN (\d+)mW", a.idle.read_text())]
        idle_w = float(np.median(v)) / 1000 if v else None

    # unc of the reference (as husky_report.py)
    unc = None
    rc = a.reference / "fusion" / "reference_check.md"
    if rc.is_file():
        for l in rc.read_text().splitlines():
            if l.startswith("| fusion ablation: no loops |"):
                unc = float(l.split("|")[3]) * 100
    lc = a.reference / "fusion" / "loops.csv"
    if lc.is_file():
        acc = [l.split(",") for l in lc.read_text().splitlines()[1:] if l.strip().endswith(",1")]
        if acc:
            unc = max(unc or 0, max(float(x[7]) for x in acc) * 50)

    base = next(c for c in cells if c["id"] == "B0")
    bmed, bsp = med(base["ate"]), (rng(base["ate"])[1] - rng(base["ate"])[0]) if base["ok"] else None

    def verdict(c):
        if c.get("missing") or not c["ok"] or c["id"] == "B0":
            return ""
        m = med(c["ate"])
        lo, hi = rng(c["ate"])
        d = m - bmed
        spread = max(bsp or 0, (hi - lo) if lo is not None else 0)
        if c.get("sim3_primary"):
            d = m - med(base["ate_sim3"])
        if abs(d) <= spread:
            return "%+.1f cm: within run-to-run" % d
        if unc is not None and abs(d) < unc:
            return "%+.1f cm: below reference uncertainty" % d
        return "**%+.1f cm: %s**" % (d, "better" if d < 0 else "worse")

    # non-visual baselines
    times = np.loadtxt(a.data / SEQ / "times.txt", dtype=np.int64)
    t_gt, T_gt = load_gt_csv(a.data / SEQ / "gt.csv")
    tf = json.loads((a.data / SEQ / "tf_static.json").read_text())
    info = json.loads((a.reference / "fusion" / "fusion_info.json").read_text())
    T_BC = chain(tf, info["frame"], json.loads((a.data / SEQ / "calib.json").read_text())["left"]["frame"])
    T_LB = np.linalg.inv(np.array(info["T_base_lidar_used"]))
    base_rows = []
    for cid, label, p, Tpost in (("E1", "wheels (/platform/odom)", a.reference / "crosschecks/odom/platform_odom.txt", T_BC),
                                 ("E2", "robot EKF (/platform/odom/filtered)", a.reference / "crosschecks/odom/platform_odom_filtered.txt", T_BC),
                                 ("E3", "lidar only (KISS-ICP, no loops)", a.reference / "crosschecks/kiss_icp/lidar_poses.txt", T_LB @ T_BC)):
        if p.is_file():
            to, To = odom_in_camera(p, Tpost, times)
            s = score(to, To, t_gt, T_gt)
            if s:
                base_rows.append((cid, label, s))

    # ---- markdown
    L = []
    w = L.append
    w("# Ablation study: Husky `%s`, Jetson ORB-SLAM3 vs the fused reference" % SEQ)
    w("")
    w("One factor changed per cell from **B0** (stereo IR, GPU ORB, DBoW2 loop closure, 15 FPS, 1250 features)."
      " Medians over runs, min–max in brackets. Every run is scored against the reference in its own output frame."
      " Reference uncertainty ≈ **%s cm**: smaller differences are not resolvable." % f(unc, 0))
    w("")
    w("**Note on scale.** Stage 1 found ORB-SLAM3 stereo distances ~6.4%% short of the lidar on this camera"
      " (Sim(3) scale ≈ 1.07). That error dominates SE(3) ATE, so SE(3) and Sim(3) ATE are both shown: SE(3)"
      " includes the scale error, Sim(3) shows the shape. Cell D3 tests the calibration hypothesis directly.")
    w("")
    w("## 1. All cells")
    w("")
    w("| Cell | Configuration | Runs ok | ATE SE(3) [cm] | ATE Sim(3) [cm] | Scale | RPE 10 m [%] | Coverage [%] | Loops | Maps | NaN aborts | Δ vs B0 |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c in cells:
        if c.get("missing"):
            w("| %s | %s | – | %s | | | | | | | | |" % (c["id"], c["label"], c["missing"]))
            continue
        w("| %s | %s | %d/%d | %s | %s | %s | %s | %s | %s | %s | %d | %s |" % (
            c["id"], c["label"] + (" *(Sim(3) primary: mono has no scale)*" if c.get("sim3_primary") else ""),
            c["ok"], c["n"], fr(c["ate_se3"], 1), fr(c["ate_sim3"], 1), f(med(c["scale"]), 3), f(med(c["rpe10"]), 2),
            f(med(c["coverage"]), 1), f(med(c["loops"]), 0), f(med(c["maps"]), 0), c["crashes"], verdict(c)))
    w("")
    w("Inertial cells: runs that completed IMU initialisation (`end VIBA 1`), and 'Not enough motion for"
      " initializing' resets (the Husky turns at most ~7°/s): %s." % ", ".join(
        "%s %d/%d (%d resets)" % (c["id"], c["imu_init"], c["n"], c["imu_resets"])
        for c in cells if not c.get("missing") and c["mode"] in ("ext_si", "ext_mi")))
    w("")
    w("## 2. Compute (tegrastats, median over runs; replay paced to 15 FPS)")
    w("")
    w("| Cell | Configuration | CPU [%] | GPU [%] | VDD_IN [W] | Energy per frame [J] |")
    w("|---|---|---|---|---|---|")
    for c in cells:
        if not c.get("missing"):
            w("| %s | %s | %s | %s | %s | %s |" % (c["id"], c["label"], f(med(c["cpu"]), 0), f(med(c["gpu"]), 1),
                                                   f(med(c["vin"]), 2), f(med(c["epf"]), 3)))
    w("")
    if idle_w:
        w("Idle board (MAXN_SUPER + jetson_clocks): **%.2f W**. SLAM's own cost per cell = VDD_IN − idle:" % idle_w)
        w("")
        w("| Cell | Configuration | SLAM power above idle [W] | vs B0 |")
        w("|---|---|---|---|")
        b0 = med(next(c for c in cells if c["id"] == "B0")["vin"]) - idle_w
        for c in cells:
            if not c.get("missing") and med(c["vin"]) is not None:
                d = med(c["vin"]) - idle_w
                w("| %s | %s | %.2f | %+.0f%% |" % (c["id"], c["label"], d, 100 * (d / b0 - 1)))
        w("")
    if any(c.get("timing") for c in cells):
        w("### Per-frame cost (REGISTER_TIMES build, 1 run per cell) [ms]")
        w("")
        w("True processing cost, independent of the 15 FPS pacing. The budget is 66.7 ms per frame at 15 FPS"
          " (133 ms at 7.5 FPS). Local mapping is per keyframe, in its own thread.")
        w("")
        cols = ["ORB Extraction", "Stereo Matching", "IMU Preintegration", "LM Track", "Total Tracking",
                "Total Local Mapping", "KFs in map", "LBA executions"]
        w("| Cell | Configuration | ORB extr. | Stereo match | IMU preint. | LM track | **Total tracking** | **FPS** | Local mapping / KF | KFs | LBA runs |")
        w("|---|---|---|---|---|---|---|---|---|---|---|")
        for c in cells:
            t = c.get("timing") or {}
            if c.get("missing"):
                continue
            tt = t.get("Total Tracking")
            w("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                c["id"], c["label"], f(t.get("ORB Extraction"), 1), f(t.get("Stereo Matching"), 1),
                f(t.get("IMU Preintegration"), 2), f(t.get("LM Track"), 1), f(tt, 1),
                f(1000 / tt if tt else None, 1), f(t.get("Total Local Mapping"), 0),
                f(t.get("KFs in map"), 0), f(t.get("LBA executions"), 0)))
        w("")
    w("## 3. Non-visual baselines (same reference, same metric)")
    w("")
    w("| Cell | Trajectory | ATE SE(3) [cm] | ATE Sim(3) [cm] | Scale | RPE 10 m [%] |")
    w("|---|---|---|---|---|---|")
    for cid, label, s in base_rows:
        w("| %s | %s | %s | %s | %s | %s |" % (cid, label, f(s["ate_se3_cm"], 1), f(s["ate_sim3_cm"], 1), f(s["scale"], 3),
                                             f(s["rpe"][10][0], 2)))
    w("")
    w("E3 shares the lidar with the reference, so it is not an independent comparison.")
    w("")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ok = [c for c in cells if not c.get("missing") and c["ok"]]
        fig, ax = plt.subplots(figsize=(11, 4.5))
        x = np.arange(len(ok))
        ax.bar(x - 0.2, [med(c["ate_se3"]) for c in ok], 0.4, label="ATE SE(3)")
        ax.bar(x + 0.2, [med(c["ate_sim3"]) for c in ok], 0.4, label="ATE Sim(3)")
        if unc:
            ax.axhline(unc, color="k", ls=":", lw=0.8, label="reference uncertainty")
        ax.set_xticks(x)
        ax.set_xticklabels([c["id"] for c in ok])
        ax.set_ylabel("cm")
        ax.legend()
        fig.savefig(a.out / "ablation_ate.png", dpi=130, bbox_inches="tight")
        fig, ax = plt.subplots(figsize=(7, 5))
        for c in ok:
            e, m_ = med(c["epf"]), med(c["ate_sim3"])
            if e is not None and m_ is not None:
                ax.scatter(e, m_)
                ax.annotate(c["id"], (e, m_), fontsize=8, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("energy per frame [J]")
        ax.set_ylabel("ATE Sim(3) [cm]")
        ax.set_title("Accuracy vs energy (lower-left is better)")
        fig.savefig(a.out / "ablation_pareto.png", dpi=130, bbox_inches="tight")
        w("## 4. Plots")
        w("")
        w("![ATE per cell](ablation_ate.png)")
        w("")
        w("![Accuracy vs energy](ablation_pareto.png)")
    except Exception as e:                                       # noqa: BLE001
        w("(plots skipped: %r)" % e)
    (a.out / "ABLATION_REPORT.md").write_text("\n".join(L) + "\n")
    (a.out / "ablation_cells.json").write_text(json.dumps(cells, indent=1, default=float))
    print("wrote", a.out / "ABLATION_REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
