#!/usr/bin/env python3
"""Build the reproduction report: our numbers next to the paper's (plan/paper_runbook.md, R5).

usage: make_report.py [RESULTS_DIR] [-o REPORT.md]

Reads every run's meta.json + eval.json (run evaluate.py first), the feature-equivalence result
(RESULTS_DIR/features/features.json) and any TensorRT latency logs (RESULTS_DIR/cnn_latency/*.log),
and writes one Markdown report. Aggregation follows the paper: per-sequence median over runs,
then the mean over sequences; |delta| is the absolute GPU-vs-CPU gap per sequence, averaged.
"""
from __future__ import annotations

import argparse
import datetime
import itertools
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_reference as P  # noqa: E402
from common import EUROC, KITTI, RESULTS_ROOT, TUMVI, read_json  # noqa: E402
from evaluate import associate, ate_rmse, load_tum  # noqa: E402

PRIMARY = "kf_ate_se3_cm"            # repository's own scorer (run_euroc.sh) uses the keyframe trajectory


def load_runs(root: Path):
    """{(suite, mode, seq, arm): [ (meta, eval, run_dir), ... ]}"""
    runs = defaultdict(list)
    for meta_p in sorted(root.rglob("meta.json")):
        meta = read_json(meta_p)
        if not meta.get("done") or meta.get("kind") == "idle":
            continue
        ev_p = meta_p.parent / "eval.json"
        ev = read_json(ev_p) if ev_p.is_file() else {}
        runs[(meta["suite"], meta["mode"], meta["seq"], meta["arm"])].append((meta, ev, meta_p.parent))
    return runs


def med(vals):
    vals = [v for v in vals if v is not None]
    return statistics.median(vals) if vals else None


def mean(vals):
    vals = [v for v in vals if v is not None]
    return statistics.mean(vals) if vals else None


def f(v, nd=2):
    return "–" if v is None else ("%.*f" % (nd, v))


def acc(run, key):
    return (run[1].get("accuracy") or {}).get(key)


def con(run, key):
    return (run[1].get("console") or {}).get(key)


def cell(runs, key, getter=acc):
    """median over runs, and 'n_ok/n' note."""
    vals = [getter(r, key) for r in runs]
    ok = [v for v in vals if v is not None]
    return med(ok), "%d/%d" % (len(ok), len(vals))


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


# --------------------------------------------------------------------------- sections
def sec_euroc_accuracy(R):
    s = ["## Table 1 – EuRoC stereo-inertial ATE (cm, SE(3)), GPU vs CPU",
         "Per-sequence **median over runs** of the keyframe-trajectory ATE (the trajectory the "
         "repository's own scorer, run_euroc.sh, uses). Paper: median of 5 runs.", ""]
    rows, d_ours, gm, cm, gs, cs = [], [], [], [], [], []
    for seq in EUROC:
        g = R.get(("euroc-accuracy", "euroc_si", seq, "gpu"), [])
        c = R.get(("euroc-accuracy", "euroc_si", seq, "cpu"), [])
        gv, gn = cell(g, PRIMARY)
        cv, cn = cell(c, PRIMARY)
        gsv, _ = cell(g, "kf_ate_sim3_cm")
        csv_, _ = cell(c, "kf_ate_sim3_cm")
        d = abs(gv - cv) if gv is not None and cv is not None else None
        p = P.TABLE1[seq]
        rows.append([seq, f(p[1]), f(p[2]), f(p[3]), f(p[4]), "%s (%s)" % (f(gv), gn), "%s (%s)" % (f(cv), cn), f(d)])
        gm.append(gv); cm.append(cv); gs.append(gsv); cs.append(csv_); d_ours.append(d)
    pm = P.TABLE1_MEAN
    rows.append(["**Mean**", f(pm[1]), f(pm[2]), f(pm[3]), f(pm[4]), f(mean(gm)), f(mean(cm)), f(mean(d_ours))])
    s.append(table(["Seq", "Paper ORB-SLAM3 (desktop)", "Paper GPU", "Paper CPU", "Paper Δ (abs)",
                    "Ours GPU (runs ok)", "Ours CPU (runs ok)", "Ours Δ (abs)"], rows))
    s += ["", "## Table 2 – mean EuRoC ATE over 11 sequences (cm)", ""]
    rows = [["Orin Nano, GPU", f(P.TABLE2["Orin Nano, GPU (ours)"][0]), f(P.TABLE2["Orin Nano, GPU (ours)"][1]), f(mean(gm)), f(mean(gs))],
            ["Orin Nano, CPU", f(P.TABLE2["Orin Nano, CPU (stock)"][0]), f(P.TABLE2["Orin Nano, CPU (stock)"][1]), f(mean(cm)), f(mean(cs))]]
    s.append(table(["Configuration", "Paper SE(3)", "Paper scaled", "Ours SE(3)", "Ours scaled"], rows))
    s += ["", "Secondary metric (every tracked frame instead of keyframes), same runs:", ""]
    rows = []
    for seq in EUROC:
        g = R.get(("euroc-accuracy", "euroc_si", seq, "gpu"), [])
        c = R.get(("euroc-accuracy", "euroc_si", seq, "cpu"), [])
        rows.append([seq, f(cell(g, "frames_ate_se3_cm")[0]), f(cell(c, "frames_ate_se3_cm")[0])])
    s.append(table(["Seq", "GPU frames SE(3)", "CPU frames SE(3)"], rows))
    return "\n".join(s)


def sec_tumvi(R):
    rows, gm, cm, dd = [], [], [], []
    for room in TUMVI:
        g, _ = cell(R.get(("tumvi", "tumvi_si", room, "gpu"), []), PRIMARY)
        c, _ = cell(R.get(("tumvi", "tumvi_si", room, "cpu"), []), PRIMARY)
        d = abs(g - c) if g is not None and c is not None else None
        p = P.TABLE3[room]
        rows.append([room, f(p[0]), f(p[1]), f(p[2]), f(p[3]), f(g), f(c), f(d)])
        gm.append(g); cm.append(c); dd.append(d)
    pm = P.TABLE3_MEAN
    rows.append(["**Mean**", f(pm[0]), f(pm[1]), f(pm[2]), f(pm[3]), f(mean(gm)), f(mean(cm)), f(mean(dd))])
    return "## Table 3 – TUM-VI rooms, stereo-inertial ATE (cm, SE(3))\n\n" + table(
        ["Room", "Paper ORB-SLAM3", "Paper GPU", "Paper CPU", "Paper Δ (abs)", "Ours GPU", "Ours CPU", "Ours Δ (abs)"], rows)


def sec_kitti(R):
    rows, cols = [], defaultdict(list)
    for seq in KITTI:
        g = R.get(("kitti", "kitti_s", seq, "gpu"), [])
        c = R.get(("kitti", "kitti_s", seq, "cpu"), [])
        gt, _ = cell(g, "trel_pct"); ct, _ = cell(c, "trel_pct")
        gr, _ = cell(g, "rrel_deg100m"); ga, _ = cell(g, "ate_se3_m")
        d = abs(gt - ct) if gt is not None and ct is not None else None
        err = next((acc(r, "error") for r in g + c if acc(r, "error")), "")
        p = P.TABLE4[seq]
        rows.append([seq, f(p[1]), f(p[2]), f(gt), f(ct), f(d), f(p[4]), f(gr), f(p[6]), f(ga), err])
        for k, v in (("gt", gt), ("ct", ct), ("d", d), ("gr", gr), ("ga", ga)):
            cols[k].append(v)
    pm = P.TABLE4_MEAN
    rows.append(["**Mean**", f(pm[1]), f(pm[2]), f(mean(cols["gt"])), f(mean(cols["ct"])), f(mean(cols["d"])),
                 f(pm[4]), f(mean(cols["gr"])), f(pm[6]), f(mean(cols["ga"])), ""])
    return "## Table 4 – KITTI stereo: t_rel (%), r_rel (°/100 m), ATE (m)\n\n" + table(
        ["Seq", "Paper t_rel GPU", "Paper t_rel CPU", "Ours t_rel GPU", "Ours t_rel CPU", "Ours Δ (abs)",
         "Paper r_rel", "Ours r_rel", "Paper ATE", "Ours ATE", "note"], rows)


def sec_table5(R):
    def trel_mean(suite, mode, seqs, arm, key):
        return mean([cell(R.get((suite, mode, s, arm), []), key)[0] for s in seqs])

    def gap(suite, mode, seqs, key):
        ds = []
        for s in seqs:
            g = cell(R.get((suite, mode, s, "gpu"), []), key)[0]
            c = cell(R.get((suite, mode, s, "cpu"), []), key)[0]
            if g is not None and c is not None:
                ds.append(abs(g - c))
        return mean(ds)
    rows = []
    for name, suite, mode, seqs, key in (("TUM-VI", "tumvi", "tumvi_si", TUMVI, "trel_5_40_pct"),
                                         ("EuRoC", "euroc-accuracy", "euroc_si", list(EUROC), "trel_5_40_pct"),
                                         ("KITTI", "kitti", "kitti_s", KITTI, "trel_pct")):
        p = P.TABLE5[name]
        rows.append([name, p[0], f(p[1], 3), f(p[2], 3), f(p[3], 3),
                     f(trel_mean(suite, mode, seqs, "gpu", key), 3), f(trel_mean(suite, mode, seqs, "cpu", key), 3),
                     f(gap(suite, mode, seqs, key), 3)])
    return "## Table 5 – cross-dataset t_rel (%) under KITTI's estimator\n\n" + table(
        ["Benchmark", "Windows", "Paper GPU", "Paper CPU", "Paper Δ (abs)", "Ours GPU", "Ours CPU", "Ours Δ (abs)"], rows)


def sec_table6(R):
    def stat(mode, seq, label):
        runs = R.get(("instrumented", mode, seq, "gpu"), [])
        return med([((r[1].get("timing") or {}).get(label) or {}).get("mean_ms") for r in runs])
    mo, so = stat("euroc_mi", "V101", "ORB Extraction"), stat("euroc_si", "MH01", "ORB Extraction")
    mt, st = stat("euroc_mi", "V101", "Total Tracking"), stat("euroc_si", "MH01", "Total Tracking")
    rows = [["ORB extraction (ms)", f(P.TABLE6["ORB extraction (ms)"][0], 1), f(mo, 1), f(P.TABLE6["ORB extraction (ms)"][1], 1), f(so, 1)],
            ["Total tracking (ms/frame)", f(P.TABLE6["Total tracking (ms/frame)"][0], 1), f(mt, 1), f(P.TABLE6["Total tracking (ms/frame)"][1], 1), f(st, 1)],
            ["Throughput (FPS)", f(P.TABLE6["Throughput (FPS)"][0], 1), f(1000 / mt if mt else None, 1),
             f(P.TABLE6["Throughput (FPS)"][1], 1), f(1000 / st if st else None, 1)]]
    return "## Table 6 – per-stage timing, instrumented (REGISTER_TIMES) build\n\n" + table(
        ["Stage", "Paper mono-inertial V101", "Ours", "Paper stereo-inertial MH01", "Ours"], rows)


def fps(runs):
    t = med([con(r, "mean_track_s") for r in runs])
    return 1.0 / t if t else None


def sec_table7(R):
    rows, cols = [], defaultdict(list)
    for seq in EUROC:
        vals = [fps(R.get(("euroc-throughput", m, seq, a), [])) for m, a in
                (("euroc_mi", "gpu"), ("euroc_mi", "gpu_pipe"), ("euroc_si", "gpu"), ("euroc_si", "gpu_pipe"))]
        p = P.TABLE7[seq]
        rows.append([seq] + ["%s / %s" % (f(p[i], 1), f(vals[i], 1)) for i in range(4)])
        for i, v in enumerate(vals):
            cols[i].append(v)
    pm = P.TABLE7_MEAN
    rows.append(["**Mean**"] + ["%s / %s" % (f(pm[i], 1), f(mean(cols[i]), 1)) for i in range(4)])
    return ("## Table 7 – throughput (FPS = 1 / mean tracking time), paper / ours\n\n"
            "EuRoC is recorded at 20 FPS; Pipe = PIPELINE_FE=1.\n\n") + table(
        ["Seq", "Mono-inertial Base", "Mono-inertial Pipe", "Stereo-inertial Base", "Stereo-inertial Pipe"], rows)


def sec_table8(R):
    def mt(suite, mode, seqs, arm):
        return mean([med([con(r, "mean_track_s") for r in R.get((suite, mode, s, arm), [])]) for s in seqs])
    rows = []
    for name, suite, mode, seqs in (("KITTI (stereo)", "kitti", "kitti_s", KITTI),
                                    ("EuRoC (stereo-inertial)", "euroc-accuracy", "euroc_si", list(EUROC)),
                                    ("TUM-VI (stereo-inertial)", "tumvi", "tumvi_si", TUMVI)):
        p = P.TABLE8[name]
        g, c = mt(suite, mode, seqs, "gpu"), mt(suite, mode, seqs, "cpu")
        rows.append([name, p[0], f(p[1], 1), f(p[2], 1), f(g * 1000 if g else None, 1), f(c * 1000 if c else None, 1)])
    return ("## Table 8 – mean tracking time (ms), GPU vs CPU reference\n\n"
            "KITTI and TUM-VI programs always open the viewer (drawn in software under xvfb), on both arms.\n\n") + table(
        ["Benchmark", "Resolution", "Paper GPU", "Paper CPU", "Ours GPU", "Ours CPU"], rows)


def sec_table9(root: Path):
    rows = []
    for log in sorted((root / "cnn_latency").glob("*.log")):
        txt = log.read_text(errors="replace")
        m = re.search(r"GPU Compute Time: .*?mean = ([0-9.]+) ms", txt) or re.search(r"Latency: .*?mean = ([0-9.]+) ms", txt)
        rows.append([log.stem, m.group(1) if m else "see log"])
    s = "## Table 9 – CosPlace ResNet-50 latency per query (ms)\n\n" + table(
        ["Execution path", "Paper"], [[k, v] for k, v in P.TABLE9.items()])
    if rows:
        s += "\n\nOurs (trtexec):\n\n" + table(["Log", "mean ms"], rows)
    else:
        s += "\n\nOurs: not measured yet (runbook Part R3, step 4)."
    return s


def sec_table10(R):
    rows, cols = [], defaultdict(list)
    for seq in list(EUROC):
        vals = [cell(R.get(("lc-ablation", "euroc_si", seq, a), []), PRIMARY)[0] for a in ("cnn", "dbow2", "nolc")]
        p = P.TABLE10.get(seq, P.TABLE10_V203 if seq == "V203" else None)
        rows.append([seq] + ["%s / %s" % (f(p[i]), f(vals[i])) for i in range(3)])
        if seq != "V203":
            for i, v in enumerate(vals):
                cols[i].append(v)
    pm = P.TABLE10_MEAN_NO_V203
    rows.append(["**Mean (no V203)**"] + ["%s / %s" % (f(pm[i]), f(mean(cols[i]))) for i in range(3)])
    return "## Table 10 – loop-closure ablation, median ATE (cm), paper / ours\n\n" + table(
        ["Seq", "DBoW2 + CNN", "DBoW2 only", "no loop closing"], rows)


def sec_power(root: Path, R):
    idle = None
    ip = root / "power" / "idle" / "eval.json"
    if ip.is_file():
        idle = (read_json(ip).get("power") or {}).get("vdd_in_mean_w")
    runs = R.get(("power", "euroc_si", "MH01", "gpu"), [])
    m = med([(r[1].get("power") or {}).get("vdd_in_mean_w") for r in runs])
    pk = med([(r[1].get("power") or {}).get("vdd_in_peak_w") for r in runs])
    return "## Power (Sec. 4.1) – VDD_IN, stereo-inertial MH01, tegrastats @ 250 ms (W)\n\n" + table(
        ["", "Paper", "Ours"], [["Idle", f(P.POWER["idle"], 1), f(idle, 2)],
                                ["Tracking mean", f(P.POWER["mean"], 1), f(m, 2)],
                                ["Tracking peak", f(P.POWER["peak"], 1), f(pk, 2)]])


def sec_features(root: Path):
    p = root / "features" / "features.json"
    pf = P.FEATURES
    if not p.is_file():
        return "## Feature-level equivalence (Sec. 3.1.5)\n\nNot measured yet (runbook Part R4, features)."
    o = read_json(p)
    rows = [["Frames", pf["frames"], o.get("frames")],
            ["Keypoints GPU / CPU", "%d / %d" % (pf["kp_gpu"], pf["kp_cpu"]), "%s / %s" % (o.get("kp_gpu"), o.get("kp_cpu"))],
            ["Exactly coinciding keypoints (%)", pf["exact_kp_pct"], f(o.get("exact_kp_pct"), 1)],
            ["Mean Hamming distance (bits of 256)", pf["mean_hamming_bits"], f(o.get("mean_hamming_bits"), 2)],
            ["Descriptor bit agreement (%)", pf["bit_agreement_pct"], f(o.get("bit_agreement_pct"), 2)],
            ["Identical descriptors (%)", pf["identical_desc_pct"], f(o.get("identical_desc_pct"), 1)]]
    return "## Feature-level equivalence (Sec. 3.1.5)\n\n" + table(["", "Paper", "Ours"], rows)


def sec_d435(R):
    seqs = sorted({k[2] for k in R if k[0] == "d435"})
    if not seqs:
        return "## Our hardware: D435 recordings\n\nNo D435 runs yet (runbook Part R6)."
    rows, agree = [], []
    for seq in seqs:
        line = [seq]
        for arm in ("gpu", "cpu", "gpu_pipe"):
            runs = R.get(("d435", "d435_s", seq, arm), [])
            t = med([con(r, "mean_track_s") for r in runs])
            loops = med([con(r, "loops") for r in runs])
            maps = med([con(r, "maps") for r in runs])
            line.append("%s ms / %s FPS / loops %s / maps %s" % (f(t * 1000 if t else None, 1),
                                                                f(1 / t if t else None, 1), f(loops, 0), f(maps, 0)))
        rows.append(line)

        def traj(arm):
            return [r[2] / "f_run.txt" for r in R.get(("d435", "d435_s", seq, arm), []) if (r[2] / "f_run.txt").is_file()]

        def pair_rms(A, B):
            out = []
            for a, b in ((x, y) for x in A for y in B if x != y):
                ta, Ta = load_tum(a)
                tb, Tb = load_tum(b)
                ia, ib = associate(ta, tb, 0.001)
                if len(ia) >= 10:
                    out.append(ate_rmse(Ta[ia, :3, 3], Tb[ib, :3, 3], False)[0] * 100)
            return med(out), len(out)
        g, c = traj("gpu"), traj("cpu")
        gc, n1 = pair_rms(g, c)
        gg, n2 = pair_rms(g, g)
        cc, n3 = pair_rms(c, c)
        agree.append([seq, "%s (%d pairs)" % (f(gc), n1), "%s (%d)" % (f(gg), n2), "%s (%d)" % (f(cc), n3)])
    s = "## Our hardware: D435 stereo recordings (no ground truth)\n\n"
    s += "Per arm: mean tracking time / FPS / loop closures / maps (medians over runs).\n\n"
    s += table(["Sequence", "GPU", "CPU (CPU_ORB=1)", "GPU + PIPELINE_FE"], rows)
    s += ("\n\nEquivalence on our data: RMS distance (cm) between trajectories after SE(3) alignment. "
          "GPU-vs-CPU should be no larger than the run-to-run spread of either arm alone.\n\n")
    s += table(["Sequence", "GPU vs CPU", "GPU vs GPU (run-to-run)", "CPU vs CPU (run-to-run)"], agree)
    return s


def sec_robustness(R):
    rows = []
    for (suite, mode, seq, arm), runs in sorted(R.items()):
        bad = [r for r in runs if not r[0].get("trajectory_written")]
        split = [r for r in runs if (con(r, "maps") or 1) > 1]
        if bad or split:
            rows.append([suite, mode, seq, arm, "%d/%d" % (len(bad), len(runs)), "%d/%d" % (len(split), len(runs))])
    if not rows:
        return "## Run health\n\nEvery run wrote a trajectory and ended in a single map."
    return "## Run health (runs without a trajectory / ending with more than one map)\n\n" + table(
        ["Suite", "Mode", "Seq", "Arm", "No trajectory", ">1 map"], rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="?", type=Path, default=RESULTS_ROOT)
    ap.add_argument("-o", "--out", type=Path)
    a = ap.parse_args()
    R = load_runs(a.results)
    any_meta = next((r[0][0] for r in R.values() if r), {})
    head = ["# Jetson-ORB-SLAM3 – reproduction of arXiv:2608.17874 on our hardware", "",
            "Generated %s from `%s`. Build %s, %s." % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                                                     a.results, any_meta.get("commit", "?"), any_meta.get("l4t", "?")),
            "Total runs: %d. Values shown as *paper / ours* or in separate columns; '–' = not run yet." % sum(len(v) for v in R.values()),
            "", ""]
    parts = [sec_euroc_accuracy(R), sec_table5(R), sec_tumvi(R), sec_kitti(R), sec_table6(R), sec_table7(R),
             sec_table8(R), sec_table9(a.results), sec_table10(R), sec_power(a.results, R), sec_features(a.results),
             sec_d435(R), sec_robustness(R)]
    text = "\n".join(head) + "\n\n".join(parts) + "\n"
    out = a.out or (a.results / "REPORT.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print("report written: %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
