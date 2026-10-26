#!/usr/bin/env python3
"""Score every run of the reproduction pipeline (plan/paper_runbook.md, Part R5).

usage:
  evaluate.py RESULTS_DIR [--force]        # score every run folder (has meta.json) below RESULTS_DIR
  evaluate.py ate EST GT [--scale]          # one-off: ATE of a trajectory against EuRoC/TUM-VI ground truth
  evaluate.py agree A B                     # one-off: SE(3)-aligned RMS distance between two trajectories

Metrics follow the paper (arXiv 2608.17874, Sec. 4.2):
  * ATE: RMS of position error after rigid alignment (Umeyama). Both SE(3) (scale fixed at 1,
    the stricter metric when scale is observable) and Sim(3) (scale optimised) are reported.
    Estimates are associated to ground truth by nearest timestamp within 20 ms, exactly as the
    repository's own scorer in run_euroc.sh. EuRoC and TUM-VI estimates from inertial modes are
    written by ORB-SLAM3 in the IMU body frame, the frame of the ground truth.
  * Relative errors: KITTI's official odometry estimator (devkit evaluate_odometry):
    first frames every 10 poses, segment lengths 100..800 m, error = inv(dEst) * dGT,
    t_rel in %, r_rel in deg/100 m. For EuRoC/TUM-VI the paper scales the windows by 1/20 (5..40 m).

numpy only (works with the Jetson's numpy 1.21).
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ground_truth, read_json, write_json  # noqa: E402

ASSOC_MAX_DT = 0.02                    # seconds, as run_euroc.sh
KITTI_LENGTHS = [100, 200, 300, 400, 500, 600, 700, 800]
SMALL_LENGTHS = [5, 10, 15, 20, 25, 30, 35, 40]
STEP = 10


# ------------------------------------------------------------------------------ loading
def quat_xyzw_to_R(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _pose(p, q_xyzw) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = quat_xyzw_to_R(np.asarray(q_xyzw, float))
    T[:3, 3] = p
    return T


def load_tum(path: Path):
    """ORB-SLAM3 EuRoC-style output: 't x y z qx qy qz qw', t in ns (or s in some builds)."""
    t, T = [], []
    for line in open(path):
        v = line.split()
        if len(v) < 8 or line.startswith("#"):
            continue
        t.append(float(v[0]))
        T.append(_pose([float(a) for a in v[1:4]], [float(a) for a in v[4:8]]))
    t = np.array(t)
    if len(t) and t[0] > 1e12:
        t = t / 1e9
    return t, np.array(T)


def load_gt_csv(path: Path):
    """EuRoC state_groundtruth_estimate0 / TUM-VI mocap0: 'ts[ns],px,py,pz,qw,qx,qy,qz,...'."""
    t, T = [], []
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        v = line.replace(",", " ").split()
        if len(v) < 8:
            continue
        qw, qx, qy, qz = (float(a) for a in v[4:8])
        t.append(float(v[0]) / 1e9)
        T.append(_pose([float(a) for a in v[1:4]], [qx, qy, qz, qw]))
    return np.array(t), np.array(T)


def load_kitti(path: Path) -> np.ndarray:
    T = []
    for line in open(path):
        v = line.split()
        if len(v) != 12:
            continue
        M = np.eye(4)
        M[:3, :4] = np.array([float(a) for a in v]).reshape(3, 4)
        T.append(M)
    return np.array(T)


# -------------------------------------------------------------------------- association
def associate(t_est: np.ndarray, t_gt: np.ndarray, max_dt: float = ASSOC_MAX_DT):
    """For each estimate, the nearest ground-truth sample within max_dt. Returns index pairs."""
    order = np.argsort(t_gt)
    ts = t_gt[order]
    ie, ig = [], []
    for k, t in enumerate(t_est):
        i = np.searchsorted(ts, t)
        best = None
        for j in (i - 1, i):
            if 0 <= j < len(ts):
                d = abs(ts[j] - t)
                if best is None or d < best[0]:
                    best = (d, j)
        if best is not None and best[0] < max_dt:
            ie.append(k)
            ig.append(order[best[1]])
    return np.array(ie, int), np.array(ig, int)


# ------------------------------------------------------------------------------ metrics
def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool):
    """Least-squares s, R, t with dst ~ s R src + t (Umeyama 1991)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    X, Y = src - mu_s, dst - mu_d
    Sigma = Y.T @ X / len(src)
    U, D, Vt = np.linalg.svd(Sigma)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S) / (X ** 2).sum(1).mean()) if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def ate_rmse(p_est: np.ndarray, p_gt: np.ndarray, with_scale: bool):
    s, R, t = umeyama(p_est, p_gt, with_scale)
    aligned = (s * (R @ p_est.T)).T + t
    return float(np.sqrt(np.mean(np.sum((aligned - p_gt) ** 2, 1)))), s


def _rot_angle(R: np.ndarray) -> float:
    return math.acos(max(min((np.trace(R) - 1) / 2, 1.0), -1.0))


def relative_errors(T_est: np.ndarray, T_gt: np.ndarray, lengths=KITTI_LENGTHS, step: int = STEP):
    """KITTI devkit estimator on index-aligned pose lists. Returns (t_rel %, r_rel deg/100m, n)."""
    if len(T_gt) < 2:
        return float("nan"), float("nan"), 0
    d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(T_gt[:, :3, 3], axis=0), axis=1))])
    t_err, r_err = [], []
    for first in range(0, len(T_gt), step):
        for L in lengths:
            last = int(np.searchsorted(d, d[first] + L, side="right"))
            if last >= len(T_gt):
                continue
            dg = np.linalg.inv(T_gt[first]) @ T_gt[last]
            de = np.linalg.inv(T_est[first]) @ T_est[last]
            E = np.linalg.inv(de) @ dg
            t_err.append(np.linalg.norm(E[:3, 3]) / L)
            r_err.append(_rot_angle(E[:3, :3]) / L)
    if not t_err:
        return float("nan"), float("nan"), 0
    return float(np.mean(t_err) * 100), float(np.degrees(np.mean(r_err)) * 100), len(t_err)


# --------------------------------------------------------------------------- console log
def parse_console(path: Path) -> dict:
    out = {}
    if not path.is_file():
        return out
    txt = path.read_text(errors="replace")
    m = re.search(r"median tracking time:\s*([0-9.eE+-]+)", txt)
    if m:
        out["median_track_s"] = float(m.group(1))
    m = re.search(r"mean tracking time:\s*([0-9.eE+-]+)", txt)
    if m:
        out["mean_track_s"] = float(m.group(1))
    m = re.findall(r"There are (\d+) maps in the atlas", txt)
    if m:
        out["maps"] = int(m[-1])
    out["loops"] = txt.count("*Loop detected")
    out["merges"] = txt.count("*Merge detected")
    out["new_maps"] = txt.count("Stored map with ID")
    out["gpu_orb"] = "GPU ORB enabled" in txt
    out["cpu_orb"] = "CPU_ORB set" in txt
    out["cnn_active"] = "[CNN] Loop detector initialized" in txt
    return out


def parse_exec_mean(path: Path) -> dict:
    """ExecMean.txt of an instrumented (REGISTER_TIMES) build: 'Label: mean$\\pm$std'."""
    out = {}
    if not path.is_file():
        return out
    for line in path.read_text(errors="replace").splitlines():
        m = re.match(r"\s*([A-Za-z ]+):\s*([0-9.eE+-]+)\$\\pm\$([0-9.eE+-]+)", line)
        if m:
            out[m.group(1).strip()] = {"mean_ms": float(m.group(2)), "std_ms": float(m.group(3))}
    return out


def parse_tegrastats(path: Path) -> dict:
    """VDD_IN instantaneous power [mW] from tegrastats lines ('VDD_IN 5834mW/5834mW')."""
    vals = []
    if path.is_file():
        for line in path.read_text(errors="replace").splitlines():
            m = re.search(r"VDD_IN (\d+)mW", line)
            if m:
                vals.append(int(m.group(1)))
    if not vals:
        return {}
    a = np.array(vals, float) / 1000.0
    return {"vdd_in_mean_w": float(a.mean()), "vdd_in_peak_w": float(a.max()), "samples": len(a)}


# ------------------------------------------------------------------------------- scoring
def score_euroc_like(run: Path, tag: str, gt_path: Path) -> dict:
    res = {}
    t_gt, T_gt = load_gt_csv(gt_path)
    for kind, fname, alt in (("kf", "kf_%s.txt" % tag, "KeyFrameTrajectory.txt"),
                             ("frames", "f_%s.txt" % tag, "CameraTrajectory.txt")):
        f = run / fname
        if not f.is_file():
            f = run / alt                     # rgbd_tum / TUM-style writers
        if not f.is_file():
            continue
        t_est, T_est = load_tum(f)
        ie, ig = associate(t_est, t_gt)
        res["%s_poses" % kind] = int(len(ie))
        res["%s_total" % kind] = int(len(t_est))
        if len(ie) < 10:
            continue
        p_e, p_g = T_est[ie, :3, 3], T_gt[ig, :3, 3]
        res["%s_ate_se3_cm" % kind] = ate_rmse(p_e, p_g, False)[0] * 100
        ate_s, scale = ate_rmse(p_e, p_g, True)
        res["%s_ate_sim3_cm" % kind] = ate_s * 100
        res["%s_sim3_scale" % kind] = scale
        if kind == "frames":
            tr, rr, n = relative_errors(T_est[ie], T_gt[ig], SMALL_LENGTHS)
            res.update(trel_5_40_pct=tr, rrel_5_40_deg100m=rr, rel_segments=n)
    return res


def score_kitti(run: Path, gt_path: Path) -> dict:
    f = run / "CameraTrajectory.txt"
    if not f.is_file():
        return {}
    T_est, T_gt = load_kitti(f), load_kitti(gt_path)
    res = {"frames_total": int(len(T_est)), "gt_total": int(len(T_gt))}
    if len(T_est) != len(T_gt):
        # The KITTI writer has no timestamps: poses can only be matched by line number.
        res["error"] = "pose count %d != ground truth %d (frames dropped?)" % (len(T_est), len(T_gt))
        return res
    tr, rr, n = relative_errors(T_est, T_gt, KITTI_LENGTHS)
    res.update(trel_pct=tr, rrel_deg100m=rr, rel_segments=n,
               ate_se3_m=ate_rmse(T_est[:, :3, 3], T_gt[:, :3, 3], False)[0])
    return res


def evaluate_run(run: Path) -> dict:
    meta = read_json(run / "meta.json")
    ev = {"console": parse_console(run / "console.log"),
          "power": parse_tegrastats(run / "tegrastats.log"),
          "timing": parse_exec_mean(run / "ExecMean.txt")}
    if meta.get("kind") == "idle":
        return ev
    gt = ground_truth(meta["mode"], meta["seq"]) if meta.get("mode") else None
    try:
        if meta.get("mode") == "kitti_s":
            ev["accuracy"] = score_kitti(run, gt)
        elif gt is not None and gt.is_file():
            ev["accuracy"] = score_euroc_like(run, meta.get("tag", "run"), gt)
    except Exception as e:                               # noqa: BLE001
        ev["accuracy"] = {"error": repr(e)}
    return ev


def evaluate_tree(root: Path, force: bool) -> int:
    n = 0
    for meta in sorted(root.rglob("meta.json")):
        run = meta.parent
        out = run / "eval.json"
        if out.is_file() and not force and out.stat().st_mtime >= meta.stat().st_mtime:
            continue
        write_json(out, evaluate_run(run))
        n += 1
    print("scored %d run(s) under %s" % (n, root))
    return 0


def main(argv) -> int:
    if not argv:
        print(__doc__)
        return 2
    if argv[0] == "ate":
        t_e, T_e = load_tum(Path(argv[1]))
        t_g, T_g = load_gt_csv(Path(argv[2]))
        ie, ig = associate(t_e, t_g)
        e, s = ate_rmse(T_e[ie, :3, 3], T_g[ig, :3, 3], "--scale" in argv)
        print("ATE RMSE %.2f cm  (%s, %d poses, scale %.4f)" % (e * 100, "Sim3" if "--scale" in argv else "SE3", len(ie), s))
        return 0
    if argv[0] == "agree":
        t_a, T_a = load_tum(Path(argv[1]))
        t_b, T_b = load_tum(Path(argv[2]))
        ia, ib = associate(t_a, t_b, 0.001)
        e, _ = ate_rmse(T_a[ia, :3, 3], T_b[ib, :3, 3], False)
        print("RMS distance after SE3 alignment: %.2f cm over %d common poses" % (e * 100, len(ia)))
        return 0
    return evaluate_tree(Path(argv[0]), "--force" in argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
