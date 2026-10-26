#!/usr/bin/env python3
"""Run the paper's experiment matrix on the Jetson (plan/paper_runbook.md, Part R4).

usage:
  run_matrix.py SUITE [--seqs S ...] [--runs N] [--arms A ...] [--build-root DIR]
                      [--results DIR] [--engine FILE] [--tegra-ms MS] [--pause S]
                      [--redo] [--dry-run]

Suites (paper table in brackets):
  euroc-accuracy    EuRoC stereo-inertial, 11 seq x {gpu, cpu} x 5 runs       [Tables 1, 2, 5, 8]
  euroc-throughput  EuRoC mono- and stereo-inertial x {gpu, gpu_pipe} x 1 run  [Table 7]
  lc-ablation       EuRoC stereo-inertial x {cnn, dbow2, nolc} x 5 runs         [Table 10]
  tumvi             TUM-VI rooms 1-6 stereo-inertial x {gpu, cpu} x 1 run       [Tables 3, 5, 8]
  kitti             KITTI 00-10 stereo x {gpu, cpu} x 1 run                     [Tables 4, 5, 8]
  instrumented      MH01 stereo-inertial + V101 mono-inertial, GPU, 1 run, with the
                    REGISTER_TIMES build (--build-root ../Jetson-ORB-SLAM3-timing) [Table 6]
  power             60 s idle, then MH01 stereo-inertial GPU, tegrastats @250 ms  [Sec. 4.1]
  d435              our own D435 recordings x {gpu, cpu, gpu_pipe} x 3 runs     [our hardware]
  ext               external D435 recordings with lidar reference, x {gpu, cpu, gpu_pipe} x 5 runs
  abl               ablation variants of an external recording (--seqs, --arms), x 3 runs
                    (plan/ablation_plan_husky.md)
                    (plan/evaluation_plan_stage_1.md)

Every configuration uses the repository's own, unmodified programs and run-time switches:
  front-end  CPU_ORB=1 selects the reference CPU extractor (src/Tracking.cc:605)
  pipelining PIPELINE_FE=1
  loop       'off'   -> settings with loopClosing: 0 (Examples/Stereo-Inertial/EuRoC_noloop.yaml)
             'cnn'   -> the TensorRT engine is linked into the run folder (the library looks for
                        cosplace_r50_512.fp16.trt in the working directory)
             'dbow2' -> no engine in the run folder
             'auto'  -> engine linked if it exists (--engine), else DBoW2 only
Each run gets its own folder with console.log, trajectories, tegrastats.log and meta.json.
Finished runs are skipped, so the command can simply be re-run after an interruption.
"""
from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (ARMS, CNN_ENGINE_NAME, EUROC, KITTI, MODES, REPO, RESULTS_ROOT, TUMVI,  # noqa: E402
                    ext_mode, git_commit, ground_truth, l4t_release, library_path, list_d435_sequences, list_ext_sequences,
                    mode_command, read_json, write_json)

SUITES = {
    "euroc-accuracy":   dict(modes=["euroc_si"], arms=["gpu", "cpu"], runs=5, seqs=list(EUROC)),
    "euroc-throughput": dict(modes=["euroc_mi", "euroc_si"], arms=["gpu", "gpu_pipe"], runs=1, seqs=list(EUROC)),
    "lc-ablation":      dict(modes=["euroc_si"], arms=["cnn", "dbow2", "nolc"], runs=5, seqs=list(EUROC)),
    "tumvi":            dict(modes=["tumvi_si"], arms=["gpu", "cpu"], runs=1, seqs=TUMVI),
    "kitti":            dict(modes=["kitti_s"], arms=["gpu", "cpu"], runs=1, seqs=KITTI),
    "instrumented":     dict(jobs=[("euroc_si", "MH01"), ("euroc_mi", "V101")], arms=["gpu"], runs=1),
    "power":            dict(modes=["euroc_si"], arms=["gpu"], runs=1, seqs=["MH01"], idle_s=60, tegra_ms=250),
    "d435":             dict(modes=["d435_s"], arms=["gpu", "cpu", "gpu_pipe"], runs=3, seqs=None),
    "ext":              dict(modes=["ext_s"], arms=["gpu", "cpu", "gpu_pipe"], runs=5, seqs=None),
    # ablation cells: each variant folder's mode.txt picks the mode (plan/ablation_plan_husky.md)
    "abl":              dict(modes=["ext_auto"], arms=["gpu"], runs=3, seqs=None),
}
JOB_TIMEOUT_S = 3 * 3600          # a hung run must not block the whole campaign
CLEAR_ENV = ("CPU_ORB", "PIPELINE_FE", "ADAPTIVE_FE", "ADAPTIVE_LC", "DYNAMIC_MASK_ENGINE",
             "AFE_ECO_NFEAT", "AFE_ECO_SCALE", "AFE_ECO_NLEVELS", "AFE_CALM", "AFE_AGGR")


def say(msg: str) -> None:
    print("==> " + msg, flush=True)


def settings_for(mode: str, loop: str, run_dir: Path, build_root: Path) -> str | None:
    if loop != "off":
        return None
    if MODES[mode]["dataset"] == "ext":
        raise SystemExit("arm 'nolc' is not supported for external sequences: use the __nolc ablation variant")
    if mode == "euroc_si":
        return str(build_root / "Examples/Stereo-Inertial/EuRoC_noloop.yaml")
    # Other modes: a copy of the mode's settings with loop closing disabled, in the run folder.
    src = build_root / MODES[mode]["settings"]
    dst = run_dir / ("noloop_" + src.name)
    dst.write_text(src.read_text().rstrip("\n") + "\n\nloopClosing: 0\n")
    return str(dst)


def preflight(build_root: Path, jobs, engine: Path | None) -> list[str]:
    problems = []
    if not (build_root / "Vocabulary/ORBvoc.txt").is_file():
        problems.append("missing %s/Vocabulary/ORBvoc.txt (run ./run_euroc.sh once to unpack it)" % build_root)
    for mode in sorted({j[0] for j in jobs}):
        b = build_root / MODES[mode]["binary"]
        if not os.access(b, os.X_OK):
            problems.append("missing program %s (build the repository first)" % b)
        if MODES[mode]["viewer"] and not shutil.which("xvfb-run"):
            problems.append("%s opens a viewer: install xvfb (sudo apt install xvfb)" % mode)
    for mode, seq, arm, _ in jobs:
        gt = ground_truth(mode, seq)
        cmd = mode_command(mode, seq, build_root)
        data = Path(cmd[3])
        if not data.exists():
            problems.append("missing data for %s %s: %s (eval/benchmark/fetch_datasets.py)" % (mode, seq, data))
        if gt is not None and not gt.exists() and MODES[mode]["dataset"] != "ext":   # ext: scored later, on the PC
            problems.append("missing ground truth %s" % gt)
        if ARMS[arm]["loop"] == "cnn" and not (engine and engine.is_file()):
            problems.append("arm 'cnn' needs the TensorRT engine (--engine, runbook Part R3)")
    if not shutil.which("tegrastats"):
        problems.append("tegrastats not found (this must run on the Jetson)")
    return sorted(set(problems))


def run_one(run_dir: Path, mode: str, seq: str, arm: str, k: int, build_root: Path,
            engine: Path | None, tegra_ms: int, suite: str, dry: bool) -> bool:
    a = ARMS[arm]
    tag = "run"
    use_engine = a["loop"] == "cnn" or (a["loop"] == "auto" and engine is not None)
    if dry:
        settings = "(loopClosing: 0)" if a["loop"] == "off" else None
        cmd = mode_command(mode, seq, build_root, None, tag)
        print("   [dry] %s  env:%s%s  engine:%s  settings:%s\n         %s%s" % (
            run_dir, " CPU_ORB=1" if a["front"] == "cpu" else "",
            (" PIPELINE_FE=1" if a["pipe"] else "") + "".join(" %s=%s" % kv for kv in a.get("env", {}).items()),
            "linked" if use_engine else "none", settings or "default",
            "xvfb-run ... " if MODES[mode]["viewer"] else "", " ".join(cmd)))
        return True
    if use_engine and engine is None:
        raise SystemExit("arm %s needs the TensorRT engine (--engine, runbook Part R3)" % arm)

    run_dir.mkdir(parents=True, exist_ok=True)
    settings = settings_for(mode, a["loop"], run_dir, build_root)
    cmd = mode_command(mode, seq, build_root, settings, tag)
    if MODES[mode]["viewer"]:
        # Pangolin needs a display; a 24-bit virtual screen keeps GLX happy.
        cmd = ["xvfb-run", "-a", "-s", "-screen 0 1280x1024x24"] + cmd

    eng_link = run_dir / CNN_ENGINE_NAME
    if eng_link.is_symlink() or eng_link.exists():
        eng_link.unlink()
    if use_engine:
        eng_link.symlink_to(engine.resolve())

    env = {k_: v for k_, v in os.environ.items() if k_ not in CLEAR_ENV}
    env["LD_LIBRARY_PATH"] = library_path(build_root)
    if a["front"] == "cpu":
        env["CPU_ORB"] = "1"
    if a["pipe"]:
        env["PIPELINE_FE"] = "1"
    env.update(a.get("env", {}))

    meta = dict(suite=suite, mode=mode, seq=seq, arm=arm, run=k, tag=tag, cmd=cmd,
                front=a["front"], pipe=a["pipe"], loop=a["loop"], engine_linked=use_engine,
                build_root=str(build_root), commit=git_commit(build_root), l4t=l4t_release(),
                settings=settings or "default", done=False)
    write_json(run_dir / "meta.json", meta)

    tegra = subprocess.Popen(["tegrastats", "--interval", str(tegra_ms)],
                             stdout=open(run_dir / "tegrastats.log", "w"), stderr=subprocess.STDOUT)
    t0 = time.time()
    rc = None
    try:
        with open(run_dir / "console.log", "w") as log:
            p = subprocess.Popen(cmd, cwd=run_dir, env=env, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
            try:
                rc = p.wait(timeout=JOB_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                rc = "timeout"
    finally:
        tegra.terminate()
        try:
            tegra.wait(timeout=5)
        except subprocess.TimeoutExpired:
            tegra.kill()
    # Trajectory written? (programs may still exit non-zero while tearing down their threads)
    out_ok = (run_dir / "CameraTrajectory.txt").is_file() if MODES[mode]["output"] in ("kitti", "tum") \
        else (run_dir / ("kf_%s.txt" % tag)).is_file()
    meta.update(done=True, exit_code=rc, wall_s=round(time.time() - t0, 1), trajectory_written=out_ok,
                timing_stats=(run_dir / "ExecMean.txt").is_file())
    if suite == "instrumented" and not meta["timing_stats"]:
        print("    no ExecMean.txt: this build was not compiled with REGISTER_TIMES (runbook Part R3)")
    write_json(run_dir / "meta.json", meta)
    return out_ok


def run_idle(run_dir: Path, seconds: int, tegra_ms: int, dry: bool) -> None:
    if dry:
        print("   [dry] idle %d s -> %s" % (seconds, run_dir))
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = dict(kind="idle", seconds=seconds, l4t=l4t_release(), done=False)
    write_json(run_dir / "meta.json", meta)
    with open(run_dir / "tegrastats.log", "w") as log:
        t = subprocess.Popen(["tegrastats", "--interval", str(tegra_ms)], stdout=log, stderr=subprocess.STDOUT)
        time.sleep(seconds)
        t.terminate()
        t.wait(timeout=5)
    meta["done"] = True
    write_json(run_dir / "meta.json", meta)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("suite", choices=sorted(SUITES))
    ap.add_argument("--seqs", nargs="*")
    ap.add_argument("--arms", nargs="*", choices=sorted(ARMS))
    ap.add_argument("--runs", type=int)
    ap.add_argument("--build-root", type=Path, default=REPO)
    ap.add_argument("--results", type=Path, default=RESULTS_ROOT)
    ap.add_argument("--engine", type=Path, default=REPO / CNN_ENGINE_NAME,
                    help="TensorRT engine for CNN loop closure (runbook Part R3)")
    ap.add_argument("--tegra-ms", type=int)
    ap.add_argument("--pause", type=float, default=10.0, help="seconds of rest between runs (thermal)")
    ap.add_argument("--idle-seconds", type=int, help="length of the power suite's idle baseline (default 60)")
    ap.add_argument("--redo", action="store_true", help="re-run finished runs")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    s = SUITES[a.suite]
    build_root = a.build_root.resolve()
    engine = a.engine if a.engine and a.engine.is_file() else None
    runs = a.runs or s["runs"]
    arms = a.arms or s["arms"]
    tegra_ms = a.tegra_ms or s.get("tegra_ms", 1000)
    if "jobs" in s:
        pairs = [(m, q) for m, q in s["jobs"] if not a.seqs or q in a.seqs]
    else:
        listed = list_ext_sequences() if a.suite in ("ext", "abl") else list_d435_sequences()
        if a.suite == "ext":                  # base sequences only; "__variant" folders belong to abl
            listed = [q for q in listed if "__" not in q]
        seqs = a.seqs or (s["seqs"] if s["seqs"] is not None else listed)
        pairs = [(ext_mode(q) if m == "ext_auto" else m, q) for m in s["modes"] for q in seqs]
    # Runs outermost: a partial campaign still has every sequence and arm equally covered,
    # and alternating arms spreads thermal drift over both.
    jobs = [(m, q, arm, k) for k in range(1, runs + 1) for (m, q) in pairs for arm in arms]
    if not jobs:
        say("nothing to run (for d435: record sequences first, runbook Part R6)")
        return 1

    problems = preflight(build_root, jobs, engine)
    if a.suite == "instrumented" and build_root == REPO.resolve():
        problems.append("the instrumented suite needs the REGISTER_TIMES build: "
                        "--build-root ../Jetson-ORB-SLAM3-timing (eval/benchmark/setup_timing_build.sh)")
    if problems and not a.dry_run:
        say("cannot start:")
        for p in problems:
            print("    - " + p)
        return 1

    suite_dir = a.results / a.suite
    say("%s: %d run(s), build %s (%s), results %s" % (a.suite, len(jobs), build_root,
                                                      git_commit(build_root), suite_dir))
    if engine is None and any(ARMS[x]["loop"] == "auto" for x in arms):
        say("no TensorRT engine at %s: 'auto' arms run with DBoW2-only loop closure" % a.engine)

    if "idle_s" in s:
        idle_dir = suite_dir / "idle"
        if a.redo or not (idle_dir / "meta.json").is_file() or not read_json(idle_dir / "meta.json").get("done"):
            say("idle baseline %d s (nothing else should be running)" % s["idle_s"])
            run_idle(idle_dir, a.idle_seconds or s["idle_s"], tegra_ms, a.dry_run)

    failed = 0
    for i, (mode, seq, arm, k) in enumerate(jobs, 1):
        run_dir = suite_dir / mode / seq / arm / ("run%d" % k)
        meta = run_dir / "meta.json"
        if not a.redo and meta.is_file() and read_json(meta).get("done"):
            continue
        say("[%d/%d] %s %s %s run %d" % (i, len(jobs), mode, seq, arm, k))
        ok = run_one(run_dir, mode, seq, arm, k, build_root, engine, tegra_ms, a.suite, a.dry_run)
        if not ok:
            failed += 1
            print("    no trajectory written -- see %s/console.log" % run_dir)
        if not a.dry_run and i < len(jobs):
            time.sleep(a.pause)
    say("done; %d run(s) without a trajectory. Next: python3 eval/benchmark/evaluate.py %s" % (failed, a.results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
