#!/usr/bin/env python3
"""Summarise rover runs (plan/stage1.md, 4.5): event counts, split detection, end-point error.

usage: summarize_run.py RUN_DIR [RUN_DIR ...]

Each RUN_DIR is a folder written by stereo_realsense_D435_rover (console.log, events.csv,
keyframes.txt). Prints one line per run, so every comparison is computed the same way.

End-point error: runs start on a floor mark, go round the loop, pass the mark again and end a
few metres further along the start of the route (ORB-SLAM3 confirms a loop only after 3
matches with the old place, which needs several keyframes over the start area, so the camera
must keep moving there). The error is the distance from the last keyframe to the nearest keyframe in the
first third of the route: after a correct loop closure the two passes lie on top of each other.
It is meaningful only when the run is not split (a new map that never merged back).
"""
import csv
import math
import statistics
import sys
from pathlib import Path


def summarize(run: Path) -> str:
    log = (run / "console.log").read_text(errors="replace") if (run / "console.log").exists() else ""
    loops = log.count("*Loop detected")
    merges = log.count("*Merge detected")
    relocs = log.count("Relocalized!!")
    new_maps = log.count("Stored map with ID")       # printed only when a new map replaces one
    dropped = sum(int(l.split()[0]) for l in log.splitlines() if l.endswith(" dropped frs"))

    states, tracked, track_ms = [], [], []
    ev = run / "events.csv"
    if ev.exists():
        with open(ev) as f:
            for r in csv.DictReader(f):
                if r["event"] == "state":
                    states.append(int(r["detail"]))
                elif r["event"] == "tracked":
                    tracked.append(int(r["detail"]))
                elif r["event"] == "track_ms":
                    track_ms.append(float(r["detail"].split()[0]))
    lost_events = sum(1 for a, b in zip([None] + states, states) if b == 4 and a != 4)
    split = new_maps > merges

    kf_file = run / "keyframes.txt"
    kf = []
    if kf_file.exists():
        kf = [list(map(float, l.split())) for l in open(kf_file) if l.strip()]
    steps = [math.dist(p[1:4], q[1:4]) for p, q in zip(kf, kf[1:])]
    path = sum(steps)
    end_err = float("nan")
    if len(kf) >= 2:
        # keyframes in the first third of the route (by distance travelled)
        early, walked = [kf[0]], 0.0
        for k, d in zip(kf[1:], steps):
            walked += d
            if walked > path / 3:
                break
            early.append(k)
        end_err = min(math.dist(kf[-1][1:4], e[1:4]) for e in early)
    pct = f"{100 * end_err / path:.2f}" if path > 0 and not split else "n/a"

    med_tracked = f"{statistics.median(tracked):.0f}" if tracked else "n/a"
    mean_ms = f"{statistics.mean(track_ms):.1f}" if track_ms else "n/a"

    return (f"run={run.name} loops={loops} merges={merges} relocalized={relocs} "
            f"new_maps={new_maps} lost_events={lost_events} split={split} "
            f"keyframes={len(kf)} path_m={path:.2f} endpoint_m={end_err:.3f} endpoint_pct={pct} "
            f"median_tracked={med_tracked} mean_track_ms={mean_ms} dropped_frames={dropped}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for arg in sys.argv[1:]:
        print(summarize(Path(arg)))
