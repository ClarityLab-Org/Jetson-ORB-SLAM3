"""Shared definitions for the paper-reproduction pipeline (plan/paper_runbook.md).

Nothing here modifies the repository: it only describes how to call the repository's own,
unmodified dataset programs (Examples/...) and where their inputs and outputs live.

Python 3.10 / standard library only (the Jetson has Python 3.10 and numpy 1.21).
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

# --------------------------------------------------------------------------------- paths
REPO = Path(__file__).resolve().parents[2]            # the repository root (eval/benchmark/../..)
DATA_ROOT = Path(os.environ.get("REPRO_DATA", Path.home() / "datasets"))
RESULTS_ROOT = Path(os.environ.get("REPRO_RESULTS", Path.home() / "repro_results"))
VOCAB = "Vocabulary/ORBvoc.txt"                       # relative to a build root
CNN_ENGINE_NAME = "cosplace_r50_512.fp16.trt"          # looked up in the *working directory*
                                                       # by src/CNNLoopDetector.cpp

# ------------------------------------------------------------------------------ datasets
EUROC = {  # sequence id -> (folder name, ETH bundle)  -- same table as run_euroc.sh
    "MH01": ("MH_01_easy", "machine_hall"),
    "MH02": ("MH_02_easy", "machine_hall"),
    "MH03": ("MH_03_medium", "machine_hall"),
    "MH04": ("MH_04_difficult", "machine_hall"),
    "MH05": ("MH_05_difficult", "machine_hall"),
    "V101": ("V1_01_easy", "vicon_room1"),
    "V102": ("V1_02_medium", "vicon_room1"),
    "V103": ("V1_03_difficult", "vicon_room1"),
    "V201": ("V2_01_easy", "vicon_room2"),
    "V202": ("V2_02_medium", "vicon_room2"),
    "V203": ("V2_03_difficult", "vicon_room2"),
}
TUMVI = ["room1", "room2", "room3", "room4", "room5", "room6"]
KITTI = ["%02d" % i for i in range(11)]                 # 00..10 have public ground truth


def euroc_dir(seq: str) -> Path:
    return DATA_ROOT / "euroc" / EUROC[seq][0]


def tumvi_dir(room: str) -> Path:
    return DATA_ROOT / "tumvi" / ("dataset-%s_512_16" % room)


def kitti_seq_dir(seq: str) -> Path:
    return DATA_ROOT / "kitti" / "sequences" / seq


def kitti_gt(seq: str) -> Path:
    return DATA_ROOT / "kitti" / "poses" / ("%s.txt" % seq)


def kitti_settings(seq: str) -> str:
    n = int(seq)
    if n <= 2:
        return "Examples/Stereo/KITTI00-02.yaml"
    if n == 3:
        return "Examples/Stereo/KITTI03.yaml"
    return "Examples/Stereo/KITTI04-12.yaml"


def d435_dir(name: str) -> Path:
    return DATA_ROOT / "d435" / name


def list_d435_sequences() -> list[str]:
    root = DATA_ROOT / "d435"
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if (p / "times.txt").is_file())


def ext_dir(name: str) -> Path:
    return DATA_ROOT / "ext" / name


def ext_mode(name: str) -> str:
    """Mode of an external sequence folder: its mode.txt (ablation variants), else stereo."""
    f = ext_dir(name) / "mode.txt"
    return f.read_text().strip() if f.is_file() else "ext_s"


def list_ext_sequences() -> list[str]:
    """External recordings converted by eval/processing/ (plan/evaluation_plan_stage_1.md)."""
    root = DATA_ROOT / "ext"
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir()
                  if all((p / f).is_file() for f in ("times.txt", "settings.yaml")))


# ---------------------------------------------------------------- how each mode is run
# Each mode names the repository's own (unmodified) program, its settings file, whether the
# program always opens the Pangolin viewer (then it needs a display -> xvfb-run), and how
# its arguments and outputs look.
MODES = {
    # EuRoC stereo-inertial: the configuration of Tables 1, 2, 6 (MH01), 7, 8, 10
    "euroc_si": dict(binary="Examples/Stereo-Inertial/stereo_inertial_euroc",
                     settings="Examples/Stereo-Inertial/EuRoC.yaml", viewer=False,
                     dataset="euroc", gt_frame="body", output="euroc"),
    # EuRoC monocular-inertial: Tables 6 (V101) and 7
    "euroc_mi": dict(binary="Examples/Monocular-Inertial/mono_inertial_euroc",
                     settings="Examples/Monocular-Inertial/EuRoC.yaml", viewer=False,
                     dataset="euroc", gt_frame="body", output="euroc"),
    # TUM-VI stereo-inertial, rooms: Table 3
    "tumvi_si": dict(binary="Examples/Stereo-Inertial/stereo_inertial_tum_vi",
                     settings="Examples/Stereo-Inertial/TUM-VI.yaml", viewer=True,
                     dataset="tumvi", gt_frame="body", output="euroc"),
    # KITTI stereo: Table 4
    "kitti_s": dict(binary="Examples/Stereo/stereo_kitti", settings=None, viewer=True,
                    dataset="kitti", gt_frame="camera", output="kitti"),
    # Our own D435 recordings (EuRoC folder layout), stereo, camera frame, no ground truth
    "d435_s": dict(binary="Examples/Stereo/stereo_euroc",
                   settings="Examples/Stereo/RealSense_D435.yaml", viewer=True,
                   dataset="d435", gt_frame=None, output="euroc"),
    # External D435 recordings with a lidar reference (plan/evaluation_plan_stage_1.md): the same
    # program, but each sequence brings its own settings.yaml (their calibration) and gt.csv
    "ext_s": dict(binary="Examples/Stereo/stereo_euroc", settings=None, viewer=True,
                  dataset="ext", gt_frame="camera", output="euroc"),
    # Ablation variants of an external recording (plan/ablation_plan_husky.md). The variant folder's
    # mode.txt names the mode; its gt.csv is in that mode's output frame (camera, IMU body, colour camera).
    "ext_m":    dict(binary="Examples/Monocular/mono_euroc", settings=None, viewer=False,
                     dataset="ext", gt_frame="camera", output="euroc"),
    "ext_mi":   dict(binary="Examples/Monocular-Inertial/mono_inertial_euroc", settings=None, viewer=False,
                     dataset="ext", gt_frame="body", output="euroc"),
    "ext_si":   dict(binary="Examples/Stereo-Inertial/stereo_inertial_euroc", settings=None, viewer=False,
                     dataset="ext", gt_frame="body", output="euroc"),
    "ext_rgbd": dict(binary="Examples/RGB-D/rgbd_tum", settings=None, viewer=True,
                     dataset="ext", gt_frame="camera", output="tum"),
}

# Arms: what changes between configurations. Everything is a run-time switch the
# repository already provides -- no rebuild, no source change.
#   front : "gpu" (default) or "cpu" (env CPU_ORB=1, src/Tracking.cc:605)
#   pipe  : env PIPELINE_FE=1 (overlap extraction with tracking, stereo modes only)
#   loop  : "auto" (CNN if the TensorRT engine is supplied, else DBoW2), "cnn", "dbow2",
#           "off" (settings with loopClosing: 0, i.e. Examples/Stereo-Inertial/EuRoC_noloop.yaml)
ARMS = {
    # env: extra run-time switches the repository reads (src/Tracking.cc, src/LoopClosing.cc)
    "adaptive_fe": dict(front="gpu", pipe=False, loop="auto", env={"ADAPTIVE_FE": "1"}),   # IMU_STEREO only
    "adaptive_lc": dict(front="gpu", pipe=False, loop="auto", env={"ADAPTIVE_LC": "1"}),   # inertial loops only
    "gpu":       dict(front="gpu", pipe=False, loop="auto"),
    "cpu":       dict(front="cpu", pipe=False, loop="auto"),
    "gpu_pipe":  dict(front="gpu", pipe=True,  loop="auto"),
    "cnn":       dict(front="gpu", pipe=False, loop="cnn"),
    "dbow2":     dict(front="gpu", pipe=False, loop="dbow2"),
    "nolc":      dict(front="gpu", pipe=False, loop="off"),
}


def mode_command(mode: str, seq: str, build_root: Path, settings_override: str | None = None,
                 tag: str = "run") -> list[str]:
    """Command line (without xvfb) that runs `seq` in `mode` with the given build."""
    m = MODES[mode]
    binary = str(build_root / m["binary"])
    voc = str(build_root / VOCAB)
    if mode in ("ext_s", "ext_m", "ext_mi", "ext_si"):
        d = ext_dir(seq)
        return [binary, voc, settings_override or str(d / "settings.yaml"), str(d), str(d / "times.txt"), tag]
    if mode == "ext_rgbd":                     # rgbd_tum: no tag; writes CameraTrajectory.txt (TUM, seconds)
        d = ext_dir(seq)
        return [binary, voc, settings_override or str(d / "settings.yaml"), str(d), str(d / "assoc.txt")]
    settings = settings_override or m["settings"] or kitti_settings(seq)
    settings = str(build_root / settings) if not os.path.isabs(settings) else settings
    if mode in ("euroc_si", "euroc_mi"):
        sub = "Stereo-Inertial" if mode == "euroc_si" else "Monocular-Inertial"
        times = build_root / "Examples" / sub / "EuRoC_TimeStamps" / ("%s.txt" % seq)
        return [binary, voc, settings, str(euroc_dir(seq)), str(times), tag]
    if mode == "tumvi_si":
        d = tumvi_dir(seq)
        return [binary, voc, settings, str(d / "mav0/cam0/data"), str(d / "mav0/cam1/data"),
                str(d / "orbslam_times.txt"), str(d / "orbslam_imu.txt"), tag]
    if mode == "kitti_s":
        return [binary, voc, settings, str(kitti_seq_dir(seq))]
    if mode == "d435_s":
        d = d435_dir(seq)
        return [binary, voc, settings, str(d), str(d / "times.txt"), tag]
    raise ValueError(mode)


def ground_truth(mode: str, seq: str) -> Path | None:
    ds = MODES[mode]["dataset"]
    if ds == "euroc":
        return euroc_dir(seq) / "mav0/state_groundtruth_estimate0/data.csv"
    if ds == "tumvi":
        return tumvi_dir(seq) / "mav0/mocap0/data.csv"
    if ds == "kitti":
        return kitti_gt(seq)
    if ds == "ext":
        return ext_dir(seq) / "gt.csv"
    return None


def library_path(build_root: Path) -> str:
    """Same search path run_euroc.sh uses for a source build (each build sees only its own lib)."""
    parts = [build_root / "lib", Path.home() / "local/lib", "/usr/local/lib",
             "/usr/lib/aarch64-linux-gnu", "/usr/local/cuda/lib64",
             build_root / "Thirdparty/DBoW2/lib", build_root / "Thirdparty/g2o/lib"]
    return ":".join(str(p) for p in parts)


# ----------------------------------------------------------------------------- helpers
def git_commit(root: Path) -> str:
    try:
        c = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=10).stdout.strip()
        d = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True, timeout=10).stdout.strip()
        return (c + ("+modified" if d else "")) if c else "unknown"
    except Exception:
        return "unknown"


def l4t_release() -> str:
    try:
        return Path("/etc/nv_tegra_release").read_text().splitlines()[0].strip()
    except Exception:
        return "unknown"


def write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    tmp.replace(path)


def read_json(path: Path):
    return json.loads(path.read_text())
