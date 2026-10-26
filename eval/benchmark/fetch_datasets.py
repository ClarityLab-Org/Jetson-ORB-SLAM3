#!/usr/bin/env python3
"""Fetch / prepare the public benchmarks used by the paper (plan/paper_runbook.md, Part R2).

usage:
  fetch_datasets.py euroc [SEQ ...]      # all 11 EuRoC sequences by default (~1.5 GB each)
  fetch_datasets.py tumvi [room1 ...]    # TUM-VI rooms 1-6, 512x512 EuRoC export
  fetch_datasets.py tumvi-prepare        # (re)write the times/IMU files the TUM-VI driver needs
  fetch_datasets.py kitti-check          # KITTI needs a manual download; this checks the layout
  fetch_datasets.py status               # what is present

Datasets go to $REPRO_DATA (default ~/datasets). Existing, complete sequences are skipped, so
the command can be re-run after an interruption. The EuRoC download is the repository's own
method (run_euroc.sh): range-GET one nested zip out of the ETH Research Collection bundle.
"""
from __future__ import annotations

import os
import struct
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_ROOT, EUROC, KITTI, TUMVI, euroc_dir, kitti_gt, kitti_seq_dir, tumvi_dir  # noqa: E402

# Same source and identifiers as run_euroc.sh.
BITSTREAM = {"machine_hall": "7b2419c1-62b5-4714-b7f8-485e5fe3e5fe",
             "vicon_room1": "02ecda9a-298f-498b-970c-b7c44334d880",
             "vicon_room2": "ea12bc01-3677-4b4c-853d-87c7870b8c44"}
URL = "https://www.research-collection.ethz.ch/server/api/core/bitstreams/%s/content"
# A non-browser User-Agent is answered with a "scraping detected" HTML page, not a 403.
UA = ("Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/126.0 Safari/537.36")
KEEP = ("cam0/", "cam1/", "imu0/", "state_groundtruth_estimate0/")

# TUM-VI "exported/euroc/512_16" archives (check https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset
# if this host ever moves; the archive name stays dataset-roomN_512_16.tar).
TUMVI_URL = "https://cdn3.vision.in.tum.de/tumvi/exported/euroc/512_16/dataset-%s_512_16.tar"


def say(msg: str) -> None:
    print("==> " + msg, flush=True)


# ------------------------------------------------------------------------------- EuRoC
def euroc_complete(seq: str) -> bool:
    d = euroc_dir(seq) / "mav0"
    gt = d / "state_groundtruth_estimate0" / "data.csv"
    return gt.is_file() and gt.stat().st_size > 0 and (d / "cam0/data").is_dir() \
        and (d / "cam1/data").is_dir() and (d / "imu0/data.csv").is_file()


def _rng(url: str, a, b=""):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Range": "bytes=%s-%s" % (a, b)})
    return urllib.request.urlopen(req, timeout=120)


def fetch_euroc_one(seq: str) -> None:
    canon, bundle = EUROC[seq]
    dest = euroc_dir(seq)
    url = URL % BITSTREAM[bundle]
    dest.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(dest) + ".inner.zip")

    size = int(_rng(url, 0, 0).headers["Content-Range"].split("/")[1])
    tail = _rng(url, size - 66000, size - 1).read()
    i = tail.rfind(b"PK\x06\x06")                       # ZIP64 end-of-central-directory
    if i >= 0:
        cdsz, cdoff = struct.unpack("<QQ", tail[i + 40:i + 56])
    else:
        i = tail.rfind(b"PK\x05\x06")
        cdsz, cdoff = struct.unpack("<II", tail[i + 12:i + 20])
    cd = _rng(url, cdoff, cdoff + cdsz - 1).read()

    entry, p = None, 0
    while p < len(cd) and cd[p:p + 4] == b"PK\x01\x02":
        csz, usz = struct.unpack("<II", cd[p + 20:p + 28])
        nl, el, cl = struct.unpack("<HHH", cd[p + 28:p + 34])
        lho = struct.unpack("<I", cd[p + 42:p + 46])[0]
        name = cd[p + 46:p + 46 + nl].decode("utf8", "replace")
        ex = cd[p + 46 + nl:p + 46 + nl + el]
        if 0xffffffff in (csz, usz, lho):               # ZIP64 extra field
            q = 0
            while q + 4 <= len(ex):
                hid, hsz = struct.unpack("<HH", ex[q:q + 4])
                blk, k = ex[q + 4:q + 4 + hsz], 0
                if hid == 1:
                    if usz == 0xffffffff:
                        usz = struct.unpack("<Q", blk[k:k + 8])[0]; k += 8
                    if csz == 0xffffffff:
                        csz = struct.unpack("<Q", blk[k:k + 8])[0]; k += 8
                    if lho == 0xffffffff:
                        lho = struct.unpack("<Q", blk[k:k + 8])[0]
                q += 4 + hsz
        if name.endswith("/%s/%s.zip" % (canon, canon)):
            entry = (lho, csz)
            break
        p += 46 + nl + el + cl
    if entry is None:
        raise RuntimeError("%s.zip not found inside the %s bundle" % (canon, bundle))

    lho, csz = entry
    hdr = _rng(url, lho, lho + 29).read()
    nl, el = struct.unpack("<HH", hdr[26:30])
    start = lho + 30 + nl + el
    try:
        r = _rng(url, start, start + csz - 1)
        dec, got = zlib.decompressobj(-15), 0
        with open(tmp, "wb") as out:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                out.write(dec.decompress(chunk))
                got += len(chunk)
                sys.stdout.write("\r    %5.1f%%  %.2f GB" % (100.0 * got / csz, got / 1e9))
                sys.stdout.flush()
            out.write(dec.flush())
        print()
        with zipfile.ZipFile(tmp) as z:
            for m in z.namelist():
                j = m.find("mav0/")
                if j < 0 or m.endswith("/") or not m[j + 5:].startswith(KEEP):
                    continue
                tgt = dest / m[j:]
                tgt.parent.mkdir(parents=True, exist_ok=True)
                with z.open(m) as src, open(tgt, "wb") as dst:
                    dst.write(src.read())
    finally:
        if tmp.exists():
            tmp.unlink()


def fetch_euroc(seqs: list[str]) -> int:
    failed = []
    for seq in seqs:
        if euroc_complete(seq):
            say("%s already present: %s" % (seq, euroc_dir(seq)))
            continue
        say("EuRoC %s -> %s (~1.5 GB download, ~5 GB free needed while unpacking)" % (seq, euroc_dir(seq)))
        for attempt in range(2):
            try:
                fetch_euroc_one(seq)
                break
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt == 0:
                    # The ETH server rate-limits; retrying in a loop only extends the block.
                    say("HTTP 429 (rate-limited). Waiting 20 min, then one retry.")
                    time.sleep(1200)
                    continue
                print("    failed: %s" % e)
                break
            except Exception as e:                       # noqa: BLE001
                print("    failed: %s" % e)
                break
        if not euroc_complete(seq):
            failed.append(seq)
    if failed:
        say("NOT complete: %s  (re-run later; completed ones are skipped)" % " ".join(failed))
        return 1
    return 0


# ------------------------------------------------------------------------------ TUM-VI
def tumvi_prepare(room: str) -> None:
    """Write the two text files stereo_inertial_tum_vi reads (the README says the upstream copies
    were dropped from this branch): one image timestamp [ns] per line, and the IMU samples as
    'ts,wx,wy,wz,ax,ay,az' -- exactly the layout of the dataset's own imu0/data.csv."""
    d = tumvi_dir(room) / "mav0"
    times_out = tumvi_dir(room) / "orbslam_times.txt"
    imu_out = tumvi_dir(room) / "orbslam_imu.txt"
    left = sorted(p.stem for p in (d / "cam0/data").glob("*.png"))
    right = {p.stem for p in (d / "cam1/data").glob("*.png")}
    stamps = [s for s in left if s in right]
    times_out.write_text("".join(s + "\n" for s in stamps))
    with open(d / "imu0/data.csv") as src, open(imu_out, "w") as dst:
        for line in src:
            line = line.strip()
            if line and not line.startswith("#"):
                dst.write(line + "\n")
    say("%s: %d stereo frames, IMU file written" % (room, len(stamps)))


def tumvi_complete(room: str) -> bool:
    d = tumvi_dir(room) / "mav0"
    return (d / "cam0/data").is_dir() and (d / "cam1/data").is_dir() and \
        (d / "imu0/data.csv").is_file() and (d / "mocap0/data.csv").is_file()


def fetch_tumvi(rooms: list[str]) -> int:
    failed = []
    for room in rooms:
        if not tumvi_complete(room):
            url = TUMVI_URL % room
            tar = DATA_ROOT / "tumvi" / ("dataset-%s_512_16.tar" % room)
            tar.parent.mkdir(parents=True, exist_ok=True)
            say("TUM-VI %s <- %s" % (room, url))
            try:
                req = urllib.request.Request(url, headers={"User-Agent": UA})
                with urllib.request.urlopen(req, timeout=120) as r, open(tar, "wb") as out:
                    total = int(r.headers.get("Content-Length", 0)) or 1
                    got = 0
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        out.write(chunk)
                        got += len(chunk)
                        sys.stdout.write("\r    %5.1f%%  %.2f GB" % (100.0 * got / total, got / 1e9))
                        sys.stdout.flush()
                print()
                with tarfile.open(tar) as t:
                    t.extractall(tar.parent)
                tar.unlink()
            except Exception as e:                       # noqa: BLE001
                print("    failed: %s\n    download it manually into %s and re-run 'tumvi-prepare'"
                      % (e, tar.parent))
        if tumvi_complete(room):
            tumvi_prepare(room)
        else:
            failed.append(room)
    if failed:
        say("NOT complete: %s" % " ".join(failed))
        return 1
    return 0


# ------------------------------------------------------------------------------- KITTI
def kitti_check() -> int:
    """KITTI odometry requires registration, so it is downloaded by hand:
    data_odometry_gray.zip (~22 GB) and data_odometry_poses.zip, both unpacked into
    $REPRO_DATA/kitti so that sequences/00/{image_0,image_1,times.txt} and poses/00.txt exist."""
    bad = []
    for s in KITTI:
        d = kitti_seq_dir(s)
        ok = (d / "image_0").is_dir() and (d / "image_1").is_dir() and (d / "times.txt").is_file() \
            and kitti_gt(s).is_file()
        if ok:
            n_img = sum(1 for _ in (d / "image_0").glob("*.png"))
            n_gt = sum(1 for _ in open(kitti_gt(s)))
            n_t = sum(1 for _ in open(d / "times.txt"))
            ok = n_img == n_gt == n_t
            print("  %s  images=%d  poses=%d  times=%d  %s" % (s, n_img, n_gt, n_t, "OK" if ok else "MISMATCH"))
        else:
            print("  %s  missing (expected %s and %s)" % (s, d, kitti_gt(s)))
        if not ok:
            bad.append(s)
    return 1 if bad else 0


def status() -> int:
    print("data root: %s" % DATA_ROOT)
    for s in EUROC:
        print("  EuRoC  %s  %s" % (s, "OK" if euroc_complete(s) else "-"))
    for r in TUMVI:
        prep = (tumvi_dir(r) / "orbslam_times.txt").is_file()
        print("  TUM-VI %s  %s" % (r, ("OK" if prep else "downloaded, run tumvi-prepare")
                                   if tumvi_complete(r) else "-"))
    kitti_check()
    return 0


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    cmd, args = argv[0], argv[1:]
    if cmd == "euroc":
        return fetch_euroc(args or list(EUROC))
    if cmd == "tumvi":
        return fetch_tumvi(args or TUMVI)
    if cmd == "tumvi-prepare":
        for r in (args or TUMVI):
            if tumvi_complete(r):
                tumvi_prepare(r)
        return 0
    if cmd == "kitti-check":
        return kitti_check()
    if cmd == "status":
        return status()
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
