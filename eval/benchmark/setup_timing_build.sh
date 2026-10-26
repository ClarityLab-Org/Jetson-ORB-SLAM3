#!/usr/bin/env bash
#
# Build an instrumented copy of the repository for the per-stage timing table (paper Table 6).
# plan/paper_runbook.md, Part R3.
#
#   eval/benchmark/setup_timing_build.sh [DEST]        # default ../Jetson-ORB-SLAM3-timing
#
# ORB-SLAM3 records per-stage times only when compiled with REGISTER_TIMES (commented out in
# include/Settings.h). The define also changes class layouts (Frame.h, Tracking.h, System.h ...),
# so library and programs must all be built with it -- and a second build directory in this
# checkout would overwrite lib/ and Examples/ (CMakeLists.txt writes there). So this script makes
# an independent git worktree of the same commit and passes -DREGISTER_TIMES on the compiler
# command line. No source file is edited, in either checkout.
#
# Afterwards: python3 eval/benchmark/run_matrix.py instrumented --build-root <DEST>
# Takes ~45 min (the CUDA kernels dominate); JOBS=2 by default (more runs an 8 GB Orin out of memory).

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="${1:-$ROOT/../Jetson-ORB-SLAM3-timing}"
JOBS="${JOBS:-2}"

if [ ! -d "$DEST" ]; then
  echo "==> creating worktree $DEST at $(git -C "$ROOT" rev-parse --short HEAD)"
  git -C "$ROOT" worktree add --detach "$DEST" HEAD
fi
DEST="$(cd "$DEST" && pwd)"
cd "$DEST"

# The vocabulary is identical; reuse the unpacked copy instead of unpacking 140 MB again.
if [ ! -e Vocabulary/ORBvoc.txt ]; then
  if [ -f "$ROOT/Vocabulary/ORBvoc.txt" ]; then
    ln -s "$ROOT/Vocabulary/ORBvoc.txt" Vocabulary/ORBvoc.txt
  else
    tar -xzf Vocabulary/ORBvoc.txt.tar.gz -C Vocabulary
  fi
fi

# User-space Eigen/Pangolin under ~/local, as run_euroc.sh does.
if [ -d "$HOME/local/include" ]; then
  export CMAKE_PREFIX_PATH="$HOME/local${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
  export CPLUS_INCLUDE_PATH="$HOME/local/include/eigen3:$HOME/local/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
fi

for t in DBoW2 g2o Sophus; do
  echo "==> Thirdparty/$t"
  cmake -S "Thirdparty/$t" -B "Thirdparty/$t/build" -DCMAKE_BUILD_TYPE=Release >/dev/null
  cmake --build "Thirdparty/$t/build" -j "$JOBS"
done

echo "==> library + programs with REGISTER_TIMES (~45 min)"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-DREGISTER_TIMES"
cmake --build build -j "$JOBS"

# Verify the define took effect: PrintTimeStats only exists in an instrumented build.
# (no `grep -q`: it exits on the first match, nm gets SIGPIPE, and pipefail turns that into a failure)
if nm -DC lib/libORB_SLAM3.so | grep -c 'Tracking::PrintTimeStats' >/dev/null; then
  echo "==> OK: instrumented build ready in $DEST"
  echo "    next: python3 $ROOT/repro/run_matrix.py instrumented --build-root $DEST"
else
  echo "==> ERROR: lib/libORB_SLAM3.so has no Tracking::PrintTimeStats -- REGISTER_TIMES was not applied" >&2
  exit 1
fi
