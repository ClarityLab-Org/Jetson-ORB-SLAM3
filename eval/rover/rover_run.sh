#!/usr/bin/env bash
#
# One rover run with the Stage 1 rover driver (plan/stage1.md, 4.4 / 4.6).
#
#   eval/rover/rover_run.sh LABEL [driver options...]
#   eval/rover/rover_run.sh A1_base --no-viewer
#
# Creates ~/runs/<date-time>_LABEL, records the repo version and tegrastats alongside the
# driver's own files, runs the driver (press q to stop and save), then prints the summary.
# SETTINGS overrides the settings file (default Examples/Stereo/RealSense_D435.yaml).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LABEL="${1:?usage: eval/rover/rover_run.sh LABEL [driver options...]}"
shift
SETTINGS="${SETTINGS:-$ROOT/Examples/Stereo/RealSense_D435.yaml}"
BIN="$ROOT/Examples/Stereo/stereo_realsense_D435_rover"
OUT="$HOME/runs/$(date +%Y%m%d-%H%M%S)_$LABEL"

[ -x "$BIN" ] || { echo "missing $BIN -- build it first (plan/stage1.md, 4.4)" >&2; exit 1; }
[ -f "$SETTINGS" ] || { echo "missing $SETTINGS -- create it first (plan/stage1.md, Step 3)" >&2; exit 1; }
[ -f "$ROOT/Vocabulary/ORBvoc.txt" ] || { echo "missing Vocabulary/ORBvoc.txt -- run ./run_euroc.sh once to unpack it" >&2; exit 1; }

mkdir -p "$OUT"
{
  echo "git_commit: $(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "git_modified_tracked_files: $(git -C "$ROOT" status --porcelain --untracked-files=no 2>/dev/null | wc -l)"
  echo "l4t: $(head -1 /etc/nv_tegra_release 2>/dev/null || echo unknown)"
} > "$OUT/versions.txt"

# Same library search path the repo's own run_euroc.sh uses for a source build.
export LD_LIBRARY_PATH="$ROOT/lib:$ROOT/Thirdparty/DBoW2/lib:$ROOT/Thirdparty/g2o/lib:/usr/local/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

tegrastats --interval 1000 > "$OUT/tegrastats.log" 2>&1 &
TEGRA=$!
trap 'kill $TEGRA 2>/dev/null || true' EXIT

echo "==> run $LABEL -> $OUT   (press q to stop and save)"
# tee -i: Ctrl-C must reach only the driver (which saves before exiting), not kill the logger.
"$BIN" "$ROOT/Vocabulary/ORBvoc.txt" "$SETTINGS" --out "$OUT" "$@" 2>&1 | tee -i "$OUT/console.log"

echo
python3 "$ROOT/eval/rover/summarize_run.py" "$OUT" | tee "$OUT/summary.txt"
