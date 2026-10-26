#!/usr/bin/env bash
#
# Python environment for eval/processing/ (plan/husky_stage1_runbook.md, step 1).
#
#   eval/processing/setup_venv.sh [--light] [VENV]        # default ~/venvs/ext
#
#   --light   only what the Jetson needs (bag reading, image extraction, report): rosbags, numpy,
#             scipy, opencv-python-headless, matplotlib. The full set (+ kiss-icp, gtsam, kiss-slam)
#             builds the reference, which is done on the PC.
#
# Separate from the system Python so the repository's own scripts (numpy 1.21 on the Jetson) are
# untouched. Prints which optional packages failed, so the runbook's fallbacks can be chosen.
set -uo pipefail
LIGHT=0
if [ "${1:-}" = "--light" ]; then LIGHT=1; shift; fi
VENV="${1:-$HOME/venvs/ext}"
PY="${PYTHON:-python3}"

if [ -x "$VENV/bin/python" ] && ! "$VENV/bin/python" -m pip --version >/dev/null 2>&1; then
  echo "==> $VENV is broken (no pip; an earlier venv creation failed): recreating"
  rm -rf "$VENV"
fi
if [ ! -x "$VENV/bin/python" ]; then
  echo "==> creating $VENV with $($PY --version)"
  "$PY" -m venv "$VENV" || { rm -rf "$VENV"; echo "ERROR: python3 -m venv failed: sudo apt install python3.10-venv (Jetson) / python3-venv"; exit 1; }
fi
"$VENV/bin/pip" install --upgrade pip wheel >/dev/null

echo "==> required: rosbags numpy scipy opencv-python-headless matplotlib"
"$VENV/bin/pip" install rosbags numpy scipy opencv-python-headless matplotlib || { echo "ERROR: required packages failed"; exit 1; }

failed=()
OPTIONAL=(kiss-icp gtsam kiss-slam)
[ $LIGHT = 1 ] && OPTIONAL=()
for pkg in "${OPTIONAL[@]}"; do
  echo "==> $pkg"
  "$VENV/bin/pip" install "$pkg" || failed+=("$pkg")
done

echo
LIGHT=$LIGHT "$VENV/bin/python" - <<'EOF'
import importlib
import os
mods = ["rosbags", "numpy", "scipy", "cv2", "matplotlib"]
if os.environ.get("LIGHT") != "1":
    mods += ["kiss_icp", "gtsam", "kiss_slam"]
for m in mods:
    try:
        mod = importlib.import_module(m)
        print("  ok      %-10s %s" % (m, getattr(mod, "__version__", "")))
    except Exception as e:  # noqa: BLE001
        print("  MISSING %-10s %s" % (m, e))
EOF
if [ ${#failed[@]} -gt 0 ]; then
  echo
  echo "Not installed: ${failed[*]}"
  echo "  kiss-icp / gtsam are required for fuse_reference.py; kiss-slam only for the cross-check."
  echo "  See plan/husky_stage1_runbook.md, 'If a package does not install'."
fi
