#!/usr/bin/env bash
#
# Everything the Husky evaluation needs BEFORE the ORB-SLAM3 runs (plan/husky_stage1_runbook.md, step 2):
# inventory, IR stereo extraction, the fused non-camera reference, its cross-checks, and gt.csv.
#
#   eval/processing/prepare_husky.sh [BAG_DIR] [SEQ_NAME]
#       BAG_DIR   default ~/datasets/external_raw/husky_20261023 (metadata.yaml + .db3)
#       SEQ_NAME  default husky_20261023  -> images + gt.csv in ~/datasets/ext/SEQ_NAME
#
# Every step writes into its own folder and is skipped if its output already exists, so the script
# can be re-run after an interruption (delete a folder to redo that step). The bag is only read.
# Log: ~/datasets/ext/SEQ_NAME.prepare.log
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BAG="${1:-$HOME/datasets/external_raw/husky_20261023}"
NAME="${2:-husky_20261023}"
SEQ="$HOME/datasets/ext/$NAME"
WORK="$HOME/datasets/ext_work/$NAME"            # reference + cross-checks (not read by run_matrix)
PY="${PY:-$HOME/venvs/ext/bin/python}"
X="$ROOT/eval/processing"
mkdir -p "$SEQ" "$WORK"
exec > >(tee -a "$HOME/datasets/ext/$NAME.prepare.log") 2>&1

step() { echo; echo "==> [$(date +%H:%M:%S)] $*"; }
[ -f "$BAG/metadata.yaml" ] || { echo "ERROR: no metadata.yaml in $BAG"; exit 1; }
"$PY" -c "import rosbags, kiss_icp, gtsam, scipy" || { echo "ERROR: run eval/processing/setup_venv.sh first"; exit 1; }

step "1/7 inventory"
[ -f "$WORK/inventory.txt" ] || "$PY" -I "$X/inspect_bag.py" "$BAG" --samples "$WORK/samples" | tee "$WORK/inventory.txt"

step "2/7 IR stereo -> $SEQ (mav0/, times.txt, settings.yaml)"
[ -f "$SEQ/settings.yaml" ] || "$PY" -I "$X/bag_to_euroc.py" "$BAG" "$SEQ"

step "3/7 robot odometry (validation only)"
[ -f "$WORK/odom/platform_odom_filtered.txt" ] || "$PY" -I "$X/bag_odom.py" "$BAG" "$WORK/odom"

step "4/7 fused reference: lidar + loops + IMU + wheels + IMU roll/pitch  (the long step)"
[ -f "$WORK/fused/fused_poses.txt" ] || "$PY" -I "$X/fuse_reference.py" "$BAG" "$WORK/fused"

step "5/7 sensitivity: the same fusion without IMU / without wheels / without loops"
for v in no-imu no-wheel no-loops; do
  [ -f "$WORK/fused_$v/fused_poses.txt" ] || "$PY" -I "$X/fuse_reference.py" "$BAG" "$WORK/fused_$v" --$v
done

step "6/7 lidar-only cross-checks: KISS-ICP, KISS-SLAM"
[ -f "$WORK/kiss_icp/lidar_poses.txt" ] || "$PY" -I "$X/lidar_odometry.py" "$BAG" "$WORK/kiss_icp"
if "$PY" -c "import kiss_slam" 2>/dev/null; then
  [ -f "$WORK/kiss_slam/lidar_poses.txt" ] || "$PY" -I "$X/lidar_slam.py" "$BAG" "$WORK/kiss_slam" || echo "  (KISS-SLAM failed; cross-check skipped)"
else
  echo "  kiss-slam not installed: cross-check skipped"
fi
KS=(); [ -f "$WORK/kiss_slam/lidar_poses.txt" ] && KS=(--kiss-slam "$WORK/kiss_slam")
"$PY" -I "$X/reference_check.py" "$WORK/fused" "$SEQ" "${KS[@]}" --kiss-icp "$WORK/kiss_icp" --odom "$WORK/odom" \
    --ablations "$WORK/fused_no-imu" "$WORK/fused_no-wheel" "$WORK/fused_no-loops"

step "7/7 gt.csv: reference -> left IR optical frame at every image time"
"$PY" -I "$X/make_gt.py" "$SEQ" "$WORK/fused/fused_poses.txt"
cp "$WORK/fused/fusion_info.json" "$SEQ/reference_info.json"

echo
echo "Done. Sequence for run_matrix: $SEQ"
echo "      Reference + checks:       $WORK (read fused/fusion_report.txt and fused/reference_check.md)"
