# Husky reference trajectory ("ground truth") for `husky_20261023`

This folder holds the **best estimate of where the Husky actually went**, built on the PC from every
**non-camera** sensor in the other team's bag. It is what our Jetson ORB-SLAM3 runs (camera only)
get scored against.

- Code: [`eval/processing/`](../../../processing)
- How it is used: [eval/README.md](../../../README.md)
- Built: 2026-10-23, from `husky_20261023_062755_0.db3` (SHA-256 `2877dab4…f851bcd`)

## What to use

| File | What it is |
|---|---|
| **`sequence/gt.csv`** | **The ground truth.** It is the pose of the left IR camera (`camera_infra1_optical_frame`) at **every image timestamp**: 5324 of 5326 images, with the first and last one outside the lidar's time span. The format is EuRoC (`#timestamp[ns],p_x,p_y,p_z,q_w,q_x,q_y,q_z`), read directly by `eval/benchmark/evaluate.py`. |
| `sequence/times.txt`, `settings.yaml`, `calib.json`, `tf_static.json`, `source_info.txt` | The sequence definition the ORB-SLAM3 runs use. `settings.yaml` holds their camera calibration with our R6 ORB settings. |
| `fusion/fused_poses.txt` | The same reference as `base_link` poses at the lidar's 10 Hz rate (`t_ns x y z qx qy qz qw`) |
| `fusion/fusion_report.txt`, `fusion_info.json` | The full log of how it was built, including every check and every correction |
| `fusion/reference_check.md`, `.png` | How it compares with every other non-camera estimate |
| `fusion/fusion_topview.png`, `fusion_timeseries.png`, `loops.csv`, `factors.txt`, `biases.txt` | Diagnostics |
| `crosschecks/` | Fusion without the IMU / wheels / loops, KISS-ICP, KISS-SLAM, and the robot's own odometry |
| `inventory.txt` | Read-only inventory of the bag |

The 1.3 GB of IR images are **not** in the repository. Regenerate them from the bag with
`eval/processing/bag_to_euroc.py`, then copy `sequence/gt.csv` into the sequence folder.

## How it was built

It is **one GTSAM factor graph** solved in batch (Levenberg–Marquardt), with one state per lidar scan
(3553 scans, 10 Hz). Each state holds the pose of `base_link`, its velocity and the IMU bias.

| Data in the bag | Role | Notes |
|---|---|---|
| 3D lidar (`/lidar_points`, Hesai, 16 rings, ±15°) | **KISS-ICP odometry** between consecutive scans (Huber-weighted), deskewed per point | each scan is predicted from the IMU gyro + wheels |
| 3D lidar | **2 loop closures** (end of the route against its start): a window of 5 keyframes matched by ICP to a submap of 9 keyframes, searched on the IMU- and wheel-corrected trajectory | Cauchy-weighted; fitness 0.41, residual RMSE 10.6 cm |
| IMU gyro + accelerometer (`/imu/data`, 83 Hz) | **IMU preintegration** (`CombinedImuFactor`, bias estimated) | noise from the first 2.5 s at rest, ×5 |
| Wheel odometry (`/platform/odom`) | between factors: forward and yaw loose, sideways tight (a skid-steer can't slide), no z, roll or pitch | **scale calibrated to the lidar: 0.956** |
| `/tf_static` | extrinsics between sensors | **checked against the data; one correction** (below) |
| Magnetometer (`/imu/mag`) | **excluded** | \|B\| varies 28%: an indoor-distorted field |
| IMU driver orientation (in `/imu/data`) | **excluded** | 84.5° away from the accelerometer's gravity direction at rest: a broken orientation convention in their driver |
| `/platform/odom/filtered`, `/tf` odom→base_link, `/imu/rpy`, `/platform/joint_states` | **validation only** | computed from the wheels and IMU already fused, so fusing them again would count the same measurements twice |
| **Camera** (IR, depth, colour) | **not used** | see "Why the camera is not used" |

### Problems found in the dataset and how they were handled

1. **The lidar's yaw in `/tf_static` is off by 179.6°.** The TF mounts the Hesai at −90° yaw. The
   data shows +90°: the robot drives along the lidar's −y axis, and the wheels confirm the direction
   of travel. Uncorrected, the lidar odometry said the robot drove backwards. The fusion measures the
   lidar's yaw from the data (lidar-only travel vs wheel travel over the first 3 m) and applies the
   correction; the TF translation is kept. The corrected `T_base_lidar` is in `fusion_info.json`.
   *Worth telling the other team.*
2. **The IMU driver's orientation quaternion is unusable**, as described above. The raw gyro and
   accelerometer are fine and are fully used: the IMU's yaw rate correlates 0.99 with the wheels.
3. **The wheel odometry over-reports distance by 4.4%** (scale 0.956, IQR 0.941–0.976 over 176
   two-second windows), which is typical of a skid-steer. It was calibrated against the lidar.
4. **The accelerometer reads \|g\| = 9.58 m/s²** instead of 9.81. The graph estimates it as an
   accelerometer bias (z ≈ −0.25 m/s²).
5. **The camera extrinsic was verified, not assumed.** The essential-matrix motion direction from the
   IR images matches the TF's forward-facing camera (median 11°, which is two-frame noise). That check
   read images only for verification; nothing from the camera entered the reference.

## How good is it

| Check | Result |
|---|---|
| Route | 65.6 m horizontal, a 21 × 16.5 m loop; it ends 3.70 m from its start; height range 0.34 m |
| **Removing one input** (RMS change of the whole trajectory) | without wheels **3.3 cm**, without the IMU **5.7 cm**, without loop closure **11.2 cm** |
| Loop residual before the loops were added (IMU + wheels + lidar) | 0.53 m / 1.8° accumulated over the loop (0.8%). The lidar front end alone needed 2.4 m / 7.4°. |
| Lidar-only odometry (KISS-ICP / KISS-SLAM) against the reference | 0.42–0.45 m RMS. Mostly **height drift** (+1.8 m, then −2.3 m) that the IMU removes; see `fusion_timeseries.png`. |
| Robot's own wheel and EKF odometry against the reference | 0.54 m RMS (mostly its 4.4% scale error) |
| Factor residuals after optimisation | small and balanced for every factor type (`factors.txt`) |

**Uncertainty estimate: about 10–25 cm.**
- **Lower bound:** removing the loop closure changes the trajectory by 11 cm RMS.
- **Conservative upper bound:** half the 0.53 m that the loop had to correct.
- **Height is the least certain axis:** the wheels don't measure z, and the lidar drifts in z. So
  `eval/processing/husky_report.py` also reports a **horizontal ATE** next to the 3D ATE.
- ORB-SLAM3 differences smaller than about 10–25 cm can't be resolved against this reference.

## Why the camera is not used

The reference has to be **independent** of what it scores. If the IR images went into the reference,
the reference would make the same mistakes ORB-SLAM3 makes in the same places: blank walls,
over-exposed frames, the same feature tracks. Those errors would cancel, and ORB-SLAM3 would look
better than it is.

The camera would also add little: the lidar measures geometry to centimetres out to 40 m, while a
stereo pair with a 5 cm baseline degrades fast beyond about 2 m. Benchmarks follow the same rule
(EuRoC: motion capture and laser tracker; TUM-VI: motion capture; KITTI: GPS/INS). The evaluated
sensor is never part of the ground truth.

## Rebuild

```bash
PYTHON=python3.12 eval/processing/setup_venv.sh   # once: ~/venvs/ext (kiss-icp has no Python 3.14 wheel)
eval/processing/prepare_husky.sh               # bag at ~/datasets/external_raw/husky_20261023
```
It takes ~3 min on the PC: about 45 s per fusion run, × 4 runs, plus the cross-checks.
