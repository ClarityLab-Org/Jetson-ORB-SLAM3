# Reference trajectory check

Reference: `/home/shiv/datasets/ext_work/husky_20261023/fused/fused_poses.txt` (GTSAM factor graph: KISS-ICP lidar odometry + lidar loop closures + IMU preintegration + wheel odometry + IMU roll/pitch).

| Fusion input | Value |
|---|---|
| imu_factors | True |
| wheel_factors | 3550 |
| attitude_factors | 0 |
| loop_factors | 2 |
| magnetometer | excluded: |B| varies 28% over the run |
| validation_only | /platform/odom/filtered, /tf odom->base_link, /imu/rpy, /platform/joint_states |
| camera_topics_read | none |
| loop_fitness_after_fusion_median | 0.4118608914622038 |
| loop_rmse_after_fusion_median_m | 0.10641462580354231 |
| path_length_m | 65.61476224843545 |
| end_to_start_m | 3.696904579733415 |
| z_range_m | 0.34161767739800086 |
| fused_vs_frontend_max_shift_m | 2.4739957233176835 |
| imu_driver_orientation | driver orientation vs accelerometer gravity at rest: 84.5 deg -> EXCLUDED (inconsistent convention) |
| extrinsic_checks | {'lidar_yaw': {'lidar_travel_m': 2.832378545930508, 'wheel_travel_m': 3.000105545206807, 'yaw_correction_deg': 179.6105677901836, 'scans_used': 181, 'applied': True}, 'imu_axes': {'yaw_rate_corr': 0.9908049123859891, 'fwd_acc_corr': 0.06390492289527984}} |
| acc_rest_norm | 9.5781374900387 |
| scan_time_minus_header_ms_median | 49.902839 |

Reference path length 65.6 m, end-to-start 3.70 m.

Distance of each alternative to the reference after SE(3) alignment (at the reference times):

| Trajectory | Poses | RMS [m] | Max [m] | Path [m] | End-start [m] |
|---|---|---|---|---|---|
| lidar front end (fusion input, no graph) | 3552 | 0.420 | 1.219 | 68.9 | 4.61 |
| KISS-SLAM (lidar only, own loop closure) | 3552 | 0.444 | 1.208 | 69.6 | 4.70 |
| KISS-ICP (lidar only, no loop closure) | 3552 | 0.445 | 1.283 | 70.3 | 4.70 |
| /platform/odom (wheels) | 3551 | 0.537 | 0.645 | 68.5 | 3.85 |
| /platform/odom/filtered (robot EKF: wheels + IMU) | 3552 | 0.537 | 0.645 | 68.5 | 3.85 |
| fusion ablation: no IMU | 3552 | 0.057 | 0.155 | 67.0 | 3.69 |
| fusion ablation: no wheels | 3552 | 0.033 | 0.088 | 65.7 | 3.72 |
| fusion ablation: no loops | 3552 | 0.112 | 0.316 | 65.6 | 4.01 |

Reading the table:
- Lidar-only estimates (front end, KISS-ICP, KISS-SLAM) are not independent of the reference (they share
  the lidar). Agreement shows the fusion did not distort the lidar geometry; the difference to the front
  end is what the IMU, wheels and loop closures corrected.
- The robot's own odometry is independent of the lidar; it drifts, so expect decimetres to metres.
- The ablation rows show how sensitive the reference is to each input. Their RMS is a practical
  uncertainty bound for the reference: ORB-SLAM3 differences smaller than it are not resolvable.
