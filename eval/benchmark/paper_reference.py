"""Numbers reported by the paper, for side-by-side comparison in the report.

Source: R. Roy, A. A. K. Yadav, H. Jain, "Jetson-ORB-SLAM3: Accuracy-Preserving GPU Implementation
for Edge Computing Devices", arXiv:2608.17874v1 (the PDF in the repository root).
Units as in the paper: ATE in cm (KITTI ATE in m), t_rel in %, r_rel in deg/100 m, times in ms.
"""

# Table 1 -- EuRoC stereo-inertial ATE (RMSE, cm, SE(3)), Orin Nano, median of 5 runs.
# columns: Jetson-SLAM (stereo, 1 run), published ORB-SLAM3 (desktop), ours GPU, ours CPU, |delta|
TABLE1 = {
    "MH01": (3.30, 3.50, 4.09, 3.60, 0.49),
    "MH02": (4.73, 2.90, 3.45, 3.46, 0.01),
    "MH03": (11.14, 2.50, 2.84, 2.76, 0.08),
    "MH04": (48.57, 3.10, 4.57, 4.75, 0.18),
    "MH05": (23.91, 2.80, 5.96, 6.26, 0.30),
    "V101": (8.81, 3.50, 3.74, 3.75, 0.01),
    "V102": (6.13, 1.10, 1.51, 1.45, 0.06),
    "V103": (6.98, 2.10, 2.52, 2.38, 0.14),
    "V201": (5.99, 2.70, 3.59, 3.34, 0.25),
    "V202": (15.36, 1.60, 1.31, 1.40, 0.09),
    "V203": (14.49, 1.40, 2.89, 4.24, 1.35),
}
TABLE1_MEAN = (13.58, 2.47, 3.32, 3.40, 0.27)

# Table 2 -- mean EuRoC ATE over the 11 sequences (cm): (SE(3), scaled)
TABLE2 = {
    "Orin Nano, GPU (ours)": (3.32, 2.75),
    "Orin Nano, CPU (stock)": (3.40, 2.91),
    "Desktop, GPU (ours)": (3.39, 2.80),
    "Desktop, CPU (stock)": (3.30, 2.73),
    "Published ORB-SLAM3": (2.47, None),
}

# Table 3 -- TUM-VI rooms, stereo-inertial, single run, SE(3) ATE (cm):
# published ORB-SLAM3, ours GPU, ours CPU, |delta|
TABLE3 = {
    "room1": (0.8, 0.98, 0.78, 0.20),
    "room2": (1.2, 1.08, 0.83, 0.25),
    "room3": (1.1, 0.84, 0.59, 0.25),
    "room4": (0.8, 0.67, 0.76, 0.09),
    "room5": (1.0, 0.80, 0.74, 0.06),
    "room6": (0.6, 0.86, 0.62, 0.24),
}
TABLE3_MEAN = (0.92, 0.87, 0.72, 0.18)

# Table 4 -- KITTI stereo, single run:
# t_rel % (ORB-SLAM2 [15], GPU, CPU, |delta|), r_rel deg/100m (ours, [15]), ATE m (ours, [15])
TABLE4 = {
    "00": (0.70, 0.72, 0.68, 0.04, 0.28, 0.25, 1.31, 1.3),
    "01": (1.39, 1.64, 1.65, 0.01, 0.28, 0.21, 13.15, 10.4),
    "02": (0.76, 0.77, 0.75, 0.02, 0.26, 0.23, 6.89, 5.7),
    "03": (0.71, 0.93, 0.94, 0.01, 0.20, 0.18, 1.31, 0.6),
    "04": (0.48, 0.56, 0.53, 0.03, 0.18, 0.13, 0.25, 0.2),
    "05": (0.40, 0.58, 0.41, 0.16, 0.19, 0.16, 1.30, 0.8),
    "06": (0.51, 0.51, 0.55, 0.04, 0.19, 0.15, 0.75, 0.8),
    "07": (0.50, 0.48, 0.47, 0.01, 0.29, 0.28, 0.48, 0.5),
    "08": (1.05, 1.02, 1.01, 0.01, 0.30, 0.32, 3.35, 3.6),
    "09": (0.87, 0.90, 0.93, 0.04, 0.28, 0.27, 1.69, 3.2),
    "10": (0.60, 0.66, 0.64, 0.02, 0.30, 0.27, 1.14, 1.0),
}
TABLE4_MEAN = (0.73, 0.80, 0.78, 0.04, 0.25, 0.22, 2.87, 2.55)

# Table 5 -- cross-dataset t_rel (%) under KITTI's estimator: (windows, GPU, CPU, mean |delta|)
TABLE5 = {
    "TUM-VI": ("5-40 m", 0.134, 0.117, 0.022),
    "EuRoC": ("5-40 m", 0.723, 0.725, 0.016),
    "KITTI": ("100-800 m", 0.800, 0.780, 0.036),
}

# Table 6 -- per-stage timing, instrumented build: (mono-inertial V101, stereo-inertial MH01)
TABLE6 = {
    "ORB extraction (ms)": (13.7, 38.9),
    "Total tracking (ms/frame)": (35.4, 76.6),
    "Throughput (FPS)": (28.3, 13.1),
}
TABLE6_UNINSTRUMENTED_FPS = (28.7, 13.5)

# Table 7 -- throughput (FPS), single run: (mono base, mono pipe, stereo base, stereo pipe)
TABLE7 = {
    "MH01": (25.6, 30.7, 13.5, 26.2),
    "MH02": (26.4, 26.3, 13.8, 27.3),
    "MH03": (31.0, 30.5, 13.8, 27.6),
    "MH04": (33.5, 32.5, 14.7, 28.7),
    "MH05": (32.4, 30.5, 14.6, 28.5),
    "V101": (28.7, 28.7, 13.6, 25.4),
    "V102": (34.6, 34.2, 14.5, 28.2),
    "V103": (35.2, 38.1, 15.6, 30.1),
    "V201": (33.2, 34.4, 14.4, 27.7),
    "V202": (34.1, 34.0, 14.1, 26.8),
    "V203": (37.6, 38.3, 15.6, 31.6),
}
TABLE7_MEAN = (32.0, 32.6, 14.4, 28.0)

# Table 8 -- mean tracking time (ms), GPU vs CPU reference: (resolution, GPU, CPU)
TABLE8 = {
    "KITTI (stereo)": ("1241x376", 98.8, 111.7),
    "EuRoC (stereo-inertial)": ("752x480", 69.7, 62.1),
    "TUM-VI (stereo-inertial)": ("512x512", 78.4, 78.3),
}

# Table 9 -- CosPlace ResNet-50 latency per query (ms)
TABLE9 = {"ONNX-Runtime CUDA/TRT EP": "fails to initialize", "ONNX-Runtime CPU": "~396",
          "TensorRT FP16 GPU": "2.2"}

# Table 10 -- loop-closure ablation, median ATE (cm) of 5 runs: (DBoW2+CNN, DBoW2 only, no LC)
TABLE10 = {
    "MH01": (3.91, 4.21, 3.72),
    "MH02": (3.18, 3.24, 3.61),
    "MH03": (2.89, 2.74, 2.82),
    "MH04": (4.64, 4.75, 4.63),
    "MH05": (5.41, 5.53, 5.94),
    "V101": (3.61, 3.64, 3.77),
    "V102": (1.49, 1.37, 1.46),
    "V103": (2.57, 2.34, 2.58),
    "V201": (3.54, 3.00, 3.59),
    "V202": (1.29, 1.29, 2.61),
}
TABLE10_MEAN_NO_V203 = (3.49, 3.41, 3.60)
TABLE10_V203 = (5.87, 5.35, 4.92)

# Sec. 4.1 -- board power, stereo-inertial MH01, tegrastats @ 250 ms, VDD_IN (W)
POWER = {"mean": 6.3, "peak": 6.9, "idle": 4.7}

# Sec. 3.1.5 -- feature-level equivalence over 205 EuRoC frames
FEATURES = {"frames": 205, "kp_gpu": 247368, "kp_cpu": 247380, "exact_kp_pct": 94.7,
            "mean_hamming_bits": 0.25, "bit_agreement_pct": 99.90, "identical_desc_pct": 78.8}
