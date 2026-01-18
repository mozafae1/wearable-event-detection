# Pinch-Hold Detection from Smartwatch IMU/PPG

Classical signal-processing baseline for detecting **pinch-hold events** using smartwatch **IMU** (accelerometer + gyroscope) and **PPG** measurements.

This repository contains:

* A cleaned-up, executable script (`main.py`) derived from the original notebook.
* The original analysis notebook (`Report_Wearable_Event_Detection.ipynb`) for full context and exploration.

The pipeline is intentionally lightweight (NumPy/SciPy/Matplotlib) and focuses on the type of filtering + peak-picking approach you would typically implement as a first baseline in a wearable-sensing workflow.

## Method overview

1. **Load** time-stamped IMU, PPG, and binary label streams from `.npy` files.
2. **Preprocess IMU**
   * Per-axis **Butterworth high-pass filtering** (zero-phase via `filtfilt`).
   * Convert to a **vector magnitude** (‖x,y,z‖).
   * Apply a small **moving-average** smoother.
3. **Detect candidate events** via `scipy.signal.find_peaks` with a minimum height and minimum distance constraint.
4. **Match candidates** to label-stream rising edges within a configurable time tolerance.
5. **Report metrics** (recall and timing error) and optionally save diagnostic plots.

Notes:
* PPG is loaded and visualized for completeness, but the baseline event detector uses IMU features.
* The goal of this repo is a clean, reproducible baseline that is easy to tune and extend (e.g., feature engineering, sensor fusion logic, ML classifiers).

## Data format

Place three NumPy arrays in `data/` (or any directory you pass via `--data-dir`):

* `imu_<subject>_<trial>.npy` shape `(N, 7)`:
  `timestamp_ms, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z`
* `ppg_<subject>_<trial>.npy` shape `(M, 3)`:
  `timestamp_ms, ppg_ir, ppg_red`
* `label_<subject>_<trial>.npy` shape `(K, 2)`:
  `timestamp_ms, label` where `label ∈ {0,1}`

The default IDs match the original notebook naming convention:

* subject: `d32dd`
* trial: `01`

## Quickstart

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the detector:

```bash
python main.py --data-dir data --subject d32dd --trial 01 --plots
```

Outputs:
* Terminal summary: number of label edges, detected edges, recall, MAE/RMSE (seconds).
* If `--plots` is provided, plots are written to `outputs/` (configurable via `--outdir`).

## Tuning

The script exposes common knobs as CLI flags:

* `--hp-acc`, `--hp-gyro`: high-pass cutoffs (Hz)
* `--smooth`: moving-average window length (samples)
* `--tol`: matching tolerance (seconds)

If you want to change peak-picking parameters (height, minimum distance), update `PreprocessConfig` in `main.py`.

## Repository structure

```text
.
├── main.py                      # executable detector pipeline
├── Report_Wearable_Event_Detection.ipynb  # original notebook / report
├── requirements.txt
└── data/                         # (not committed) your .npy files
```

## License

Add a license file if you plan to distribute or reuse this code outside personal projects.
