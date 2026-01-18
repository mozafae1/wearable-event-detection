"""Pinch-hold event detection from smartwatch IMU/PPG streams.

It:
1) loads IMU (accelerometer + gyroscope), PPG, and label streams from .npy files
2) preprocesses IMU signals with Butterworth high-pass filtering + moving average
3) finds candidate peaks
4) matches peaks to label rising edges (within a time tolerance)
5) optionally saves diagnostic plots

Data format
-----------
The code expects three NumPy files in a data directory:

* imu_*.npy   : shape (N, 7) -> [timestamp_ms, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
* ppg_*.npy   : shape (M, 3) -> [timestamp_ms, ppg_ir, ppg_red]
* label_*.npy : shape (K, 2) -> [timestamp_ms, label] where label is 0/1

By default, it uses the filename template used in the notebook:
    data/{stream}_d32dd_01.npy

Run
---
    python main.py --data-dir data --subject d32dd --trial 01 --plots

The script prints basic detection metrics (recall and timing error) and (optionally)
writes plots to an output directory.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks


Array = np.ndarray


@dataclass(frozen=True)
class Streams:
    imu_t_s: Array  # (N,)
    acc: Array  # (N, 3)
    gyro: Array  # (N, 3)
    ppg_t_s: Array  # (M,)
    ppg: Array  # (M, 2)
    lbl_t_s: Array  # (K,)
    lbl: Array  # (K,)


@dataclass(frozen=True)
class PreprocessConfig:
    fs_imu_hz: float = 960.0
    # High-pass cutoffs (Hz) used in the notebook.
    hp_acc_hz: float = 5.0
    hp_gyro_hz: float = 5.0
    # Moving-average smoothing on magnitude.
    smooth_window_samples: int = 5
    # Peak detection (scipy.signal.find_peaks)
    peak_min_height_acc: float = 0.5
    peak_min_height_gyro: float = 0.04
    peak_min_distance_s: float = 0.04
    # Matching tolerance for aligning peaks to label edges.
    match_tolerance_s: float = 0.02
    # Filter order
    butter_order: int = 4


def _butterworth(signal_1d: Array, cutoff_hz: float, fs_hz: float, order: int, *, btype: str) -> Array:
    """Zero-phase Butterworth filtering for 1D signals."""
    if cutoff_hz <= 0 or cutoff_hz >= 0.5 * fs_hz:
        raise ValueError(
            f"Invalid cutoff_hz={cutoff_hz}. Must satisfy 0 < cutoff < Nyquist ({0.5*fs_hz})."
        )
    wn = cutoff_hz / (0.5 * fs_hz)
    b, a = butter(order, wn, btype=btype, analog=False)
    return filtfilt(b, a, signal_1d)


def highpass(signal_1d: Array, cutoff_hz: float, fs_hz: float, order: int = 4) -> Array:
    return _butterworth(signal_1d, cutoff_hz, fs_hz, order, btype="highpass")


def moving_average(x: Array, window: int) -> Array:
    if window <= 1:
        return x
    pad = window // 2
    kernel = np.ones(window, dtype=float) / float(window)
    x_pad = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(x_pad, kernel, mode="valid")


def load_streams(data_dir: Path, subject: str, trial: str) -> Streams:
    """Load IMU, PPG, and label streams from .npy files."""
    # The notebook used: data/{stream}_d32dd_01.npy
    # We generalize this to: data/{stream}_{subject}_{trial}.npy
    template = str(data_dir / "{stream}_{subject}_{trial}.npy")

    def _load(stream: str) -> Array:
        f = Path(template.format(stream=stream, subject=subject, trial=trial))
        if not f.exists():
            raise FileNotFoundError(
                f"Missing '{stream}' file. Expected: {f}. "
                "Use --data-dir/--subject/--trial to point to the correct files."
            )
        arr = np.load(f)
        if not isinstance(arr, np.ndarray) or arr.ndim != 2:
            raise ValueError(f"Unexpected array for {stream}: expected 2D numpy array, got {type(arr)} shape={getattr(arr, 'shape', None)}")
        return arr

    imu = _load("imu")
    ppg = _load("ppg")
    lbl = _load("label")

    if imu.shape[1] < 7:
        raise ValueError(f"IMU array must have at least 7 columns; got shape {imu.shape}")
    if ppg.shape[1] < 3:
        raise ValueError(f"PPG array must have at least 3 columns; got shape {ppg.shape}")
    if lbl.shape[1] < 2:
        raise ValueError(f"Label array must have at least 2 columns; got shape {lbl.shape}")

    imu_t_s = imu[:, 0].astype(float) / 1e3
    acc = imu[:, 1:4].astype(float)
    gyro = imu[:, 4:7].astype(float)

    ppg_t_s = ppg[:, 0].astype(float) / 1e3
    ppg_xy = ppg[:, 1:3].astype(float)

    lbl_t_s = lbl[:, 0].astype(float) / 1e3
    lbl_y = lbl[:, 1].astype(int)
    lbl_y = np.clip(lbl_y, 0, 1)

    return Streams(
        imu_t_s=imu_t_s,
        acc=acc,
        gyro=gyro,
        ppg_t_s=ppg_t_s,
        ppg=ppg_xy,
        lbl_t_s=lbl_t_s,
        lbl=lbl_y,
    )


def rising_edges(t: Array, y01: Array) -> Array:
    """Return timestamps (seconds) of rising edges in a 0/1 label stream."""
    if len(y01) == 0:
        return np.array([], dtype=float)
    y01 = np.asarray(y01).astype(int)
    idx = np.where((y01[1:] == 1) & (y01[:-1] == 0))[0] + 1
    return t[idx]


def imu_magnitude_features(streams: Streams, cfg: PreprocessConfig) -> Tuple[Array, Array]:
    """Compute smoothed high-pass IMU magnitudes for acc and gyro."""
    # High-pass filter each axis, then take vector magnitude and smooth.
    acc_hp = np.column_stack(
        [highpass(streams.acc[:, i], cfg.hp_acc_hz, cfg.fs_imu_hz, cfg.butter_order) for i in range(3)]
    )
    gyro_hp = np.column_stack(
        [highpass(streams.gyro[:, i], cfg.hp_gyro_hz, cfg.fs_imu_hz, cfg.butter_order) for i in range(3)]
    )
    acc_mag = np.linalg.norm(acc_hp, axis=1)
    gyro_mag = np.linalg.norm(gyro_hp, axis=1)

    acc_mag_s = moving_average(acc_mag, cfg.smooth_window_samples)
    gyro_mag_s = moving_average(gyro_mag, cfg.smooth_window_samples)
    return acc_mag_s, gyro_mag_s


def detect_peaks(t: Array, x: Array, *, min_height: float, min_distance_s: float, fs_hz: float) -> Array:
    """Peak detection returning peak timestamps."""
    distance_samples = max(1, int(round(min_distance_s * fs_hz)))
    peaks, _props = find_peaks(x, height=min_height, distance=distance_samples)
    return t[peaks]


def match_events(
    edge_times_s: Array,
    candidate_times_s: Array,
    candidate_scores: Optional[Array] = None,
    tol_s: float = 0.02,
) -> Tuple[Array, Array]:
    """Match each label edge to at most one candidate peak.

    Strategy:
      For each edge time, find all candidate peaks within +/- tol.
      If candidate_scores is provided, select the candidate with the largest score.
      Otherwise select the nearest in time.

    Returns:
      matched_times: shape (E,) with matched candidate timestamp or NaN if none
      matched_idx: shape (E,) index into candidate_times_s or -1 if none
    """
    edge_times_s = np.asarray(edge_times_s, dtype=float)
    candidate_times_s = np.asarray(candidate_times_s, dtype=float)
    if candidate_scores is not None:
        candidate_scores = np.asarray(candidate_scores, dtype=float)
        if len(candidate_scores) != len(candidate_times_s):
            raise ValueError("candidate_scores must be the same length as candidate_times_s")

    matched_times = np.full(edge_times_s.shape, np.nan, dtype=float)
    matched_idx = np.full(edge_times_s.shape, -1, dtype=int)
    if len(edge_times_s) == 0 or len(candidate_times_s) == 0:
        return matched_times, matched_idx

    # Pre-sort candidates to allow windowed lookup.
    order = np.argsort(candidate_times_s)
    cand_t = candidate_times_s[order]
    cand_s = candidate_scores[order] if candidate_scores is not None else None

    for i, t0 in enumerate(edge_times_s):
        lo = np.searchsorted(cand_t, t0 - tol_s, side="left")
        hi = np.searchsorted(cand_t, t0 + tol_s, side="right")
        if lo >= hi:
            continue
        window_t = cand_t[lo:hi]
        if cand_s is None:
            j = int(np.argmin(np.abs(window_t - t0)))
            sel = lo + j
        else:
            window_s = cand_s[lo:hi]
            j = int(np.argmax(window_s))
            sel = lo + j
        matched_times[i] = cand_t[sel]
        matched_idx[i] = int(order[sel])

    return matched_times, matched_idx


def run_pipeline(streams: Streams, cfg: PreprocessConfig) -> Dict[str, Array | float | int]:
    acc_mag, gyro_mag = imu_magnitude_features(streams, cfg)

    acc_peaks_t = detect_peaks(
        streams.imu_t_s,
        acc_mag,
        min_height=cfg.peak_min_height_acc,
        min_distance_s=cfg.peak_min_distance_s,
        fs_hz=cfg.fs_imu_hz,
    )
    gyro_peaks_t = detect_peaks(
        streams.imu_t_s,
        gyro_mag,
        min_height=cfg.peak_min_height_gyro,
        min_distance_s=cfg.peak_min_distance_s,
        fs_hz=cfg.fs_imu_hz,
    )

    edges_t = rising_edges(streams.lbl_t_s, streams.lbl)

    # Score peaks by magnitude at nearest IMU sample.
    def score_peaks(peak_times: Array, mag: Array) -> Array:
        if len(peak_times) == 0:
            return np.array([], dtype=float)
        t_imu = streams.imu_t_s
        idx = np.searchsorted(t_imu, peak_times)
        idx = np.clip(idx, 0, len(t_imu) - 1)
        idx2 = np.clip(idx - 1, 0, len(t_imu) - 1)
        better = np.abs(t_imu[idx2] - peak_times) < np.abs(t_imu[idx] - peak_times)
        idx = np.where(better, idx2, idx)
        return mag[idx]

    acc_scores = score_peaks(acc_peaks_t, acc_mag)
    gyro_scores = score_peaks(gyro_peaks_t, gyro_mag)

    acc_match_t, _ = match_events(edges_t, acc_peaks_t, acc_scores, tol_s=cfg.match_tolerance_s)
    gyro_match_t, _ = match_events(edges_t, gyro_peaks_t, gyro_scores, tol_s=cfg.match_tolerance_s)

    # Fuse: if both matched, take the earlier one (less latency) unless one is NaN.
    fused = np.full_like(edges_t, np.nan, dtype=float)
    for i in range(len(edges_t)):
        a = acc_match_t[i]
        g = gyro_match_t[i]
        if np.isnan(a) and np.isnan(g):
            continue
        if np.isnan(a):
            fused[i] = g
        elif np.isnan(g):
            fused[i] = a
        else:
            fused[i] = a if a <= g else g

    detected_mask = ~np.isnan(fused)
    n_edges = int(len(edges_t))
    n_detected = int(np.sum(detected_mask))
    recall = float(n_detected / n_edges) if n_edges else float("nan")
    mae = float(np.mean(np.abs(fused[detected_mask] - edges_t[detected_mask]))) if n_detected else float("nan")
    rmse = (
        float(np.sqrt(np.mean((fused[detected_mask] - edges_t[detected_mask]) ** 2)))
        if n_detected
        else float("nan")
    )

    return {
        "edges_t": edges_t,
        "acc_mag": acc_mag,
        "gyro_mag": gyro_mag,
        "acc_peaks_t": acc_peaks_t,
        "gyro_peaks_t": gyro_peaks_t,
        "fused_t": fused,
        "recall": recall,
        "mae_s": mae,
        "rmse_s": rmse,
        "n_edges": n_edges,
        "n_detected": n_detected,
    }


def plot_diagnostics(streams: Streams, result: Dict[str, Array | float | int], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    edges_t = np.asarray(result["edges_t"], dtype=float)
    fused_t = np.asarray(result["fused_t"], dtype=float)
    acc_mag = np.asarray(result["acc_mag"], dtype=float)
    gyro_mag = np.asarray(result["gyro_mag"], dtype=float)

    # Overview plot
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(12, 8))
    axes[0].plot(streams.imu_t_s, streams.acc, linewidth=0.8)
    axes[0].set_ylabel("acc")
    axes[1].plot(streams.imu_t_s, streams.gyro, linewidth=0.8)
    axes[1].set_ylabel("gyro")
    axes[2].plot(streams.ppg_t_s, streams.ppg, linewidth=0.8)
    axes[2].set_ylabel("ppg")
    axes[2].set_xlabel("time (s)")

    for ax in axes:
        ax2 = ax.twinx()
        ax2.fill_between(streams.lbl_t_s, streams.lbl, step="post", alpha=0.2)
        ax2.set_yticks([])
        ax.spines[["top", "right"]].set_visible(False)
        ax2.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(outdir / "signals_overview.png", dpi=150)
    plt.close(fig)

    # Magnitudes + detections
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 6))
    axes[0].plot(streams.imu_t_s, acc_mag, linewidth=1.0)
    axes[0].set_ylabel("acc |hp| (smoothed)")
    axes[1].plot(streams.imu_t_s, gyro_mag, linewidth=1.0)
    axes[1].set_ylabel("gyro |hp| (smoothed)")
    axes[1].set_xlabel("time (s)")

    for ax in axes:
        ax2 = ax.twinx()
        ax2.fill_between(streams.lbl_t_s, streams.lbl, step="post", alpha=0.2)
        ax2.set_yticks([])
        ax.spines[["top", "right"]].set_visible(False)
        ax2.spines[["top", "right"]].set_visible(False)

        # Mark label edges and fused detections.
        if len(edges_t):
            ax.vlines(edges_t, ymin=np.min(ax.get_lines()[0].get_ydata()), ymax=np.max(ax.get_lines()[0].get_ydata()), linewidth=0.5)
        det = fused_t[~np.isnan(fused_t)]
        if len(det):
            ax.vlines(det, ymin=np.min(ax.get_lines()[0].get_ydata()), ymax=np.max(ax.get_lines()[0].get_ydata()), linewidth=1.0)

    fig.tight_layout()
    fig.savefig(outdir / "magnitudes_and_events.png", dpi=150)
    plt.close(fig)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory containing .npy files")
    p.add_argument("--subject", type=str, default="d32dd", help="Subject identifier in filename")
    p.add_argument("--trial", type=str, default="01", help="Trial identifier in filename")
    p.add_argument("--plots", action="store_true", help="Save diagnostic plots")
    p.add_argument("--outdir", type=Path, default=Path("outputs"), help="Output directory")

    # Common tuning knobs
    p.add_argument("--hp-acc", type=float, default=PreprocessConfig.hp_acc_hz)
    p.add_argument("--hp-gyro", type=float, default=PreprocessConfig.hp_gyro_hz)
    p.add_argument("--smooth", type=int, default=PreprocessConfig.smooth_window_samples)
    p.add_argument("--tol", type=float, default=PreprocessConfig.match_tolerance_s)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    cfg = PreprocessConfig(
        hp_acc_hz=float(args.hp_acc),
        hp_gyro_hz=float(args.hp_gyro),
        smooth_window_samples=int(args.smooth),
        match_tolerance_s=float(args.tol),
    )

    streams = load_streams(args.data_dir, args.subject, args.trial)
    result = run_pipeline(streams, cfg)

    print("\nDetection summary")
    print("-----------------")
    print(f"Label rising edges: {result['n_edges']}")
    print(f"Detected edges:     {result['n_detected']}")
    print(f"Recall:            {result['recall']:.3f}")
    if not math.isnan(float(result["mae_s"])):
        print(f"MAE (s):           {result['mae_s']:.4f}")
        print(f"RMSE (s):          {result['rmse_s']:.4f}")

    if args.plots:
        plot_diagnostics(streams, result, args.outdir)
        print(f"\nSaved plots to: {args.outdir.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
