"""Utilities for exploring beat-to-beat arterial pressure waveforms.

This module provides a simple interactive experiment that loads a text
file exported from a continuous blood pressure monitor, extracts the
`reBAP` channel, and attempts to automatically detect systolic,
diastolic, and mean arterial pressure (MAP) values for each beat.

Usage
-----
Run the module directly to open a file picker and visualise a recording::

    python Beat-to-beat_test.py [optional/path/to/file.txt]

The script will plot the waveform and overlay the detected beat
landmarks.  It also prints summary statistics for the detected beats and
flags potential artefacts using simple heuristics.

Command-line flags allow overriding the column separator used when
reading whitespace-heavy exports (``--separator``).
"""

from __future__ import annotations

import csv
import math
import re
import sys
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import tkinter as tk
    from tkinter import filedialog, ttk
except Exception:  # pragma: no cover - tkinter may be unavailable in some envs.
    tk = None
    filedialog = None
    ttk = None

try:
    import matplotlib
    if tk is not None:
        preferred_backend = "TkAgg"
    else:
        preferred_backend = "Agg"
    current_backend = matplotlib.get_backend().lower()
    if current_backend != preferred_backend.lower():
        try:
            matplotlib.use(preferred_backend, force=True)
        except Exception:
            if current_backend.startswith("qt"):
                matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    plt = None

try:
    import numpy as np
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
    raise SystemExit("numpy is required to run this script. Please install it and retry.") from exc

try:
    import pandas as pd
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
    raise SystemExit("pandas is required to run this script. Please install it and retry.") from exc

try:  # pragma: no cover - optional dependency for zero-phase filtering/PSD
    from scipy import signal as sp_signal
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    sp_signal = None


@dataclass
class Beat:
    """Container describing a detected beat."""

    systolic_time: float
    systolic_pressure: float
    diastolic_time: float
    diastolic_pressure: float
    notch_time: Optional[float]
    map_time: float
    map_pressure: float
    rr_interval: Optional[float]
    systolic_prominence: float
    is_artifact: bool
    artifact_reasons: List[str] = field(default_factory=list)


@dataclass
class ArtifactConfig:
    """Configuration values for the SciPy-only detection pipeline."""

    rr_bounds: Tuple[float, float] = (0.3, 2.0)
    min_prominence: float = 8.0
    prominence_noise_factor: float = 0.18
    scipy_mad_multiplier: float = 3.0
    abs_sys_difference_artifact: float = 10.0
    abs_dia_difference_artifact: float = 10.0
    max_rr_artifact: float = 0.5


@dataclass
class BPFilePreview:
    """Lightweight summary of an export gathered without full ingestion."""

    metadata: Dict[str, str]
    column_names: List[str]
    preview: Optional[pd.DataFrame]
    skiprows: List[int]
    detected_separator: Optional[str]
    raw_lines: List[str] = field(default_factory=list)
    preview_rows: int = 200


@dataclass
class ImportDialogResult:
    """Selections made in the delimiter/column configuration dialog."""

    preview: BPFilePreview
    separator: Optional[str]
    time_column: str
    pressure_column: str
    header_row: int
    first_data_row: int
    time_column_index: int
    pressure_column_index: int
    comment_column: Optional[str] = None
    comment_column_index: Optional[int] = None
    analysis_downsample: int = 1
    plot_downsample: int = 1


def _apply_scipy_artifact_fallback(beats: List[Beat], *, config: ArtifactConfig) -> None:
    """Apply SciPy-only artifact logic for beats detected via the SciPy backend."""

    if not beats:
        return

    systolic = np.asarray([beat.systolic_pressure for beat in beats], dtype=float)
    diastolic = np.asarray([beat.diastolic_pressure for beat in beats], dtype=float)
    mean_arterial = np.asarray([beat.map_pressure for beat in beats], dtype=float)
    rr_intervals = np.asarray(
        [beat.rr_interval if beat.rr_interval is not None else math.nan for beat in beats],
        dtype=float,
    )

    sys_median, sys_mad = _robust_central_tendency(systolic)
    dia_median, dia_mad = _robust_central_tendency(diastolic)
    rr_median, rr_mad = _robust_central_tendency(rr_intervals)
    map_median, map_mad = _robust_central_tendency(mean_arterial)

    mad_multiplier = max(config.scipy_mad_multiplier, 0.0)

    for idx, beat in enumerate(beats):
        triggered: List[str] = []

        if (
            idx > 0
            and math.isfinite(sys_median)
            and math.isfinite(sys_mad)
            and math.isfinite(systolic[idx])
            and math.isfinite(systolic[idx - 1])
        ):
            deviation = abs(systolic[idx] - sys_median)
            diff_prev = abs(systolic[idx] - systolic[idx - 1])
            if (
                deviation > mad_multiplier * sys_mad
                and diff_prev > config.abs_sys_difference_artifact
            ):
                triggered.append("systolic deviation")

        if (
            idx >= 2
            and math.isfinite(dia_median)
            and math.isfinite(dia_mad)
            and math.isfinite(diastolic[idx - 1])
            and math.isfinite(diastolic[idx - 2])
        ):
            prev_dev = abs(diastolic[idx - 1] - dia_median)
            prev_diff = abs(diastolic[idx - 1] - diastolic[idx - 2])
            if (
                prev_dev > mad_multiplier * dia_mad
                and prev_diff > config.abs_dia_difference_artifact
            ):
                triggered.append("preceding diastolic deviation")

        if (
            idx > 0
            and math.isfinite(rr_median)
            and math.isfinite(rr_mad)
            and math.isfinite(rr_intervals[idx])
        ):
            rr_dev = abs(rr_intervals[idx] - rr_median)
            if (
                rr_dev > mad_multiplier * rr_mad
                and rr_intervals[idx] < config.max_rr_artifact
            ):
                triggered.append("short RR interval")

        criteria_count = len(triggered)
        if criteria_count >= 2:
            beat.is_artifact = True
            action = "exclusion" if criteria_count == 3 else "flag"
            beat.artifact_reasons.append(
                "SciPy artifact {} ({} criteria: {})".format(
                    action,
                    f"{criteria_count}/3",
                    ", ".join(triggered),
                )
            )

        point_flags: List[str] = []
        if (
            math.isfinite(sys_median)
            and math.isfinite(sys_mad)
            and math.isfinite(systolic[idx])
            and abs(systolic[idx] - sys_median) > mad_multiplier * sys_mad
        ):
            point_flags.append("systolic")
        if (
            math.isfinite(dia_median)
            and math.isfinite(dia_mad)
            and math.isfinite(diastolic[idx])
            and abs(diastolic[idx] - dia_median) > mad_multiplier * dia_mad
        ):
            point_flags.append("diastolic")
        if (
            math.isfinite(map_median)
            and math.isfinite(map_mad)
            and math.isfinite(mean_arterial[idx])
            and abs(mean_arterial[idx] - map_median) > mad_multiplier * map_mad
        ):
            point_flags.append("MAP")

        if point_flags:
            beat.is_artifact = True
            beat.artifact_reasons.append(
                "SciPy MAD flag for {}".format(", ".join(point_flags))
            )

    for beat in beats:
        if beat.artifact_reasons:
            beat.artifact_reasons = sorted(set(beat.artifact_reasons))


def _robust_central_tendency(values: np.ndarray) -> Tuple[float, float]:
    """Return the median and MAD (with sensible fallbacks)."""

    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return math.nan, math.nan

    median = float(np.median(finite))
    deviations = np.abs(finite - median)
    if deviations.size == 0:
        return median, 0.0

    mad = float(1.4826 * np.median(deviations))
    if mad < 1e-6:
        mad = float(np.std(finite, ddof=0))
    return median, mad


def detect_systolic_peaks(
    signal: np.ndarray,
    fs: float,
    *,
    config: ArtifactConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect systolic peaks using SciPy's peak finder with adaptive prominence."""

    if sp_signal is None:
        raise RuntimeError("SciPy is required for beat detection but is not available.")

    filtered = np.asarray(signal, dtype=float)
    if filtered.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    finite_mask = np.isfinite(filtered)
    if not np.any(finite_mask):
        return np.array([], dtype=int), np.array([], dtype=float)
    if not np.all(finite_mask):
        filtered = filtered.copy()
        valid_idx = np.flatnonzero(finite_mask)
        missing_idx = np.flatnonzero(~finite_mask)
        filtered[missing_idx] = np.interp(missing_idx, valid_idx, filtered[valid_idx])

    detrended = filtered
    if filtered.size >= 3:
        try:
            detrended = sp_signal.detrend(filtered, type="linear")
        except Exception:
            detrended = filtered

    window = max(5, int(round(fs * 0.05)))
    if window % 2 == 0:
        window += 1
    if window >= detrended.size:
        window = detrended.size - 1 if detrended.size % 2 == 0 else detrended.size
    smoothed = detrended
    if window >= 5 and window < detrended.size:
        try:
            smoothed = sp_signal.savgol_filter(
                detrended,
                window_length=window,
                polyorder=2,
                mode="interp",
            )
        except Exception:
            smoothed = detrended

    diff = np.diff(smoothed)
    noise_level = float(np.median(np.abs(diff))) if diff.size else 0.0
    amplitude_span = float(np.percentile(smoothed, 95) - np.percentile(smoothed, 5)) if smoothed.size >= 2 else 0.0
    adaptive_floor = max(
        config.min_prominence,
        config.prominence_noise_factor * max(noise_level * 6.0, amplitude_span * 0.15),
    )

    min_distance = max(1, int(round(config.rr_bounds[0] * fs)))

    peaks, properties = sp_signal.find_peaks(
        smoothed,
        distance=min_distance,
        prominence=adaptive_floor,
    )

    prominences = properties.get("prominences")
    if prominences is None:
        prominences = np.zeros(peaks.shape, dtype=float)
    else:
        prominences = np.asarray(prominences, dtype=float)

    return peaks.astype(int), prominences


def derive_beats(
    time: np.ndarray,
    pressure: np.ndarray,
    *,
    fs: float,
    config: ArtifactConfig,
) -> List[Beat]:
    """Derive beat landmarks using the SciPy-only detection pipeline."""

    if sp_signal is None:
        raise RuntimeError("SciPy is required for beat detection but is not available.")

    time = np.asarray(time, dtype=float)
    pressure = np.asarray(pressure, dtype=float)
    if time.shape != pressure.shape:
        raise ValueError("Time and pressure arrays must have the same length.")
    if fs <= 0:
        raise ValueError("Sampling frequency must be positive.")

    if pressure.size == 0:
        return []

    finite_mask = np.isfinite(pressure)
    if not np.any(finite_mask):
        raise ValueError("Pressure signal contains no valid samples.")

    pressure_filled = pressure.copy()
    if not np.all(finite_mask):
        valid_idx = np.flatnonzero(finite_mask)
        missing_idx = np.flatnonzero(~finite_mask)
        pressure_filled[missing_idx] = np.interp(
            missing_idx,
            valid_idx,
            pressure_filled[valid_idx],
        )

    window = max(5, int(round(fs * 0.05)))
    if window % 2 == 0:
        window += 1
    smoothed = pressure_filled
    if window >= 5 and window < pressure_filled.size:
        try:
            smoothed = sp_signal.savgol_filter(
                pressure_filled,
                window_length=window,
                polyorder=2,
                mode="interp",
            )
        except Exception:
            smoothed = pressure_filled

    systolic_indices, prominences = detect_systolic_peaks(
        smoothed,
        fs=fs,
        config=config,
    )

    beats: List[Beat] = []
    if systolic_indices.size == 0:
        return beats

    rr_samples_limit = int(round(config.rr_bounds[1] * fs)) if config.rr_bounds[1] > 0 else None

    for idx, sys_idx in enumerate(systolic_indices):
        prev_sys = systolic_indices[idx - 1] if idx > 0 else None
        next_sys = systolic_indices[idx + 1] if idx + 1 < systolic_indices.size else None

        search_start = sys_idx
        if next_sys is not None:
            search_end = next_sys
        elif rr_samples_limit is not None:
            search_end = min(pressure_filled.size - 1, sys_idx + rr_samples_limit)
        else:
            search_end = pressure_filled.size - 1
        if search_end <= search_start:
            search_end = min(pressure_filled.size - 1, search_start + 1)

        segment = smoothed[search_start : search_end + 1]
        if segment.size == 0:
            continue
        dia_rel = int(np.argmin(segment))
        dia_idx = search_start + dia_rel

        systolic_time = float(time[sys_idx])
        systolic_pressure = float(pressure[sys_idx]) if math.isfinite(pressure[sys_idx]) else float(pressure_filled[sys_idx])
        diastolic_time = float(time[dia_idx])
        diastolic_pressure = float(pressure[dia_idx]) if math.isfinite(pressure[dia_idx]) else float(pressure_filled[dia_idx])

        map_start = dia_idx
        if next_sys is not None and next_sys > map_start:
            map_end = next_sys
        elif rr_samples_limit is not None:
            map_end = min(pressure_filled.size - 1, map_start + rr_samples_limit)
        else:
            map_end = pressure_filled.size - 1

        map_time = float(time[sys_idx])
        map_pressure = diastolic_pressure + (systolic_pressure - diastolic_pressure) / 3.0
        if map_end > map_start:
            seg_time = time[map_start : map_end + 1]
            seg_pressure = pressure_filled[map_start : map_end + 1]
            duration = float(seg_time[-1] - seg_time[0])
            if duration > 0:
                area = float(np.trapezoid(seg_pressure, seg_time))
                map_pressure = area / duration
                map_time = float((seg_time[0] + seg_time[-1]) / 2.0)

        rr_interval = None
        if prev_sys is not None:
            rr_interval = float(time[sys_idx] - time[prev_sys])

        prominence_value = float(prominences[idx]) if idx < len(prominences) else math.nan

        beats.append(
            Beat(
                systolic_time=systolic_time,
                systolic_pressure=systolic_pressure,
                diastolic_time=diastolic_time,
                diastolic_pressure=diastolic_pressure,
                notch_time=math.nan,
                map_time=map_time,
                map_pressure=map_pressure,
                rr_interval=rr_interval,
                systolic_prominence=prominence_value,
                is_artifact=False,
                artifact_reasons=[],
            )
        )

    _apply_scipy_artifact_fallback(beats, config=config)
    return beats


def plot_waveform(
    time: np.ndarray,
    pressure: np.ndarray,
    beats: Sequence[Beat],
    *,
    show: bool = True,
    save_path: Optional[Path] = None,
    downsample_stride: int = 1,
) -> None:
    """Plot the waveform with annotated beat landmarks."""

    if plt is None:  # pragma: no cover - handled in main
        raise RuntimeError("matplotlib is required for plotting")

    plt.figure(figsize=(12, 6))
    time_values = np.asarray(time, dtype=float)
    pressure_values = np.asarray(pressure, dtype=float)

    if downsample_stride is None or downsample_stride <= 0:
        downsample_stride = 1

    if downsample_stride > 1:
        plot_time = time_values[::downsample_stride]
        plot_pressure = pressure_values[::downsample_stride]
        if plot_time.size == 0 or plot_time[-1] != time_values[-1]:
            plot_time = np.append(plot_time, time_values[-1])
            plot_pressure = np.append(plot_pressure, pressure_values[-1])
    else:
        plot_time = time_values
        plot_pressure = pressure_values

    plt.plot(plot_time, plot_pressure, label="reBAP", color="tab:blue")

    systolic_times = [b.systolic_time for b in beats]
    systolic_pressures = [b.systolic_pressure for b in beats]
    diastolic_times = [b.diastolic_time for b in beats]
    diastolic_pressures = [b.diastolic_pressure for b in beats]
    map_times = [b.map_time for b in beats]
    map_pressures = [b.map_pressure for b in beats]

    artifacts = [idx for idx, beat in enumerate(beats) if beat.is_artifact]
    clean = [idx for idx, beat in enumerate(beats) if not beat.is_artifact]

    if clean:
        plt.scatter(
            np.array(systolic_times)[clean],
            np.array(systolic_pressures)[clean],
            color="tab:red",
            label="Systolic",
        )
        plt.scatter(
            np.array(diastolic_times)[clean],
            np.array(diastolic_pressures)[clean],
            color="tab:green",
            label="Diastolic",
        )
        plt.scatter(
            np.array(map_times)[clean],
            np.array(map_pressures)[clean],
            color="tab:orange",
            label="MAP",
        )

    if artifacts:
        plt.scatter(
            np.array(systolic_times)[artifacts],
            np.array(systolic_pressures)[artifacts],
            marker="x",
            color="k",
            label="Flagged beats",
        )

    plt.xlabel("Time (s)")
    plt.ylabel("Pressure (mmHg)")
    plt.title("Continuous blood pressure with beat landmarks")
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150)

    if show:
        plt.show()
    else:
        plt.close()


def summarise_beats(beats: Sequence[Beat]) -> pd.DataFrame:
    """Build a DataFrame summarising the detected beats."""

    base_columns = {
        "systolic_time": pd.Series(dtype=float),
        "systolic_pressure": pd.Series(dtype=float),
        "diastolic_time": pd.Series(dtype=float),
        "diastolic_pressure": pd.Series(dtype=float),
        "notch_time": pd.Series(dtype=float),
        "map_time": pd.Series(dtype=float),
        "map_pressure": pd.Series(dtype=float),
        "rr_interval": pd.Series(dtype=float),
        "prominence": pd.Series(dtype=float),
        "artifact": pd.Series(dtype=bool),
        "artifact_reasons": pd.Series(dtype=object),
    }

    records = []
    for beat in beats:
        records.append(
            {
                "systolic_time": beat.systolic_time,
                "systolic_pressure": beat.systolic_pressure,
                "diastolic_time": beat.diastolic_time,
                "diastolic_pressure": beat.diastolic_pressure,
                "notch_time": beat.notch_time,
                "map_time": beat.map_time,
                "map_pressure": beat.map_pressure,
                "rr_interval": beat.rr_interval,
                "prominence": beat.systolic_prominence,
                "artifact": beat.is_artifact,
                "artifact_reasons": ", ".join(beat.artifact_reasons) if beat.artifact_reasons else "",
            }
        )
    if not records:
        return pd.DataFrame(base_columns)

    return pd.DataFrame.from_records(records)

def _resample_even_grid(times: np.ndarray, values: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate ``values`` onto an even time grid at ``fs`` Hz."""

    if times.size == 0 or values.size == 0 or fs <= 0:
        return np.array([]), np.array([])

    start = float(times[0])
    end = float(times[-1])
    if end <= start:
        return np.array([]), np.array([])

    grid = np.arange(start, end, 1.0 / fs, dtype=float)
    finite_mask = np.isfinite(values)
    if not np.any(finite_mask):
        return grid, np.full_like(grid, np.nan)

    interp_values = np.interp(grid, times[finite_mask], values[finite_mask])
    return grid, interp_values


def compute_bpv_psd(
    beat_times: np.ndarray,
    beat_values: np.ndarray,
    *,
    resample_fs: float,
    nperseg: int = 256,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Compute Welch PSD for beat-to-beat series when SciPy is present."""

    if sp_signal is None:
        return None

    grid_time, resampled = _resample_even_grid(beat_times, beat_values, resample_fs)
    if grid_time.size < nperseg:
        return None

    try:
        freqs, power = sp_signal.welch(resampled, fs=resample_fs, nperseg=min(nperseg, len(resampled)))
    except Exception:
        return None

    return freqs, power


def _bandpower(freqs: np.ndarray, power: np.ndarray, band: Tuple[float, float]) -> float:
    """Integrate Welch spectrum within ``band`` (in Hz)."""

    low, high = band
    mask = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        return math.nan
    return float(np.trapezoid(power[mask], freqs[mask]))


def compute_rr_metrics(rr_intervals: np.ndarray, *, pnn_threshold: float) -> Dict[str, float]:
    """Compute summary statistics for RR-intervals."""

    rr = rr_intervals[np.isfinite(rr_intervals)]
    if rr.size == 0:
        return {
            "mean": math.nan,
            "sd": math.nan,
            "cv": math.nan,
            "rmssd": math.nan,
            "arv": math.nan,
            "pnn": math.nan,
        }

    diffs = np.diff(rr)
    rmssd = float(np.sqrt(np.mean(diffs**2))) if diffs.size else math.nan
    arv = float(np.mean(np.abs(diffs))) if diffs.size else math.nan
    if diffs.size:
        pnn = float(np.mean(np.abs(diffs) > pnn_threshold))
    else:
        pnn = math.nan

    mean_rr = float(np.mean(rr))
    sd_rr = float(np.std(rr, ddof=1)) if rr.size > 1 else 0.0
    cv_rr = float(sd_rr / mean_rr) if mean_rr else math.nan

    return {
        "mean": mean_rr,
        "sd": sd_rr,
        "cv": cv_rr,
        "rmssd": rmssd,
        "arv": arv,
        "pnn": pnn,
    }


def print_column_overview(frame: Optional[pd.DataFrame]) -> None:
    """Display column names, dtypes, and first numeric samples to aid selection."""

    if frame is None:
        print("  Preview unavailable; data will be loaded during import.")
        return

    print("\nAvailable columns:")
    for name in frame.columns:
        series = frame[name]
        dtype = str(series.dtype)
        if pd.api.types.is_numeric_dtype(series):
            sample = series.dropna().astype(float).head(3).to_list()
            preview = ", ".join(f"{value:.3f}" for value in sample)
        else:
            sample = series.dropna().astype(str).head(3).to_list()
            preview = ", ".join(sample)
        if not preview:
            preview = "(no finite samples)"
        print(f"  - {name} [{dtype}]: {preview}")


def select_file_via_dialog() -> Optional[Path]:
    """Open a file picker to select a waveform export."""

    if tk is None or filedialog is None:
        print("Tkinter is not available; please supply a file path as an argument.")
        return None

    root = tk.Tk()
    root.withdraw()
    file_types = [("Text files", "*.txt"), ("All files", "*")]
    filename = filedialog.askopenfilename(title="Select continuous BP export", filetypes=file_types)
    root.destroy()

    if not filename:
        return None
    return Path(filename)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Explore beat-to-beat arterial pressure waveforms.")
    parser.add_argument("file", nargs="?", help="Path to the exported waveform text file.")
    parser.add_argument("--column", default="reBAP", help="Name of the arterial pressure column to analyse.")
    parser.add_argument("--time-column", default="Time", help="Name of the time column (defaults to 'Time').")
    parser.add_argument("--min-rr", type=float, default=0.3, help="Minimum RR interval in seconds.")
    parser.add_argument("--max-rr", type=float, default=2.0, help="Maximum RR interval in seconds.")
    parser.add_argument("--min-prominence", type=float, default=8.0, help="Minimum systolic prominence (mmHg).")
    parser.add_argument(
        "--prominence-factor",
        type=float,
        default=0.18,
        help="Multiplier for adaptive prominence threshold based on noise.",
    )
    parser.add_argument(
        "--scipy-mad-multiplier",
        type=float,
        default=3.0,
        help="MAD multiplier used by the SciPy fallback artifact logic.",
    )
    parser.add_argument(
        "--scipy-abs-sys-diff",
        type=float,
        default=10.0,
        help="Absolute systolic difference (mmHg) threshold for SciPy artifact exclusion.",
    )
    parser.add_argument(
        "--scipy-abs-dia-diff",
        type=float,
        default=10.0,
        help="Absolute diastolic difference (mmHg) threshold for SciPy artifact exclusion.",
    )
    parser.add_argument(
        "--scipy-max-rr",
        type=float,
        default=0.5,
        help="Maximum RR interval (s) considered in SciPy artifact exclusion.",
    )
    parser.add_argument(
        "--pnn-threshold",
        type=float,
        default=50.0,
        help="pNN threshold in milliseconds for RR statistics.",
    )
    parser.add_argument("--psd", action="store_true", help="Compute Welch PSD for clean beat series.")
    parser.add_argument("--psd-fs", type=float, default=4.0, help="Resampling frequency for PSD (Hz).")
    parser.add_argument("--psd-nperseg", type=int, default=256, help="nperseg parameter for Welch PSD.")
    parser.add_argument(
        "--separator",
        help="Override the column separator if auto-detection fails (e.g., ',' or \\t).",
    )
    parser.add_argument("--out", type=Path, help="Optional path to export the beat summary CSV.")
    parser.add_argument("--savefig", type=Path, help="Save the plot to this path instead of/as well as showing it.")
    parser.add_argument("--no-plot", action="store_true", help="Skip interactive plot display.")

    return parser


def main(argv: Sequence[str]) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv[1:])

    if args.min_rr >= args.max_rr:
        parser.error("--min-rr must be less than --max-rr")

    separator_override: Optional[str] = None
    if args.separator:
        raw_separator = args.separator
        try:
            raw_separator = bytes(raw_separator, "utf-8").decode("unicode_escape")
        except Exception:
            pass
        lowered = raw_separator.lower()
        if lowered == "tab":
            raw_separator = "\t"
        elif lowered == "space":
            raw_separator = " "
        separator_override = raw_separator if raw_separator else None

    if args.file:
        file_path = Path(args.file)
    else:
        file_path = select_file_via_dialog()
        if file_path is None:
            print("No file selected. Exiting.")
            return 1

    if not file_path.exists():
        print(f"File not found: {file_path}")
        return 1

    try:
        preview = _scan_bp_file(file_path, separator_override=separator_override)
    except Exception as exc:  # pragma: no cover - interactive feedback
        print(f"Failed to parse file header: {exc}")
        return 1

    time_default = _default_column_selection(
        preview.column_names,
        args.time_column,
        fallback="Time",
    )
    remaining = [name for name in preview.column_names if name != time_default]
    pressure_default = _default_column_selection(
        remaining if remaining else preview.column_names,
        args.column if args.column != time_default else None,
        fallback="reBAP",
    )

    selected_time_column: Optional[str] = None
    selected_pressure_column: Optional[str] = None
    dialog_shown = False
    dialog_result: Optional[ImportDialogResult] = None
    analysis_downsample = 1
    plot_downsample = 10
    if tk is not None:
        try:
            dialog_result = launch_import_configuration_dialog(
                file_path,
                preview,
                time_default=time_default,
                pressure_default=pressure_default,
                separator_override=separator_override,
            )
            dialog_shown = True
        except Exception as exc:  # pragma: no cover - interactive feedback
            print(f"Failed to open import configuration dialog: {exc}")
            dialog_result = None
        if dialog_result:
            preview = dialog_result.preview
            separator_override = dialog_result.separator
            selected_time_column = dialog_result.time_column
            selected_pressure_column = dialog_result.pressure_column
            analysis_downsample = max(1, int(dialog_result.analysis_downsample))
            plot_downsample = max(1, int(dialog_result.plot_downsample))

    print(f"Loaded file preview: {file_path}")
    print(f"\nColumn preview (first {preview.preview_rows} rows loaded for preview):")
    preview_frame = preview.preview
    if preview_frame is None:
        try:
            preview_frame = _build_preview_dataframe(
                file_path,
                column_names=preview.column_names,
                skiprows=preview.skiprows,
                separator=preview.detected_separator,
                max_rows=preview.preview_rows,
            )
            preview.preview = preview_frame
        except Exception as exc:  # pragma: no cover - interactive feedback
            print(f"  Unable to build preview table: {exc}")
            preview_frame = None
    print_column_overview(preview_frame)

    if selected_time_column is None or selected_pressure_column is None:
        selected_time_column, selected_pressure_column = select_time_pressure_columns(
            preview,
            requested_time=args.time_column,
            requested_pressure=args.column,
            allow_gui=not dialog_shown,
        )

    try:
        frame, metadata = load_bp_file(
            file_path,
            preview=preview,
            time_column=selected_time_column,
            pressure_column=selected_pressure_column,
            separator=separator_override,
        )
    except Exception as exc:  # pragma: no cover - interactive feedback
        print(f"Failed to load file: {exc}")
        return 1

    print(f"Loaded columns: time='{selected_time_column}', pressure='{selected_pressure_column}'")

    if selected_time_column in frame.columns:
        time_series = pd.to_numeric(
            frame[selected_time_column], errors="coerce", downcast="float"
        ).astype(np.float32, copy=False)
        time = time_series.to_numpy()
    else:
        time = None

    pressure_series = pd.to_numeric(
        frame[selected_pressure_column], errors="coerce", downcast="float"
    ).astype(np.float32, copy=False)
    pressure = pressure_series.to_numpy()

    interval = parse_interval(metadata, frame)
    fs = 1.0 / interval if interval > 0 else 1.0

    if time is None or not np.any(np.isfinite(time)):
        time = np.arange(len(frame), dtype=np.float32) * np.float32(interval)

    if analysis_downsample > 1:
        downsampled_time = time[::analysis_downsample]
        downsampled_pressure = pressure[::analysis_downsample]
        if downsampled_time.size == 0 or downsampled_pressure.size == 0:
            print(
                "Warning: analysis downsampling factor removed all samples; using full resolution."
            )
            analysis_downsample = 1
        else:
            time = downsampled_time
            pressure = downsampled_pressure
            interval *= analysis_downsample
            fs = 1.0 / interval if interval > 0 else fs / analysis_downsample
            print(
                "Applying analysis downsampling: "
                f"using every {analysis_downsample}th sample (effective fs {fs:.3f} Hz)."
            )

    config = ArtifactConfig(
        rr_bounds=(args.min_rr, args.max_rr),
        min_prominence=args.min_prominence,
        prominence_noise_factor=args.prominence_factor,
        scipy_mad_multiplier=args.scipy_mad_multiplier,
        abs_sys_difference_artifact=args.scipy_abs_sys_diff,
        abs_dia_difference_artifact=args.scipy_abs_dia_diff,
        max_rr_artifact=args.scipy_max_rr,
    )

    try:
        beats = derive_beats(
            time,
            pressure,
            fs=fs,
            config=config,
        )
    except Exception as exc:  # pragma: no cover - interactive feedback
        print(f"Beat detection failed: {exc}")
        return 1

    summary = summarise_beats(beats)

    if args.out:
        try:
            summary.to_csv(args.out, index=False)
            print(f"Beat summary exported to {args.out}")
        except Exception as exc:  # pragma: no cover - filesystem feedback
            print(f"Failed to save summary CSV: {exc}")

    clean_beats = summary[~summary["artifact"]].copy()
    if not clean_beats.empty:
        print("\nDetected beats (first clean entries):")
        print(
            clean_beats[
                [
                    "systolic_time",
                    "systolic_pressure",
                    "diastolic_pressure",
                    "map_pressure",
                    "rr_interval",
                ]
            ].head()
        )
        print()
        print(
            "Averages — Systolic: "
            f"{clean_beats['systolic_pressure'].mean():.1f} mmHg, Diastolic: "
            f"{clean_beats['diastolic_pressure'].mean():.1f} mmHg, MAP: "
            f"{clean_beats['map_pressure'].mean():.1f} mmHg"
        )

        rr_metrics = compute_rr_metrics(
            clean_beats["rr_interval"].to_numpy(dtype=float),
            pnn_threshold=args.pnn_threshold / 1000.0,
        )
        print(
            "RR metrics — mean: "
            f"{rr_metrics['mean']:.3f}s, SD: {rr_metrics['sd']:.3f}s, CV: {rr_metrics['cv']:.3f}, "
            f"RMSSD: {rr_metrics['rmssd']:.3f}s, ARV: {rr_metrics['arv']:.3f}s, "
            f"pNN>{args.pnn_threshold:.0f}ms: {rr_metrics['pnn'] * 100 if not math.isnan(rr_metrics['pnn']) else float('nan'):.1f}%"
        )
    else:
        print("No clean beats detected; all beats were flagged as artefacts.")

    if summary["artifact"].any():
        print()
        print(
            f"{summary['artifact'].sum()} beats were flagged as potential artefacts."
        )

    if args.psd:
        if sp_signal is None:
            print("SciPy is not installed; skipping PSD computation.")
        elif clean_beats.empty:
            print("PSD skipped because no clean beats are available.")
        else:
            beat_times = clean_beats["systolic_time"].to_numpy(dtype=float)
            for label, column in [
                ("SBP", "systolic_pressure"),
                ("DBP", "diastolic_pressure"),
                ("MAP", "map_pressure"),
            ]:
                result = compute_bpv_psd(
                    beat_times,
                    clean_beats[column].to_numpy(dtype=float),
                    resample_fs=args.psd_fs,
                    nperseg=args.psd_nperseg,
                )
                if result is None:
                    print(f"PSD for {label} unavailable (insufficient data or SciPy error).")
                    continue
                freqs, power = result
                lf = _bandpower(freqs, power, (0.04, 0.15))
                hf = _bandpower(freqs, power, (0.15, 0.4))
                hf = hf if hf > 0 else math.nan
                lf_hf = lf / hf if not math.isnan(hf) and hf != 0 else math.nan
                print(
                    f"{label} PSD — LF: {lf:.3f} mmHg^2, HF: {hf:.3f} mmHg^2, LF/HF: {lf_hf:.3f}"
                )

    show_plot = not args.no_plot
    if plt is None and (show_plot or args.savefig):
        print("matplotlib is required to visualise or save the waveform plot.")
    elif show_plot or args.savefig:
        plot_waveform(
            time,
            pressure,
            beats,
            show=show_plot,
            save_path=args.savefig,
            downsample_stride=plot_downsample,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))