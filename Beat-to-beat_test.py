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
"""

from __future__ import annotations

import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import tkinter as tk
    from tkinter import filedialog
except Exception:  # pragma: no cover - tkinter may be unavailable in some envs.
    tk = None
    filedialog = None

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


@dataclass
class Beat:
    """Container describing a detected beat."""

    systolic_time: float
    systolic_pressure: float
    diastolic_time: float
    diastolic_pressure: float
    map_time: float
    map_pressure: float
    rr_interval: Optional[float]
    systolic_prominence: float
    is_artifact: bool


def _parse_list_field(raw_value: str) -> List[str]:
    """Split a metadata field that contains multiple values.

    The files we ingest typically separate list fields using tabs.  When
    tabs are not present, we fall back to any amount of whitespace.
    """

    raw_value = raw_value.strip()
    if not raw_value:
        return []

    # Try splitting on one or more tab characters first.
    parts = [item.strip() for item in re.split(r"\t+", raw_value) if item.strip()]
    if parts:
        return parts

    # Fallback: split on whitespace while keeping multi-word labels intact
    # by joining adjacent capitalised tokens (e.g., "Respiratory Belt").
    tokens = [tok for tok in raw_value.split(" ") if tok]
    merged: List[str] = []
    buffer: List[str] = []
    for tok in tokens:
        if tok and tok[0].isupper() and not tok.isupper():
            if buffer:
                merged.append(" ".join(buffer))
                buffer = []
            buffer.append(tok)
        elif buffer:
            buffer.append(tok)
        else:
            merged.append(tok)
    if buffer:
        merged.append(" ".join(buffer))

    return merged or tokens


def load_bp_file(file_path: Path) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Load a text export containing beat-to-beat blood pressure data."""

    metadata: Dict[str, str] = {}
    data_rows: List[List[float]] = []

    with file_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            # Metadata entries contain an equals sign.
            if "=" in line and not re.match(r"^[+-]?[0-9]", line):
                key, value = line.split("=", 1)
                metadata[key.strip()] = value.strip()
                continue

            # Remaining lines are expected to be numeric data.
            if "#" in line:
                line = line.split("#", 1)[0].strip()
                if not line:
                    continue

            parts = [p for p in re.split(r"\s+", line) if p]
            try:
                row = [float(p) if p.lower() != "nan" else math.nan for p in parts]
            except ValueError as exc:
                raise ValueError(f"Could not parse data row: {raw_line.strip()!r}") from exc
            data_rows.append(row)

    if not data_rows:
        raise ValueError("No numeric data found in file.")

    channels = _parse_list_field(metadata.get("ChannelTitle", ""))
    if channels and len(data_rows[0]) == len(channels) + 1:
        column_names = ["Time"] + channels
    else:
        # Fallback: generate generic column names.
        column_names = [f"col_{idx}" for idx in range(len(data_rows[0]))]

    frame = pd.DataFrame(data_rows, columns=column_names)
    return frame, metadata


def parse_interval(metadata: Dict[str, str], frame: pd.DataFrame) -> float:
    """Extract the sampling interval from metadata or infer it from the data."""

    interval_raw = metadata.get("Interval")
    if interval_raw:
        match = re.search(r"([0-9]+\.?[0-9]*)", interval_raw)
        if match:
            return float(match.group(1))

    # Infer from the time column if available.
    if "Time" in frame.columns:
        time_values = frame["Time"].to_numpy()
        if len(time_values) > 1:
            diffs = np.diff(time_values)
            diffs = diffs[~np.isnan(diffs)]
            if len(diffs) > 0:
                return float(np.median(diffs))

    # Fallback to 1 Hz.
    return 1.0


def smooth_signal(signal: np.ndarray, window_seconds: float, fs: float) -> np.ndarray:
    """Apply a simple moving average filter to suppress high-frequency noise."""

    window_samples = max(1, int(round(window_seconds * fs)))
    if window_samples <= 1:
        return signal.astype(float, copy=False)

    kernel = np.ones(window_samples, dtype=float) / float(window_samples)
    return np.convolve(signal, kernel, mode="same")


def detrend_signal(signal: np.ndarray, fs: float, window_seconds: float = 0.6) -> np.ndarray:
    """Remove slow-varying trends using a long moving-average window."""

    if window_seconds <= 0:
        return signal

    baseline = smooth_signal(signal, window_seconds=window_seconds, fs=fs)
    return signal - baseline


def detect_systolic_peaks(
    signal: np.ndarray,
    fs: float,
    *,
    min_rr: float,
    max_rr: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect systolic peaks using adaptive upstroke and prominence checks.

    The Lab chart module applies a two-step process: first it emphasises the
    rapid systolic upstroke and then it validates peaks based on their
    prominence relative to neighbouring troughs.  To emulate that behaviour we
    detrend the signal, analyse the velocity profile, and adaptively gate the
    candidate peaks before evaluating their prominences.
    """

    if signal.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    detrended = detrend_signal(signal, fs, window_seconds=0.6)
    velocity = np.gradient(detrended)
    velocity = smooth_signal(velocity, window_seconds=0.03, fs=fs)
    positive_velocity = np.clip(velocity, a_min=0.0, a_max=None)

    if not np.any(positive_velocity > 0):
        return np.array([], dtype=int), np.array([], dtype=float)

    baseline = float(np.median(positive_velocity[positive_velocity > 0]))
    mad = float(np.median(np.abs(positive_velocity[positive_velocity > 0] - baseline)))
    if mad == 0.0:
        mad = float(np.std(positive_velocity))
    if mad == 0.0:
        mad = 1.0

    upstroke_threshold = baseline + 2.5 * mad
    if upstroke_threshold <= 0:
        upstroke_threshold = baseline * 1.5 if baseline > 0 else 0.5

    crossings = np.flatnonzero(
        (positive_velocity[:-1] < upstroke_threshold)
        & (positive_velocity[1:] >= upstroke_threshold)
    )
    if crossings.size:
        candidate_onsets = crossings + 1
    else:
        candidate_onsets = np.arange(signal.size)

    min_distance = max(1, int(round(min_rr * fs)))
    max_distance = max(min_distance + 1, int(round(max_rr * fs)))
    search_horizon = max(min_distance, int(round(0.45 * fs)))

    amplitude_span = float(np.percentile(signal, 95) - np.percentile(signal, 5))
    min_prominence = max(8.0, 0.18 * amplitude_span)

    peaks: List[int] = []
    prominences: List[float] = []
    last_peak = -max_distance

    for onset in candidate_onsets:
        search_start = int(onset)
        search_end = min(signal.size, search_start + search_horizon)
        if search_end <= search_start:
            continue

        segment = signal[search_start:search_end]
        peak_rel = int(np.argmax(segment))
        peak_idx = search_start + peak_rel

        if peak_idx - last_peak < min_distance:
            if peaks and signal[peak_idx] > signal[peaks[-1]]:
                peaks[-1] = peak_idx
            continue

        left = signal[max(0, peak_idx - max_distance) : peak_idx + 1]
        right = signal[peak_idx : min(signal.size, peak_idx + max_distance)]
        if left.size == 0 or right.size == 0:
            continue

        left_min = float(np.min(left))
        right_min = float(np.min(right))
        prominence = float(signal[peak_idx] - max(left_min, right_min))
        if prominence < min_prominence:
            continue

        peaks.append(peak_idx)
        prominences.append(prominence)
        last_peak = peak_idx

    if len(peaks) < 3:
        fallback_peaks, fallback_prom = find_prominent_peaks(
            signal, fs=fs, min_rr=min_rr, max_rr=max_rr, min_prominence=min_prominence
        )
        if fallback_peaks.size:
            return fallback_peaks, fallback_prom

    return np.asarray(peaks, dtype=int), np.asarray(prominences, dtype=float)


def find_prominent_peaks(
    signal: np.ndarray,
    fs: float,
    min_rr: float = 0.3,
    max_rr: float = 2.0,
    min_prominence: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect systolic peaks using a simple prominence-based heuristic.

    Parameters
    ----------
    signal:
        The (smoothed) arterial pressure signal.
    fs:
        Sampling frequency in Hz.
    min_rr / max_rr:
        Minimum and maximum plausible RR-intervals (seconds).
    min_prominence:
        Minimum prominence in mmHg required to accept a peak.  When not
        provided, it defaults to 20% of the signal's standard deviation.
    """

    if min_prominence is None:
        finite = signal[np.isfinite(signal)]
        if len(finite) == 0:
            raise ValueError("Signal contains no finite samples.")
        min_prominence = 0.2 * float(np.std(finite))

    min_distance = max(1, int(min_rr * fs))
    max_distance = max(1, int(max_rr * fs))

    peaks: List[int] = []
    prominences: List[float] = []
    last_peak = -max_distance

    for idx in range(1, len(signal) - 1):
        if signal[idx] <= signal[idx - 1] or signal[idx] < signal[idx + 1]:
            continue

        left_window = signal[max(0, idx - max_distance) : idx]
        right_window = signal[idx + 1 : min(len(signal), idx + max_distance + 1)]
        if left_window.size == 0 or right_window.size == 0:
            continue

        left_min = float(np.min(left_window))
        right_min = float(np.min(right_window))
        prominence = signal[idx] - max(left_min, right_min)
        if prominence < min_prominence:
            continue

        if idx - last_peak < min_distance:
            # Retain the peak with the larger prominence within the refractory period.
            if prominence > prominences[-1]:
                peaks[-1] = idx
                prominences[-1] = prominence
            continue

        peaks.append(idx)
        prominences.append(prominence)
        last_peak = idx

    return np.asarray(peaks, dtype=int), np.asarray(prominences, dtype=float)


def derive_beats(
    time: np.ndarray,
    pressure: np.ndarray,
    fs: float,
    min_rr: float = 0.3,
    max_rr: float = 2.0,
) -> List[Beat]:
    """Derive systolic/diastolic/MAP landmarks from a continuous waveform."""

    if len(time) != len(pressure):
        raise ValueError("Time and pressure arrays must have the same length.")

    finite_mask = np.isfinite(pressure)
    if not np.any(finite_mask):
        raise ValueError("Pressure signal contains no valid samples.")

    pressure_filled = pressure.copy()
    if not np.all(finite_mask):
        pressure_filled[~finite_mask] = np.interp(
            np.flatnonzero(~finite_mask),
            np.flatnonzero(finite_mask),
            pressure[finite_mask],
        )

    smoothed = smooth_signal(pressure_filled, window_seconds=0.03, fs=fs)
    systolic_indices, prominences = detect_systolic_peaks(
        smoothed, fs=fs, min_rr=min_rr, max_rr=max_rr
    )

    beats: List[Beat] = []
    for idx, sys_idx in enumerate(systolic_indices):
        prev_sys = systolic_indices[idx - 1] if idx > 0 else None
        next_sys = systolic_indices[idx + 1] if idx + 1 < len(systolic_indices) else None

        if prev_sys is not None:
            search_start = prev_sys + max(1, int(round(0.05 * fs)))
        else:
            search_start = max(0, sys_idx - int(round(max_rr * fs)))

        search_end = sys_idx
        if next_sys is not None:
            search_end = min(search_end, next_sys - max(1, int(round(0.15 * fs))))

        search_start = max(0, min(search_start, sys_idx))
        search_end = max(search_start + 1, min(sys_idx + 1, search_end + 1))

        segment = smoothed[search_start:search_end]
        if segment.size == 0:
            continue
        dia_rel = int(np.argmin(segment))
        dia_idx = search_start + dia_rel

        systolic_time = float(time[sys_idx])
        systolic_pressure = float(pressure[sys_idx])
        diastolic_time = float(time[dia_idx])
        diastolic_pressure = float(pressure[dia_idx])
        map_pressure = diastolic_pressure + (systolic_pressure - diastolic_pressure) / 3.0
        map_time = float((systolic_time + diastolic_time) / 2.0)
        rr_interval = None
        if idx > 0:
            rr_interval = float(systolic_time - time[systolic_indices[idx - 1]])

        beats.append(
            Beat(
                systolic_time=systolic_time,
                systolic_pressure=systolic_pressure,
                diastolic_time=diastolic_time,
                diastolic_pressure=diastolic_pressure,
                map_time=map_time,
                map_pressure=map_pressure,
                rr_interval=rr_interval,
                systolic_prominence=float(prominences[idx]),
                is_artifact=False,
            )
        )

    apply_artifact_rules(beats)
    return beats


def _rolling_median_and_mad(values: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return rolling median and MAD for ``values`` using an odd-sized window."""

    if window % 2 == 0:
        raise ValueError("window must be odd to compute centred statistics")

    medians = np.full_like(values, np.nan, dtype=float)
    mads = np.full_like(values, np.nan, dtype=float)
    half_window = window // 2

    for idx in range(len(values)):
        start = max(0, idx - half_window)
        end = min(len(values), idx + half_window + 1)
        segment = values[start:end]
        segment = segment[np.isfinite(segment)]
        if segment.size < 3:
            continue
        median = float(np.median(segment))
        deviations = np.abs(segment - median)
        mad = float(1.4826 * np.median(deviations)) if deviations.size else 0.0
        if mad < 1e-3:
            # Fallback to standard deviation when the MAD is tiny (near-constant segment).
            mad = float(np.std(segment, ddof=0))
        medians[idx] = median
        mads[idx] = mad

    return medians, mads


def _robust_central_tendency(values: np.ndarray) -> Tuple[float, float]:
    """Compute median and MAD with sensible fallbacks for empty arrays."""

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return math.nan, math.nan
    median = float(np.median(finite))
    deviations = np.abs(finite - median)
    mad = float(1.4826 * np.median(deviations)) if deviations.size else 0.0
    if mad < 1e-3:
        mad = float(np.std(finite, ddof=0))
    return median, mad


def apply_artifact_rules(beats: List[Beat]) -> None:
    """Flag beats that violate simple physiological heuristics."""

    if not beats:
        return

    systolic_values = np.array([b.systolic_pressure for b in beats], dtype=float)
    diastolic_values = np.array([b.diastolic_pressure for b in beats], dtype=float)
    rr_values = np.array([
        b.rr_interval if b.rr_interval is not None else math.nan for b in beats
    ])

    sys_med, sys_mad = _robust_central_tendency(systolic_values)
    dia_med, dia_mad = _robust_central_tendency(diastolic_values)
    rr_med, rr_mad = _robust_central_tendency(rr_values)

    sys_roll_med, sys_roll_mad = _rolling_median_and_mad(systolic_values, window=9)
    dia_roll_med, dia_roll_mad = _rolling_median_and_mad(diastolic_values, window=9)
    rr_roll_med, rr_roll_mad = _rolling_median_and_mad(rr_values, window=9)

    for idx, beat in enumerate(beats):
        severe_reasons: List[str] = []
        soft_reasons: List[str] = []

        if not math.isfinite(beat.systolic_pressure) or not math.isfinite(beat.diastolic_pressure):
            severe_reasons.append("non-finite pressure")
        if beat.systolic_pressure < 40 or beat.systolic_pressure > 260:
            severe_reasons.append("implausible systolic")
        if beat.diastolic_pressure < 20 or beat.diastolic_pressure > 160:
            severe_reasons.append("implausible diastolic")
        if beat.systolic_pressure - beat.diastolic_pressure < 10:
            severe_reasons.append("low pulse pressure")

        sys_ref = sys_roll_med[idx] if math.isfinite(sys_roll_med[idx]) else sys_med
        sys_scale = sys_roll_mad[idx] if math.isfinite(sys_roll_mad[idx]) else sys_mad
        if not math.isfinite(sys_scale):
            sys_scale = 0.0
        sys_scale = max(sys_scale, 5.0)
        if math.isfinite(sys_ref) and math.isfinite(beat.systolic_pressure):
            if abs(beat.systolic_pressure - sys_ref) > max(45.0, 4.0 * sys_scale):
                soft_reasons.append("systolic deviation")

        dia_ref = dia_roll_med[idx] if math.isfinite(dia_roll_med[idx]) else dia_med
        dia_scale = dia_roll_mad[idx] if math.isfinite(dia_roll_mad[idx]) else dia_mad
        if not math.isfinite(dia_scale):
            dia_scale = 0.0
        dia_scale = max(dia_scale, 4.0)
        if math.isfinite(dia_ref) and math.isfinite(beat.diastolic_pressure):
            if abs(beat.diastolic_pressure - dia_ref) > max(35.0, 4.0 * dia_scale):
                soft_reasons.append("diastolic deviation")

        if beat.rr_interval is not None and math.isfinite(beat.rr_interval):
            if beat.rr_interval < 0.3 or beat.rr_interval > 2.5:
                severe_reasons.append("rr outside bounds")
            rr_ref = rr_roll_med[idx] if math.isfinite(rr_roll_med[idx]) else rr_med
            rr_scale = rr_roll_mad[idx] if math.isfinite(rr_roll_mad[idx]) else rr_mad
            if not math.isfinite(rr_scale):
                rr_scale = 0.0
            rr_scale = max(rr_scale, 0.1)
            if math.isfinite(rr_ref):
                if abs(beat.rr_interval - rr_ref) > max(0.5, 4.0 * rr_scale):
                    soft_reasons.append("rr deviation")

        if beat.systolic_prominence < 5:
            soft_reasons.append("low prominence")

        beat.is_artifact = bool(severe_reasons or len(soft_reasons) >= 2)


def plot_waveform(time: np.ndarray, pressure: np.ndarray, beats: Sequence[Beat]) -> None:
    """Plot the waveform with annotated beat landmarks."""

    if plt is None:  # pragma: no cover - handled in main
        raise RuntimeError("matplotlib is required for plotting")

    plt.figure(figsize=(12, 6))
    plt.plot(time, pressure, label="reBAP", color="tab:blue")

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
    plt.show()


def summarise_beats(beats: Sequence[Beat]) -> pd.DataFrame:
    """Build a DataFrame summarising the detected beats."""

    records = []
    for beat in beats:
        records.append(
            {
                "systolic_time": beat.systolic_time,
                "systolic_pressure": beat.systolic_pressure,
                "diastolic_time": beat.diastolic_time,
                "diastolic_pressure": beat.diastolic_pressure,
                "map_time": beat.map_time,
                "map_pressure": beat.map_pressure,
                "rr_interval": beat.rr_interval,
                "prominence": beat.systolic_prominence,
                "artifact": beat.is_artifact,
            }
        )
    return pd.DataFrame.from_records(records)


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


def main(argv: Sequence[str]) -> int:
    if plt is None:
        print("matplotlib is required to visualise the waveform. Please install it and retry.")
        return 1
    if len(argv) > 1:
        file_path = Path(argv[1])
    else:
        file_path = select_file_via_dialog()
        if file_path is None:
            print("No file selected. Exiting.")
            return 1

    if not file_path.exists():
        print(f"File not found: {file_path}")
        return 1

    try:
        frame, metadata = load_bp_file(file_path)
    except Exception as exc:  # pragma: no cover - interactive feedback
        print(f"Failed to load file: {exc}")
        return 1

    if "reBAP" not in frame.columns:
        print("The selected file does not contain a 'reBAP' column.")
        return 1

    interval = parse_interval(metadata, frame)
    fs = 1.0 / interval if interval > 0 else 1.0

    time = frame["Time"].to_numpy() if "Time" in frame.columns else np.arange(len(frame)) * interval
    pressure = frame["reBAP"].to_numpy()

    try:
        beats = derive_beats(time, pressure, fs=fs)
    except Exception as exc:  # pragma: no cover - interactive feedback
        print(f"Beat detection failed: {exc}")
        return 1

    summary = summarise_beats(beats)

    clean_beats = summary[~summary["artifact"]]
    if not clean_beats.empty:
        print("Detected beats (clean):")
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
    else:
        print("No clean beats detected; all beats were flagged as artefacts.")

    if summary["artifact"].any():
        print()
        print(
            f"{summary['artifact'].sum()} beats were flagged as potential artefacts "
            "based on amplitude, RR-interval, or waveform prominence heuristics."
        )

    plot_waveform(time, pressure, beats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
