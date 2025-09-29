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

    window_samples = max(1, int(window_seconds * fs))
    kernel = np.ones(window_samples) / window_samples
    return np.convolve(signal, kernel, mode="same")


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

    smoothed = smooth_signal(pressure_filled, window_seconds=0.04, fs=fs)
    systolic_indices, prominences = find_prominent_peaks(
        smoothed, fs=fs, min_rr=min_rr, max_rr=max_rr
    )

    beats: List[Beat] = []
    for idx, sys_idx in enumerate(systolic_indices):
        start = systolic_indices[idx - 1] if idx > 0 else 0
        end = systolic_indices[idx + 1] if idx + 1 < len(systolic_indices) else len(smoothed) - 1

        search_start = max(start, sys_idx - int(max_rr * fs))
        search_end = min(end, sys_idx + int(max_rr * fs))
        segment = smoothed[search_start:sys_idx + 1]
        if segment.size == 0:
            continue
        dia_rel = np.argmin(segment)
        dia_idx = search_start + int(dia_rel)

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


def apply_artifact_rules(beats: List[Beat]) -> None:
    """Flag beats that violate simple physiological heuristics."""

    if not beats:
        return

    systolic_values = np.array([b.systolic_pressure for b in beats])
    diastolic_values = np.array([b.diastolic_pressure for b in beats])
    rr_values = np.array([b.rr_interval for b in beats if b.rr_interval is not None])

    sys_median = float(np.nanmedian(systolic_values))
    dia_median = float(np.nanmedian(diastolic_values))
    rr_median = float(np.nanmedian(rr_values)) if rr_values.size else None

    for beat in beats:
        reasons: List[str] = []
        if not math.isfinite(beat.systolic_pressure) or not math.isfinite(beat.diastolic_pressure):
            reasons.append("non-finite pressure")
        if beat.systolic_pressure < 40 or beat.systolic_pressure > 260:
            reasons.append("implausible systolic")
        if beat.diastolic_pressure < 20 or beat.diastolic_pressure > 160:
            reasons.append("implausible diastolic")
        if beat.systolic_pressure - beat.diastolic_pressure < 10:
            reasons.append("low pulse pressure")
        if abs(beat.systolic_pressure - sys_median) > 40:
            reasons.append("systolic jump")
        if abs(beat.diastolic_pressure - dia_median) > 30:
            reasons.append("diastolic jump")
        if beat.rr_interval is not None and rr_median is not None:
            if beat.rr_interval < 0.3 or beat.rr_interval > 2.5:
                reasons.append("rr outside bounds")
            if abs(beat.rr_interval - rr_median) > 0.4:
                reasons.append("rr jump")
        if beat.systolic_prominence < 5:
            reasons.append("low prominence")

        beat.is_artifact = bool(reasons)


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
