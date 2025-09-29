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
import argparse
from dataclasses import dataclass, field
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
    map_time: float
    map_pressure: float
    rr_interval: Optional[float]
    systolic_prominence: float
    is_artifact: bool
    artifact_reasons: List[str] = field(default_factory=list)


@dataclass
class ArtifactConfig:
    """Tunable thresholds used to classify artefactual beats."""

    systolic_mad_multiplier: float = 4.5
    diastolic_mad_multiplier: float = 4.5
    rr_mad_multiplier: float = 3.5
    systolic_abs_floor: float = 15.0
    diastolic_abs_floor: float = 12.0
    pulse_pressure_min: float = 10.0
    rr_bounds: Tuple[float, float] = (0.3, 2.0)
    min_prominence: float = 8.0
    prominence_noise_factor: float = 0.18
    max_missing_gap: float = 10.0


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

    metadata_interval: Optional[float] = None
    inferred_interval: Optional[float] = None

    interval_raw = metadata.get("Interval")
    if interval_raw:
        match = re.search(r"([0-9]+\.?[0-9]*)", interval_raw)
        if match:
            metadata_interval = float(match.group(1))

    # Infer from the time column if available.
    if "Time" in frame.columns:
        time_values = frame["Time"].to_numpy()
        if len(time_values) > 1:
            diffs = np.diff(time_values)
            diffs = diffs[~np.isnan(diffs)]
            if len(diffs) > 0:
                inferred_interval = float(np.median(diffs))

    if metadata_interval and inferred_interval:
        if metadata_interval > 0 and abs(inferred_interval - metadata_interval) / metadata_interval > 0.1:
            print(
                "Warning: metadata interval"
                f" ({metadata_interval:.4f}s) differs from inferred interval"
                f" ({inferred_interval:.4f}s)."
            )
        return metadata_interval

    if metadata_interval:
        return metadata_interval

    if inferred_interval:
        return inferred_interval

    # Fallback to 1 Hz.
    return 1.0


def smooth_signal(signal: np.ndarray, window_seconds: float, fs: float) -> np.ndarray:
    """Apply a simple moving average filter to suppress high-frequency noise."""

    window_samples = max(1, int(round(window_seconds * fs)))
    if window_samples <= 1:
        return signal.astype(float, copy=False)

    kernel = np.ones(window_samples, dtype=float) / float(window_samples)
    return np.convolve(signal, kernel, mode="same")


def zero_phase_filter(
    signal: np.ndarray,
    fs: float,
    *,
    lowcut: float = 0.5,
    highcut: float = 20.0,
    order: int = 4,
) -> np.ndarray:
    """Apply a zero-phase bandpass filter when SciPy is available.

    Falls back to returning the input data when filtering fails or SciPy is
    unavailable. The cutoff range is clipped to the [0, Nyquist) interval to
    avoid numerical errors on very low/high sampling rates.
    """

    if sp_signal is None or fs <= 0:
        return signal

    nyquist = 0.5 * fs
    if nyquist <= 0:
        return signal

    low = max(0.01, lowcut) / nyquist
    high = min(highcut, nyquist * 0.99) / nyquist
    if not (0 < low < high < 1):
        return signal

    try:
        b, a = sp_signal.butter(order, [low, high], btype="bandpass")
        return sp_signal.filtfilt(b, a, signal.astype(float, copy=False))
    except Exception:
        return signal


def detrend_signal(signal: np.ndarray, fs: float, window_seconds: float = 0.6) -> np.ndarray:
    """Remove slow-varying trends using a long moving-average window."""

    if window_seconds <= 0:
        return signal

    baseline = smooth_signal(signal, window_seconds=window_seconds, fs=fs)
    return signal - baseline


def _find_long_gaps(time: np.ndarray, mask: np.ndarray, gap_seconds: float) -> List[Tuple[float, float]]:
    """Return start/stop times for gaps that exceed ``gap_seconds``."""

    if gap_seconds <= 0 or time.size == 0:
        return []

    invalid_spans: List[Tuple[float, float]] = []
    gap_start: Optional[int] = None

    for idx, missing in enumerate(mask):
        if missing and gap_start is None:
            gap_start = idx
        elif not missing and gap_start is not None:
            gap_end = idx - 1
            if gap_end >= gap_start:
                start_time = float(time[gap_start])
                end_time = float(time[min(gap_end, len(time) - 1)])
                if end_time - start_time >= gap_seconds:
                    invalid_spans.append((start_time, end_time))
            gap_start = None

    if gap_start is not None:
        start_time = float(time[gap_start])
        end_time = float(time[-1])
        if end_time - start_time >= gap_seconds:
            invalid_spans.append((start_time, end_time))

    return invalid_spans


def detect_systolic_peaks(
    signal: np.ndarray,
    fs: float,
    *,
    min_rr: float,
    max_rr: float,
    prominence_floor: float,
    prominence_noise_factor: float,
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
    max_search_window = min(0.45, 0.6 * max(min_rr, 0.2))
    search_horizon = max(min_distance, int(round(max_search_window * fs)))

    amplitude_span = float(np.percentile(signal, 95) - np.percentile(signal, 5))
    if signal.size > 1:
        noise_proxy = float(np.median(np.abs(np.diff(signal))))
    else:
        noise_proxy = 0.0
    adaptive_floor = max(amplitude_span * prominence_noise_factor, noise_proxy * 5.0)
    min_prominence = max(prominence_floor, adaptive_floor)

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
            signal,
            fs=fs,
            min_rr=min_rr,
            max_rr=max_rr,
            min_prominence=min_prominence,
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
    *,
    config: ArtifactConfig,
) -> List[Beat]:
    """Derive systolic/diastolic/MAP landmarks from a continuous waveform."""

    if len(time) != len(pressure):
        raise ValueError("Time and pressure arrays must have the same length.")

    finite_mask = np.isfinite(pressure)
    if not np.any(finite_mask):
        raise ValueError("Pressure signal contains no valid samples.")

    pressure_filled = pressure.copy()
    missing_mask = ~finite_mask
    invalid_spans = _find_long_gaps(time, missing_mask, config.max_missing_gap)
    if not np.all(finite_mask):
        pressure_filled[~finite_mask] = np.interp(
            np.flatnonzero(~finite_mask),
            np.flatnonzero(finite_mask),
            pressure[finite_mask],
        )

    min_rr, max_rr = config.rr_bounds

    filtered = zero_phase_filter(pressure_filled, fs=fs)
    smoothed = smooth_signal(filtered, window_seconds=0.03, fs=fs)
    systolic_indices, prominences = detect_systolic_peaks(
        smoothed,
        fs=fs,
        min_rr=min_rr,
        max_rr=max_rr,
        prominence_floor=config.min_prominence,
        prominence_noise_factor=config.prominence_noise_factor,
    )

    beats: List[Beat] = []
    prev_dia_sample: Optional[int] = None

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
        raw_sys = pressure[sys_idx]
        raw_dia = pressure[dia_idx]
        systolic_pressure = float(raw_sys) if math.isfinite(raw_sys) else float(pressure_filled[sys_idx])
        diastolic_time = float(time[dia_idx])
        diastolic_pressure = float(raw_dia) if math.isfinite(raw_dia) else float(pressure_filled[dia_idx])
        map_time = float((systolic_time + diastolic_time) / 2.0)
        area_map = math.nan
        if prev_dia_sample is not None and prev_dia_sample < dia_idx:
            start_idx = prev_dia_sample
            end_idx = dia_idx + 1
            if end_idx - start_idx > 2:
                segment_time = time[start_idx:end_idx]
                segment_pressure = pressure_filled[start_idx:end_idx]
                area = float(np.trapz(segment_pressure, segment_time))
                duration = float(segment_time[-1] - segment_time[0])
                if duration > 0:
                    area_map = area / duration

        if math.isnan(area_map):
            map_pressure = diastolic_pressure + (systolic_pressure - diastolic_pressure) / 3.0
        else:
            map_pressure = area_map
        rr_interval = None
        if idx > 0:
            rr_interval = float(systolic_time - time[systolic_indices[idx - 1]])

        reasons: List[str] = []
        for start_time, end_time in invalid_spans:
            if start_time <= systolic_time <= end_time:
                reasons.append("long gap interpolation")
                break

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
                artifact_reasons=reasons,
            )
        )

        prev_dia_sample = dia_idx

    apply_artifact_rules(beats, config=config)
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


def apply_artifact_rules(beats: List[Beat], *, config: ArtifactConfig) -> None:
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

    min_rr, max_rr = config.rr_bounds

    for idx, beat in enumerate(beats):
        severe_reasons: List[str] = []
        soft_reasons: List[str] = []

        if not math.isfinite(beat.systolic_pressure) or not math.isfinite(beat.diastolic_pressure):
            severe_reasons.append("non-finite pressure")
        if beat.systolic_pressure < 40 or beat.systolic_pressure > 260:
            severe_reasons.append("implausible systolic")
        if beat.diastolic_pressure < 20 or beat.diastolic_pressure > 160:
            severe_reasons.append("implausible diastolic")
        if beat.systolic_pressure - beat.diastolic_pressure < config.pulse_pressure_min:
            severe_reasons.append("low pulse pressure")

        sys_ref = sys_roll_med[idx] if math.isfinite(sys_roll_med[idx]) else sys_med
        sys_scale = sys_roll_mad[idx] if math.isfinite(sys_roll_mad[idx]) else sys_mad
        if math.isfinite(sys_ref) and math.isfinite(beat.systolic_pressure) and math.isfinite(sys_scale):
            limit = max(config.systolic_abs_floor, config.systolic_mad_multiplier * max(sys_scale, 1.0))
            if abs(beat.systolic_pressure - sys_ref) > limit:
                soft_reasons.append("systolic deviation")

        dia_ref = dia_roll_med[idx] if math.isfinite(dia_roll_med[idx]) else dia_med
        dia_scale = dia_roll_mad[idx] if math.isfinite(dia_roll_mad[idx]) else dia_mad
        if math.isfinite(dia_ref) and math.isfinite(beat.diastolic_pressure) and math.isfinite(dia_scale):
            limit = max(config.diastolic_abs_floor, config.diastolic_mad_multiplier * max(dia_scale, 1.0))
            if abs(beat.diastolic_pressure - dia_ref) > limit:
                soft_reasons.append("diastolic deviation")

        if beat.rr_interval is not None and math.isfinite(beat.rr_interval):
            if beat.rr_interval < min_rr * 0.7 or beat.rr_interval > max_rr * 1.3:
                severe_reasons.append("rr outside bounds")
            rr_ref = rr_roll_med[idx] if math.isfinite(rr_roll_med[idx]) else rr_med
            rr_scale = rr_roll_mad[idx] if math.isfinite(rr_roll_mad[idx]) else rr_mad
            if math.isfinite(rr_ref) and math.isfinite(rr_scale):
                limit = max(0.1, config.rr_mad_multiplier * max(rr_scale, 0.05))
                if abs(beat.rr_interval - rr_ref) > limit:
                    soft_reasons.append("rr deviation")

        if beat.systolic_prominence < config.min_prominence:
            soft_reasons.append("low prominence")

        combined_reasons = list(beat.artifact_reasons)
        if any(reason == "long gap interpolation" for reason in beat.artifact_reasons):
            severe_reasons.append("long gap interpolation")
        combined_reasons.extend(severe_reasons)
        combined_reasons.extend(soft_reasons)
        beat.is_artifact = bool(severe_reasons or len(soft_reasons) >= 2)
        beat.artifact_reasons = combined_reasons if beat.is_artifact else []


def plot_waveform(
    time: np.ndarray,
    pressure: np.ndarray,
    beats: Sequence[Beat],
    *,
    show: bool = True,
    save_path: Optional[Path] = None,
) -> None:
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
    return float(np.trapz(power[mask], freqs[mask]))


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


def print_column_overview(frame: pd.DataFrame) -> None:
    """Display column names, dtypes, and first numeric samples to aid selection."""

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
        "--systolic-mad-multiplier",
        type=float,
        default=4.5,
        help="MAD multiplier for systolic outlier detection.",
    )
    parser.add_argument(
        "--diastolic-mad-multiplier",
        type=float,
        default=4.5,
        help="MAD multiplier for diastolic outlier detection.",
    )
    parser.add_argument(
        "--rr-mad-multiplier",
        type=float,
        default=3.5,
        help="MAD multiplier for RR-interval outlier detection.",
    )
    parser.add_argument("--pulse-pressure-min", type=float, default=10.0, help="Minimum pulse pressure (mmHg).")
    parser.add_argument(
        "--systolic-abs-floor",
        type=float,
        default=15.0,
        help="Absolute systolic deviation floor in mmHg.",
    )
    parser.add_argument(
        "--diastolic-abs-floor",
        type=float,
        default=12.0,
        help="Absolute diastolic deviation floor in mmHg.",
    )
    parser.add_argument(
        "--max-gap",
        type=float,
        default=10.0,
        help="Maximum tolerated gap (s) before interpolated beats are flagged.",
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
    parser.add_argument("--out", type=Path, help="Optional path to export the beat summary CSV.")
    parser.add_argument("--savefig", type=Path, help="Save the plot to this path instead of/as well as showing it.")
    parser.add_argument("--no-plot", action="store_true", help="Skip interactive plot display.")
    return parser


def main(argv: Sequence[str]) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv[1:])

    if args.min_rr >= args.max_rr:
        parser.error("--min-rr must be less than --max-rr")

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
        frame, metadata = load_bp_file(file_path)
    except Exception as exc:  # pragma: no cover - interactive feedback
        print(f"Failed to load file: {exc}")
        return 1

    print(f"Loaded file: {file_path}")
    print_column_overview(frame)

    if args.column not in frame.columns:
        print(f"Column '{args.column}' not found. Available columns: {', '.join(frame.columns)}")
        return 1

    if args.time_column in frame.columns:
        time_series = pd.to_numeric(frame[args.time_column], errors="coerce")
        time = time_series.to_numpy()
    else:
        time = None

    pressure_series = pd.to_numeric(frame[args.column], errors="coerce")
    pressure = pressure_series.to_numpy()

    interval = parse_interval(metadata, frame)
    fs = 1.0 / interval if interval > 0 else 1.0

    if time is None or not np.any(np.isfinite(time)):
        time = np.arange(len(frame), dtype=float) * interval

    config = ArtifactConfig(
        systolic_mad_multiplier=args.systolic_mad_multiplier,
        diastolic_mad_multiplier=args.diastolic_mad_multiplier,
        rr_mad_multiplier=args.rr_mad_multiplier,
        systolic_abs_floor=args.systolic_abs_floor,
        diastolic_abs_floor=args.diastolic_abs_floor,
        pulse_pressure_min=args.pulse_pressure_min,
        rr_bounds=(args.min_rr, args.max_rr),
        min_prominence=args.min_prominence,
        prominence_noise_factor=args.prominence_factor,
        max_missing_gap=args.max_gap,
    )

    try:
        beats = derive_beats(time, pressure, fs=fs, config=config)
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
            if np.any(np.diff(beat_times) > config.max_missing_gap):
                print(
                    "PSD skipped because long gaps (>"
                    f"{config.max_missing_gap}s) remain after artefact removal."
                )
            else:
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
        plot_waveform(time, pressure, beats, show=show_plot, save_path=args.savefig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
