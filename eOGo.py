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


def _parse_list_field(raw_value: str) -> List[str]:
    """Split a metadata field that contains multiple values.

    Continuous BP exports frequently separate list-style metadata (such as
    ``ChannelTitle`` entries) using tab characters.  When tabs are not
    available we fall back to splitting on whitespace while attempting to
    preserve multi-word labels.
    """

    raw_value = raw_value.strip()
    if not raw_value:
        return []

    parts = [item.strip() for item in re.split(r"\t+", raw_value) if item.strip()]
    if parts:
        return parts

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


def _parse_bp_header(
    file_path: Path, *, max_data_lines: int = 5
) -> Tuple[Dict[str, str], List[int], List[str], Optional[str]]:
    """Return metadata, skip rows, column names, and a detected separator."""

    metadata: Dict[str, str] = {}
    skiprows: List[int] = []
    column_count: Optional[int] = None
    first_numeric_row_idx: Optional[int] = None
    numeric_samples: List[str] = []
    numeric_lines_checked = 0

    if max_data_lines <= 0:
        max_data_lines = 1

    def _numeric_values(candidate: str) -> np.ndarray:
        tokens = [tok for tok in re.split(r"[\s,;|]+", candidate) if tok]
        if not tokens:
            return np.array([], dtype=float)
        try:
            return np.asarray([float(tok) for tok in tokens], dtype=float)
        except ValueError:
            return np.array([], dtype=float)

    with file_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for idx, raw_line in enumerate(handle):
            stripped = raw_line.strip()

            if first_numeric_row_idx is None:
                if not stripped:
                    skiprows.append(idx)
                    continue

                candidate = stripped
                if "#" in candidate:
                    candidate = candidate.split("#", 1)[0].strip()
                    if not candidate:
                        skiprows.append(idx)
                        continue

                if "=" in candidate and not re.match(r"^[+-]?[0-9]", candidate):
                    key, value = candidate.split("=", 1)
                    metadata[key.strip()] = value.strip()
                    skiprows.append(idx)
                    continue

                values = _numeric_values(candidate)
                if values.size == 0:
                    skiprows.append(idx)
                    continue

                column_count = int(values.size)
                first_numeric_row_idx = idx
                numeric_samples.append(raw_line.rstrip("\n"))
                numeric_lines_checked = 1

                if numeric_lines_checked >= max_data_lines:
                    break
                continue

            if not stripped:
                break

            candidate = stripped
            if "#" in candidate:
                candidate = candidate.split("#", 1)[0].strip()
                if not candidate:
                    break

            values = _numeric_values(candidate)
            if values.size == 0:
                break

            if column_count is None:
                column_count = int(values.size)
            elif int(values.size) != column_count:
                raise ValueError(
                    "Encountered data row with an unexpected column count at "
                    f"line {idx + 1}: expected {column_count}, found {int(values.size)}."
                )

            numeric_samples.append(raw_line.rstrip("\n"))
            numeric_lines_checked += 1
            if numeric_lines_checked >= max_data_lines:
                break

    if column_count is None or first_numeric_row_idx is None:
        raise ValueError("No numeric data found in file.")

    channels = _parse_list_field(metadata.get("ChannelTitle", ""))
    if channels and column_count == len(channels) + 1:
        column_names = ["Time"] + channels
    else:
        column_names = [f"col_{idx}" for idx in range(column_count)]

    detected_separator: Optional[str] = None
    if numeric_samples:
        candidate_lines = numeric_samples

        def _consistently_splits(
            delimiter: str, *, allow_blank_tokens: bool = False
        ) -> bool:
            for line in candidate_lines:
                stripped_line = line.strip()
                parts = stripped_line.split(delimiter)
                if len(parts) != column_count:
                    return False
                if not allow_blank_tokens and any(part == "" for part in parts):
                    return False
            return True

        for delimiter in ("\t", ",", ";", "|"):
            if all(delimiter in line for line in candidate_lines) and _consistently_splits(
                delimiter, allow_blank_tokens=True
            ):
                detected_separator = delimiter
                break

        if detected_separator is None and all("\t" not in line for line in candidate_lines):
            if _consistently_splits(" "):
                detected_separator = " "

    return metadata, skiprows, column_names, detected_separator


def _scan_bp_file(
    file_path: Path,
    *,
    max_numeric_rows: int = 200,
    header_sample_lines: int = 5,
    separator_override: Optional[str] = None,
) -> BPFilePreview:
    """Perform a quick pass to gather metadata and column previews."""

    metadata, skiprows, column_names, detected_separator = _parse_bp_header(
        file_path, max_data_lines=header_sample_lines
    )

    if separator_override:
        detected_separator = separator_override

    preview_frame: Optional[pd.DataFrame] = None
    read_kwargs = dict(
        filepath_or_buffer=file_path,
        comment="#",
        header=None,
        names=column_names,
        skiprows=skiprows,
        nrows=max_numeric_rows,
        na_values=["nan", "NaN", "NA"],
        dtype=float,
    )

    if detected_separator:
        try:
            if len(detected_separator) == 1:
                preview_frame = pd.read_csv(
                    sep=detected_separator,
                    engine="c",
                    **read_kwargs,
                )
            else:
                preview_frame = pd.read_csv(
                    sep=detected_separator,
                    engine="python",
                    **read_kwargs,
                )
        except Exception:
            preview_frame = pd.read_csv(
                delim_whitespace=True,
                engine="python",
                **read_kwargs,
            )
    else:
        preview_frame = pd.read_csv(
            delim_whitespace=True,
            engine="python",
            **read_kwargs,
        )

    raw_lines: List[str] = []
    try:
        with file_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for _ in range(200):
                line = handle.readline()
                if not line:
                    break
                raw_lines.append(line.rstrip("\r\n"))
    except Exception:
        raw_lines = []

    return BPFilePreview(
        metadata=metadata,
        column_names=column_names,
        preview=preview_frame,
        skiprows=skiprows,
        detected_separator=detected_separator,
        raw_lines=raw_lines,
        preview_rows=max(1, max_numeric_rows),
    )


def _tokenise_preview_line(line: str, separator: Optional[str]) -> List[str]:
    """Split a raw preview line using the provided separator."""

    if not line:
        return []

    if separator is None:
        stripped = line.strip()
        if not stripped:
            return []
        return [tok for tok in re.split(r"\s+", stripped) if tok]

    if separator == " ":
        stripped = line.strip()
        if not stripped:
            return []
        return [tok for tok in re.split(r"\s+", stripped) if tok]

    try:
        return next(csv.reader([line], delimiter=separator))
    except Exception:
        return line.split(separator)


def _build_preview_dataframe(
    file_path: Path,
    *,
    column_names: Sequence[str],
    skiprows: Sequence[int],
    separator: Optional[str],
    max_rows: int,
) -> pd.DataFrame:
    """Construct a small DataFrame preview for the selected layout."""

    read_kwargs = dict(
        filepath_or_buffer=file_path,
        header=None,
        names=list(column_names),
        skiprows=list(skiprows),
        nrows=max_rows,
        na_values=["nan", "NaN", "NA"],
    )

    if separator:
        try:
            engine = "c" if len(separator) == 1 else "python"
            return pd.read_csv(sep=separator, engine=engine, **read_kwargs)
        except Exception:
            return pd.read_csv(delim_whitespace=True, engine="python", **read_kwargs)

    return pd.read_csv(delim_whitespace=True, engine="python", **read_kwargs)


def _render_preview_to_text_widget(
    widget: tk.Text,
    preview_rows: Sequence[Sequence[str]],
    *,
    header_row_index: int,
    first_data_row_index: int,
    time_column_index: Optional[int],
    pressure_column_index: Optional[int],
    max_rows: int = 200,
) -> None:
    """Render a preview table into a Tkinter text widget with highlights."""

    if tk is None:
        return

    widget.configure(state="normal")
    widget.delete("1.0", tk.END)

    if not preview_rows:
        widget.insert("end", "No preview data available.\n")
        widget.configure(state="disabled")
        return

    header_row_index = max(1, header_row_index)
    first_data_row_index = max(1, first_data_row_index)

    widths: List[int] = []
    sample_rows = list(preview_rows[:max_rows])
    for row in sample_rows:
        for idx, value in enumerate(row):
            if value is None:
                value = ""
            text_value = str(value)
            if idx >= len(widths):
                widths.append(len(text_value))
            else:
                widths[idx] = max(widths[idx], len(text_value))

    line_no = 1
    for row_number, row_values in enumerate(sample_rows, start=1):
        line_parts: List[str] = []
        positions: List[Tuple[int, int, int]] = []
        cursor = 0
        for col_idx, width in enumerate(widths):
            value = ""
            if col_idx < len(row_values) and row_values[col_idx] is not None:
                value = str(row_values[col_idx])
            padded = value.ljust(width + 2)
            start = cursor
            end = cursor + len(padded)
            positions.append((start, end, col_idx))
            line_parts.append(padded)
            cursor = end

        line_text = "".join(line_parts)
        widget.insert("end", line_text + "\n")

        row_tag = "row_even" if row_number % 2 == 0 else "row_odd"
        widget.tag_add(row_tag, f"{line_no}.0", f"{line_no}.end")

        if row_number == header_row_index:
            widget.tag_add("header", f"{line_no}.0", f"{line_no}.end")
        if row_number == first_data_row_index:
            widget.tag_add("data_start", f"{line_no}.0", f"{line_no}.end")

        for start, end, col_idx in positions:
            if time_column_index is not None and col_idx == time_column_index:
                widget.tag_add("highlight_time", f"{line_no}.{start}", f"{line_no}.{end}")
            if pressure_column_index is not None and col_idx == pressure_column_index:
                widget.tag_add("highlight_pressure", f"{line_no}.{start}", f"{line_no}.{end}")

        line_no += 1

    widget.configure(state="disabled")
    widget.tag_raise("column_index")
    widget.tag_raise("highlight_time")
    widget.tag_raise("highlight_pressure")
    widget.tag_raise("data_start")
    widget.tag_raise("header")


def _default_column_selection(
    column_names: Sequence[str],
    requested: Optional[str],
    *,
    fallback: Optional[str] = None,
) -> str:
    """Return a sensible default column name."""

    if requested and requested in column_names:
        return requested

    if fallback and fallback in column_names:
        return fallback

    if column_names:
        return column_names[0]

    raise ValueError("No columns available for selection.")


def _select_columns_via_cli(
    column_names: Sequence[str],
    *,
    default_time: str,
    default_pressure: str,
) -> Tuple[str, str]:
    """Prompt for time/pressure selection using stdin."""

    if not sys.stdin.isatty():
        return default_time, default_pressure

    print("\nSelect time and pressure columns (press Enter to accept defaults).")
    for idx, name in enumerate(column_names):
        marker = ""
        if name == default_time:
            marker = " (default time)"
        elif name == default_pressure:
            marker = " (default pressure)"
        print(f"  [{idx}] {name}{marker}")

    def _prompt(label: str, default: str) -> str:
        while True:
            response = input(f"{label} [{default}]: ").strip()
            if not response:
                return default
            if response in column_names:
                return response
            if response.isdigit():
                idx = int(response)
                if 0 <= idx < len(column_names):
                    return column_names[idx]
            print("Invalid selection. Choose a column name or index.")

    time_column = _prompt("Time column", default_time)
    while True:
        pressure_column = _prompt("Pressure column", default_pressure)
        if pressure_column == time_column:
            print("Pressure column must differ from the time column.")
        else:
            break

    return time_column, pressure_column


def _select_columns_via_tk(
    column_names: Sequence[str],
    *,
    default_time: str,
    default_pressure: str,
) -> Tuple[str, str]:
    """Display a Tkinter dialog for column selection."""

    if tk is None:
        return default_time, default_pressure

    root = tk.Tk()
    root.title("Select waveform columns")
    root.geometry("360x180")

    time_var = tk.StringVar(value=default_time)
    pressure_var = tk.StringVar(value=default_pressure)
    error_var = tk.StringVar(value="")

    tk.Label(root, text="Time column:").pack(pady=(12, 0))
    tk.OptionMenu(root, time_var, *column_names).pack()

    tk.Label(root, text="Pressure column:").pack(pady=(8, 0))
    tk.OptionMenu(root, pressure_var, *column_names).pack()

    error_label = tk.Label(root, textvariable=error_var, fg="red")
    error_label.pack(pady=(6, 0))

    selection: Dict[str, str] = {}

    def confirm() -> None:
        time_choice = time_var.get()
        pressure_choice = pressure_var.get()
        if time_choice == pressure_choice:
            error_var.set("Time and pressure must differ.")
            return
        selection["time"] = time_choice
        selection["pressure"] = pressure_choice
        root.quit()

    tk.Button(root, text="Confirm", command=confirm).pack(pady=12)

    root.mainloop()
    root.destroy()

    if "time" in selection and "pressure" in selection:
        return selection["time"], selection["pressure"]

    return default_time, default_pressure


def select_time_pressure_columns(
    preview: BPFilePreview,
    *,
    requested_time: Optional[str] = None,
    requested_pressure: Optional[str] = None,
    allow_gui: bool = True,
) -> Tuple[str, str]:
    """Resolve the time and pressure columns, prompting the user when possible."""

    column_names = preview.column_names
    time_default = _default_column_selection(column_names, requested_time, fallback="Time")

    remaining = [name for name in column_names if name != time_default]
    pressure_default = _default_column_selection(
        remaining if remaining else column_names,
        requested_pressure if requested_pressure != time_default else None,
        fallback="reBAP",
    )

    if allow_gui and tk is not None:
        try:
            return _select_columns_via_tk(
                column_names,
                default_time=time_default,
                default_pressure=pressure_default,
            )
        except Exception:
            pass

    return _select_columns_via_cli(
        column_names,
        default_time=time_default,
        default_pressure=pressure_default,
    )


def launch_import_configuration_dialog(
    file_path: Path,
    preview: BPFilePreview,
    *,
    time_default: str,
    pressure_default: str,
    separator_override: Optional[str] = None,
) -> Optional[ImportDialogResult]:
    """Display a dialog allowing the user to adjust delimiter and column selections."""

    if tk is None or ttk is None:
        return None

    delimiter_options: List[Tuple[str, Optional[str]]] = [
        ("Auto-detect", None),
        ("Tab (\t)", "	"),
        ("Comma (,)", ","),
        ("Semicolon (;)", ";"),
        ("Pipe (|)", "|"),
        ("Space", " "),
        ("Colon (:)", ":"),
    ]

    downsample_options: List[Tuple[str, int]] = [
        ("Full resolution (1×)", 1),
        ("Every 2nd sample (2×)", 2),
        ("Every 4th sample (4×)", 4),
        ("Every 5th sample (5×)", 5),
        ("Every 10th sample (10×)", 10),
        ("Every 20th sample (20×)", 20),
        ("Every 50th sample (50×)", 50),
        ("Every 100th sample (100×)", 100),
    ]

    def _label_for_separator(value: Optional[str]) -> str:
        for label, candidate in delimiter_options:
            if value == candidate:
                return label
        return "Auto-detect"

    def _resolve_separator_from_label(label: str) -> Optional[str]:
        for option_label, value in delimiter_options:
            if option_label == label:
                return value
        return None

    def _label_for_downsample(value: int) -> str:
        for label, candidate in downsample_options:
            if value == candidate:
                return label
        return downsample_options[0][0]

    def _resolve_downsample_from_label(label: str) -> Optional[int]:
        for option_label, value in downsample_options:
            if option_label == label:
                return value
        return None

    def _effective_separator(value: Optional[str]) -> Optional[str]:
        if value is not None:
            return value
        return preview.detected_separator

    raw_lines = list(preview.raw_lines)
    if not raw_lines:
        with file_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for _ in range(200):
                line = handle.readline()
                if not line:
                    break
                raw_lines.append(line.rstrip("\r\n"))

    def _split_line(line: str, separator: Optional[str]) -> List[str]:
        return _tokenise_preview_line(line, separator)

    def _split_preview_lines(separator: Optional[str]) -> List[List[str]]:
        effective = _effective_separator(separator)
        return [_split_line(line, effective) for line in raw_lines]

    initial_separator = separator_override if separator_override is not None else preview.detected_separator
    current_rows = _split_preview_lines(initial_separator)

    if preview.skiprows:
        header_default = max(1, max(preview.skiprows) + 1)
    else:
        header_default = 1
    max_available_rows = max(len(raw_lines), 1)
    header_default = min(header_default, max_available_rows)
    first_data_default = header_default + 1 if header_default < max_available_rows else header_default + 1

    default_time_index = 1
    default_pressure_index = 2 if len(preview.column_names) >= 2 else 1
    if preview.column_names:
        if time_default in preview.column_names:
            default_time_index = preview.column_names.index(time_default) + 1
        if pressure_default in preview.column_names:
            default_pressure_index = preview.column_names.index(pressure_default) + 1
        if len(preview.column_names) >= 2:
            for idx in range(1, len(preview.column_names) + 1):
                if idx != default_time_index:
                    default_pressure_index = idx
                    break
        if default_time_index == default_pressure_index:
            default_pressure_index = max(1, min(default_time_index + 1, len(preview.column_names) or 1))

    root = tk.Tk()
    root.title("Import options")
    root.geometry("960x640")

    main_frame = ttk.Frame(root, padding=16)
    main_frame.grid(row=0, column=0, sticky="nsew")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    main_frame.columnconfigure(0, weight=1)
    main_frame.rowconfigure(2, weight=1)

    controls_frame = ttk.Frame(main_frame)
    controls_frame.grid(row=0, column=0, sticky="ew", pady=(0, 12))
    controls_frame.columnconfigure(6, weight=1)

    ttk.Label(controls_frame, text="Delimiter:").grid(row=0, column=0, sticky="w")
    delimiter_var = tk.StringVar(value=_label_for_separator(initial_separator))
    delimiter_combo = ttk.Combobox(
        controls_frame,
        state="readonly",
        values=[label for label, _ in delimiter_options],
        textvariable=delimiter_var,
        width=18,
    )
    delimiter_combo.grid(row=0, column=1, sticky="w", padx=(8, 24))

    ttk.Label(controls_frame, text="Header row:").grid(row=0, column=2, sticky="w")
    header_var = tk.IntVar(value=header_default)
    header_spin = tk.Spinbox(
        controls_frame,
        from_=1,
        to=max(max_available_rows, header_default + 100),
        textvariable=header_var,
        width=8,
    )
    header_spin.grid(row=0, column=3, sticky="w", padx=(8, 24))

    ttk.Label(controls_frame, text="First data row:").grid(row=0, column=4, sticky="w")
    first_data_var = tk.IntVar(value=first_data_default)
    first_data_spin = tk.Spinbox(
        controls_frame,
        from_=1,
        to=max(max_available_rows, first_data_default + 100),
        textvariable=first_data_var,
        width=8,
    )
    first_data_spin.grid(row=0, column=5, sticky="w")

    ttk.Label(controls_frame, text="Time column:").grid(row=1, column=0, sticky="w", pady=(12, 0))
    time_var = tk.StringVar()
    time_combo = ttk.Combobox(
        controls_frame,
        state="readonly",
        textvariable=time_var,
        width=24,
    )
    time_combo.grid(row=1, column=1, sticky="w", padx=(8, 24), pady=(12, 0))

    ttk.Label(controls_frame, text="Pressure column:").grid(row=1, column=2, sticky="w", pady=(12, 0))
    pressure_var = tk.StringVar()
    pressure_combo = ttk.Combobox(
        controls_frame,
        state="readonly",
        textvariable=pressure_var,
        width=24,
    )
    pressure_combo.grid(row=1, column=3, sticky="w", padx=(8, 24), pady=(12, 0))

    comment_enabled_var = tk.BooleanVar(value=False)
    comment_index_var = tk.StringVar(value="")

    def _on_comment_toggle(*_: object) -> None:
        enabled = bool(comment_enabled_var.get())
        state = "normal" if enabled else "disabled"
        comment_entry.configure(state=state)
        if not enabled:
            comment_index_var.set("")
        _update_column_options()
        _update_preview_widget()

    def _on_comment_index_change(*_: object) -> None:
        _update_column_options()
        _update_preview_widget()

    ttk.Checkbutton(
        controls_frame,
        text="Include comment column",
        variable=comment_enabled_var,
        command=_on_comment_toggle,
    ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(12, 0))

    ttk.Label(controls_frame, text="Column #:").grid(row=2, column=2, sticky="w", pady=(12, 0))
    comment_entry = ttk.Entry(
        controls_frame,
        width=8,
        textvariable=comment_index_var,
        state="disabled",
    )
    comment_entry.grid(row=2, column=3, sticky="w", padx=(8, 24), pady=(12, 0))

    ttk.Label(controls_frame, text="Analysis downsampling:").grid(
        row=3, column=0, sticky="w", pady=(12, 0)
    )
    analysis_downsample_var = tk.StringVar(value=_label_for_downsample(1))
    analysis_downsample_combo = ttk.Combobox(
        controls_frame,
        state="readonly",
        values=[label for label, _ in downsample_options],
        textvariable=analysis_downsample_var,
        width=28,
    )
    analysis_downsample_combo.grid(row=3, column=1, sticky="w", padx=(8, 24), pady=(12, 0))

    ttk.Label(controls_frame, text="Plot downsampling:").grid(
        row=3, column=2, sticky="w", pady=(12, 0)
    )
    plot_downsample_var = tk.StringVar(value=_label_for_downsample(10))
    plot_downsample_combo = ttk.Combobox(
        controls_frame,
        state="readonly",
        values=[label for label, _ in downsample_options],
        textvariable=plot_downsample_var,
        width=28,
    )
    plot_downsample_combo.grid(row=3, column=3, sticky="w", padx=(8, 24), pady=(12, 0))

    initial_comment_index: Optional[int] = None
    for idx, name in enumerate(preview.column_names, start=1):
        if str(name).strip().lower() == "comment":
            initial_comment_index = idx
            break
    if initial_comment_index is not None:
        comment_enabled_var.set(True)
        comment_entry.configure(state="normal")
        comment_index_var.set(str(initial_comment_index))

    ttk.Label(
        main_frame,
        text="Data preview (first 200 rows)",
        font=("TkDefaultFont", 10, "bold"),
    ).grid(row=1, column=0, sticky="w")

    preview_frame = ttk.Frame(main_frame, relief=tk.SOLID, borderwidth=1)
    preview_frame.grid(row=2, column=0, sticky="nsew")
    preview_frame.columnconfigure(0, weight=1)
    preview_frame.rowconfigure(0, weight=1)

    preview_text = tk.Text(
        preview_frame,
        wrap="none",
        font=("Courier New", 10),
        height=20,
    )
    preview_text.grid(row=0, column=0, sticky="nsew")
    preview_text.configure(cursor="arrow")
    preview_text.bind("<Key>", lambda _: "break")

    y_scroll = ttk.Scrollbar(preview_frame, orient="vertical", command=preview_text.yview)
    y_scroll.grid(row=0, column=1, sticky="ns")
    x_scroll = ttk.Scrollbar(preview_frame, orient="horizontal", command=preview_text.xview)
    x_scroll.grid(row=1, column=0, sticky="ew")
    preview_text.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)

    preview_text.tag_configure("column_index", background="#ccd6f6", font=("Courier New", 10, "bold"))
    preview_text.tag_configure("header", background="#d9d9d9", font=("Courier New", 10, "bold"))
    preview_text.tag_configure("row_even", background="#f5f5f5")
    preview_text.tag_configure("row_odd", background="#ededed")
    preview_text.tag_configure("highlight_time", background="#dbeafe")
    preview_text.tag_configure("highlight_pressure", background="#fdeac5")
    preview_text.tag_configure("data_start", background="#e7f5e4")

    error_var = tk.StringVar(value="")
    error_label = ttk.Label(main_frame, textvariable=error_var, foreground="red")
    error_label.grid(row=3, column=0, sticky="w", pady=(8, 0))

    button_frame = ttk.Frame(main_frame)
    button_frame.grid(row=4, column=0, sticky="e", pady=(16, 0))

    result: Dict[str, object] = {}
    current_preview_state: Dict[str, object] = {
        "preview": preview,
        "separator": initial_separator,
        "resolved_separator": _effective_separator(initial_separator),
        "rows": current_rows,
        "header_row": header_default,
        "first_data_row": first_data_default,
        "preview_rows": preview.preview_rows,
    }

    column_option_map: Dict[str, int] = {}

    def _current_preview() -> BPFilePreview:
        stored = current_preview_state.get("preview")
        if isinstance(stored, BPFilePreview):
            return stored
        return preview

    def _current_rows() -> List[List[str]]:
        rows = current_preview_state.get("rows")
        if isinstance(rows, list):
            return rows
        return current_rows

    def _parse_column_index(value: str) -> Optional[int]:
        if not value:
            return None
        if value in column_option_map:
            return column_option_map[value]
        prefix = value.split(":", 1)[0].strip()
        if prefix.isdigit():
            return int(prefix)
        return None

    def _parse_comment_index() -> Optional[int]:
        if not comment_enabled_var.get():
            return None
        raw = comment_index_var.get().strip()
        if not raw:
            return None
        if not raw.isdigit():
            return None
        value = int(raw)
        if value <= 0:
            return None
        return value

    def _get_header_row() -> int:
        try:
            value = int(header_var.get())
        except (tk.TclError, ValueError):
            value = header_default
        return max(1, value)

    def _get_first_data_row() -> int:
        try:
            value = int(first_data_var.get())
        except (tk.TclError, ValueError):
            value = first_data_default
        return max(1, value)

    def _build_column_options() -> List[Tuple[str, int]]:
        rows = _current_rows()
        header_index = _get_header_row()
        if rows:
            header_index = min(header_index, len(rows))
        header_values: Sequence[str] = []
        if 1 <= header_index <= len(rows):
            header_values = rows[header_index - 1]
        sample_rows = rows[:50]
        column_count = max((len(row) for row in sample_rows), default=len(preview.column_names))
        column_count = max(column_count, len(header_values))
        comment_index = _parse_comment_index()
        if comment_index:
            column_count = max(column_count, comment_index)
        if column_count == 0:
            column_count = len(preview.column_names) or 1
        options: List[Tuple[str, int]] = []
        for idx in range(column_count):
            header_label = ""
            if idx < len(header_values):
                raw_value = header_values[idx]
                header_label = str(raw_value).strip() if raw_value is not None else ""
            if not header_label:
                if comment_index and (idx + 1) == comment_index:
                    header_label = "Comment"
                else:
                    header_label = f"Column {idx + 1}"
            options.append((f"{idx + 1}: {header_label}", idx + 1))
        return options

    def _select_label_for_index(options: Sequence[Tuple[str, int]], desired_index: int) -> str:
        for label, idx in options:
            if idx == desired_index:
                return label
        return options[0][0] if options else ""

    def _update_column_options(initial: bool = False) -> None:
        nonlocal column_option_map
        options = _build_column_options()
        column_option_map = {label: idx for label, idx in options}
        labels = [label for label, _ in options]
        time_combo.configure(values=labels)
        pressure_combo.configure(values=labels)

        if not options:
            time_var.set("")
            pressure_var.set("")
            return

        current_time_index = _parse_column_index(time_var.get()) or default_time_index
        current_pressure_index = _parse_column_index(pressure_var.get()) or default_pressure_index

        if initial:
            current_time_index = default_time_index
            current_pressure_index = default_pressure_index

        current_time_index = max(1, min(current_time_index, options[-1][1]))
        current_pressure_index = max(1, min(current_pressure_index, options[-1][1]))

        if current_pressure_index == current_time_index and len(options) > 1:
            for _, idx in options:
                if idx != current_time_index:
                    current_pressure_index = idx
                    break

        time_var.set(_select_label_for_index(options, current_time_index))
        pressure_var.set(_select_label_for_index(options, current_pressure_index))

    def _get_time_index() -> Optional[int]:
        return _parse_column_index(time_var.get())

    def _get_pressure_index() -> Optional[int]:
        return _parse_column_index(pressure_var.get())

    def _get_comment_index() -> Optional[int]:
        return _parse_comment_index()

    def _update_preview_widget() -> None:
        rows = _current_rows()
        header_index = _get_header_row()
        data_index = _get_first_data_row()
        time_index = _get_time_index()
        pressure_index = _get_pressure_index()
        preview_limit = current_preview_state.get("preview_rows")
        if not isinstance(preview_limit, int) or preview_limit <= 0:
            preview_limit = _current_preview().preview_rows
        _render_preview_to_text_widget(
            preview_text,
            rows,
            header_row_index=header_index,
            first_data_row_index=data_index,
            time_column_index=(time_index - 1) if time_index else None,
            pressure_column_index=(pressure_index - 1) if pressure_index else None,
            max_rows=preview_limit,
        )

    def _refresh_preview_from_separator(*_: object) -> None:
        selection = delimiter_var.get()
        desired_separator = _resolve_separator_from_label(selection)
        rows = _split_preview_lines(desired_separator)
        current_preview_state["rows"] = rows
        current_preview_state["separator"] = desired_separator
        current_preview_state["resolved_separator"] = _effective_separator(desired_separator)
        _update_column_options()
        _update_preview_widget()

    def _on_structure_change(*_: object) -> None:
        error_var.set("")
        current_preview_state["header_row"] = _get_header_row()
        current_preview_state["first_data_row"] = _get_first_data_row()
        _update_column_options()
        _update_preview_widget()

    def _read_tokens_for_line(line_number: int, separator: Optional[str]) -> List[str]:
        if line_number <= len(raw_lines):
            return _split_line(raw_lines[line_number - 1], separator)
        with file_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for idx, raw_line in enumerate(handle, start=1):
                if idx == line_number:
                    return _split_line(raw_line.rstrip("\r\n"), separator)
        return []

    def _derive_column_names(
        header_tokens: Sequence[str],
        column_count: int,
        *,
        comment_index: Optional[int] = None,
    ) -> List[str]:
        names: List[str] = []
        seen: Dict[str, int] = {}
        for idx in range(column_count):
            raw_value = header_tokens[idx] if idx < len(header_tokens) else ""
            candidate = str(raw_value).strip() if raw_value is not None else ""
            if not candidate:
                if comment_index and (idx + 1) == comment_index:
                    candidate = "Comment"
                else:
                    candidate = f"column_{idx + 1}"
            base = candidate
            suffix = 1
            while candidate in seen:
                suffix += 1
                candidate = f"{base}_{suffix}"
            seen[candidate] = suffix
            names.append(candidate)
        return names

    def _confirm() -> None:
        header_index = _get_header_row()
        data_index = _get_first_data_row()
        time_index = _get_time_index()
        pressure_index = _get_pressure_index()
        comment_index = _get_comment_index()

        if time_index is None or pressure_index is None:
            error_var.set("Select both time and pressure columns.")
            return
        if time_index == pressure_index:
            error_var.set("Time and pressure selections must differ.")
            return
        if comment_enabled_var.get() and comment_index is None:
            error_var.set("Enter a valid comment column number.")
            return
        if data_index <= header_index:
            error_var.set("First data row must be after the header row.")
            return

        rows = _current_rows()
        resolved_candidate = current_preview_state.get("resolved_separator")
        if isinstance(resolved_candidate, str) or resolved_candidate is None:
            resolved_separator = resolved_candidate
        else:
            resolved_separator = _effective_separator(current_preview_state.get("separator"))

        header_tokens = _read_tokens_for_line(header_index, resolved_separator)
        sample_rows = rows[:50]
        column_count = max((len(row) for row in sample_rows), default=0)
        column_count = max(column_count, len(header_tokens), time_index, pressure_index)
        if comment_index:
            column_count = max(column_count, comment_index)
        if column_count <= 0:
            column_count = max(time_index, pressure_index)

        downsample_label = analysis_downsample_var.get()
        downsample_value = _resolve_downsample_from_label(downsample_label)
        if downsample_value is None or downsample_value <= 0:
            error_var.set("Select a valid analysis downsampling factor.")
            return

        plot_downsample_label = plot_downsample_var.get()
        plot_downsample_value = _resolve_downsample_from_label(plot_downsample_label)
        if plot_downsample_value is None or plot_downsample_value <= 0:
            error_var.set("Select a valid plot downsampling factor.")
            return

        column_names = _derive_column_names(
            header_tokens,
            column_count,
            comment_index=comment_index,
        )
        if time_index > len(column_names) or pressure_index > len(column_names):
            error_var.set("Selected columns exceed detected column count.")
            return
        if comment_index and comment_index > len(column_names):
            error_var.set("Comment column exceeds detected column count.")
            return

        skiprows = set(preview.skiprows)
        skiprows.update(range(max(0, data_index - 1)))
        skiprows_list = sorted(skiprows)

        preview_row_limit = current_preview_state.get("preview_rows")
        if isinstance(preview_row_limit, int) and preview_row_limit > 0:
            max_rows = preview_row_limit
        else:
            max_rows = preview.preview_rows

        try:
            preview_df = _build_preview_dataframe(
                file_path,
                column_names=column_names,
                skiprows=skiprows_list,
                separator=resolved_separator,
                max_rows=max_rows,
            )
        except Exception as exc:  # pragma: no cover - interactive feedback
            error_var.set(f"Failed to build preview: {exc}")
            return

        updated_preview = BPFilePreview(
            metadata=preview.metadata,
            column_names=column_names,
            preview=preview_df,
            skiprows=skiprows_list,
            detected_separator=resolved_separator,
            raw_lines=raw_lines,
            preview_rows=max_rows,
        )

        current_preview_state["preview"] = updated_preview
        current_preview_state["header_row"] = header_index
        current_preview_state["first_data_row"] = data_index
        current_preview_state["preview_rows"] = max_rows

        result["time"] = column_names[time_index - 1]
        result["pressure"] = column_names[pressure_index - 1]
        result["time_index"] = time_index
        result["pressure_index"] = pressure_index
        if comment_index:
            result["comment_index"] = comment_index
            result["comment_name"] = column_names[comment_index - 1]
        else:
            result["comment_index"] = None
            result["comment_name"] = None
        result["analysis_downsample"] = downsample_value
        result["plot_downsample"] = plot_downsample_value
        result["separator"] = current_preview_state.get("separator")
        result["preview"] = updated_preview
        result["header_row"] = header_index
        result["first_data_row"] = data_index
        error_var.set("")
        root.quit()

    def _cancel() -> None:
        result.clear()
        root.quit()

    _update_column_options(initial=True)
    _update_preview_widget()

    delimiter_var.trace_add("write", _refresh_preview_from_separator)
    header_var.trace_add("write", _on_structure_change)
    first_data_var.trace_add("write", _on_structure_change)
    time_var.trace_add("write", lambda *_: _update_preview_widget())
    pressure_var.trace_add("write", lambda *_: _update_preview_widget())
    comment_index_var.trace_add("write", _on_comment_index_change)

    ttk.Button(button_frame, text="Cancel", command=_cancel).grid(row=0, column=0, padx=(0, 8))
    ttk.Button(button_frame, text="Import", command=_confirm).grid(row=0, column=1)

    root.protocol("WM_DELETE_WINDOW", _cancel)

    root.mainloop()
    root.destroy()

    required_keys = {
        "time",
        "pressure",
        "preview",
        "separator",
        "header_row",
        "first_data_row",
        "time_index",
        "pressure_index",
        "analysis_downsample",
        "plot_downsample",
    }
    if required_keys.issubset(result.keys()):
        return ImportDialogResult(
            preview=result["preview"],
            separator=result.get("separator"),
            time_column=str(result["time"]),
            pressure_column=str(result["pressure"]),
            header_row=int(result["header_row"]),
            first_data_row=int(result["first_data_row"]),
            time_column_index=int(result["time_index"]),
            pressure_column_index=int(result["pressure_index"]),
            comment_column=(
                str(result["comment_name"])
                if result.get("comment_name") not in (None, "")
                else None
            ),
            comment_column_index=(
                int(result["comment_index"])
                if isinstance(result.get("comment_index"), int)
                else None
            ),
            analysis_downsample=int(result["analysis_downsample"]),
            plot_downsample=int(result["plot_downsample"]),
        )

    return None


def load_bp_file(
    file_path: Path,
    *,
    preview: Optional[BPFilePreview] = None,
    time_column: Optional[str] = None,
    pressure_column: Optional[str] = None,
    separator: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Load the selected columns from a waveform export."""

    if preview is None:
        preview = _scan_bp_file(file_path, separator_override=separator)
    elif separator:
        preview.detected_separator = separator

    if (
        time_column is None
        or time_column not in preview.column_names
        or pressure_column is None
        or pressure_column not in preview.column_names
    ):
        time_column, pressure_column = select_time_pressure_columns(
            preview,
            requested_time=time_column,
            requested_pressure=pressure_column,
            allow_gui=False,
        )

    desired = [time_column, pressure_column]
    usecols = list(dict.fromkeys(desired))
    dtype_map = {name: np.float32 for name in usecols}

    read_kwargs = dict(
        filepath_or_buffer=file_path,
        comment="#",
        header=None,
        names=preview.column_names,
        usecols=usecols,
        skiprows=preview.skiprows,
        na_values=["nan", "NaN", "NA"],
        dtype=dtype_map,
    )

    detected_separator = preview.detected_separator
    frame: pd.DataFrame
    if detected_separator:
        try:
            if len(detected_separator) == 1:
                frame = pd.read_csv(
                    sep=detected_separator,
                    engine="c",
                    **read_kwargs,
                )
            else:
                frame = pd.read_csv(
                    sep=detected_separator,
                    engine="python",
                    **read_kwargs,
                )
        except Exception:
            frame = pd.read_csv(
                delim_whitespace=True,
                engine="python",
                **read_kwargs,
            )
    else:
        frame = pd.read_csv(
            delim_whitespace=True,
            engine="python",
            **read_kwargs,
        )

    return frame, preview.metadata


def parse_interval(metadata: Dict[str, str], frame: pd.DataFrame) -> float:
    """Extract the sampling interval from metadata or infer it from the data."""

    metadata_interval: Optional[float] = None
    inferred_interval: Optional[float] = None

    interval_raw = metadata.get("Interval")
    if interval_raw:
        match = re.search(r"([0-9]+\.?[0-9]*)", interval_raw)
        if match:
            metadata_interval = float(match.group(1))

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

    return 1.0


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