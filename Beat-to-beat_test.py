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
reading whitespace-heavy exports (``--separator``) and limiting the
number of raw samples rendered in the interactive plot
(``--plot-max-points``) to speed up large-file analysis.
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
    from tkinter import filedialog, messagebox, ttk
except Exception:  # pragma: no cover - tkinter may be unavailable in some envs.
    tk = None
    filedialog = None
    messagebox = None
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

try:  # pragma: no cover - optional acceleration for rolling statistics
    from numpy.lib.stride_tricks import sliding_window_view
except Exception:  # pragma: no cover - fallback when unavailable
    sliding_window_view = None  # type: ignore[assignment]

try:
    import pandas as pd
    from pandas import DataFrame
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
    raise SystemExit("pandas is required to run this script. Please install it and retry.") from exc

try:  # pragma: no cover - optional fast moving-window statistics
    import bottleneck as bn
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    bn = None  # type: ignore[assignment]

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
    is_post_calibration_recovery: bool = False


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


@dataclass
class BPFilePreview:
    """Lightweight summary of an export gathered without full ingestion."""

    metadata: Dict[str, str]
    column_names: List[str]
    preview: Optional[DataFrame]
    skiprows: List[int]
    detected_separator: Optional[str]
    raw_lines: List[str]
    preview_rows: int = 50


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


class FastParserError(RuntimeError):
    """Raised when the fast column loader fails to parse the file."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


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


def _parse_bp_header(
    file_path: Path, *, max_data_lines: int = 5
) -> Tuple[Dict[str, str], List[int], List[str], Optional[str]]:
    """Return metadata, line numbers to skip, and inferred column names.

    The parser treats blank lines, comments, and ``key=value`` metadata entries
    as non-numeric content.  All remaining rows are expected to contain the
    same number of whitespace-delimited values; inconsistent rows raise a
    ``ValueError`` instead of being silently truncated.
    """

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

            # Once numeric data begins, avoid adding more skip rows and only
            # validate a limited sample for column consistency.
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

        def _consistently_splits(delimiter: str, *, allow_blank_tokens: bool = False) -> bool:
            for line in candidate_lines:
                stripped_line = line.strip()
                parts = stripped_line.split(delimiter)
                if len(parts) != column_count:
                    return False
                if not allow_blank_tokens and any(part == "" for part in parts):
                    return False
            return True

        for delimiter in ("\t", ",", ";", "|"):
            if all(delimiter in line for line in candidate_lines) and _consistently_splits(delimiter, allow_blank_tokens=True):
                detected_separator = delimiter
                break

        if detected_separator is None and all("\t" not in line for line in candidate_lines):
            if _consistently_splits(" "):
                detected_separator = " "

    return metadata, skiprows, column_names, detected_separator


def _scan_bp_file(
    file_path: Path,
    max_numeric_rows: int = 50,
    *,
    header_sample_lines: int = 5,
    separator_override: Optional[str] = None,
) -> BPFilePreview:
    """Perform a quick pass to gather metadata and column previews."""

    metadata, skiprows, column_names, detected_separator = _parse_bp_header(
        file_path, max_data_lines=header_sample_lines
    )

    if separator_override:
        detected_separator = separator_override

    preview_line_cap = max(max_numeric_rows, 200)
    raw_lines: List[str] = []
    with file_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for _ in range(preview_line_cap):
            line = handle.readline()
            if not line:
                break
            raw_lines.append(line.rstrip("\r\n"))

    return BPFilePreview(
        metadata=metadata,
        column_names=column_names,
        preview=None,
        skiprows=skiprows,
        detected_separator=detected_separator,
        raw_lines=raw_lines,
        preview_rows=max_numeric_rows,
    )


def _build_preview_dataframe(
    file_path: Path,
    *,
    column_names: Sequence[str],
    skiprows: Sequence[int],
    separator: Optional[str],
    max_rows: int,
) -> DataFrame:
    """Construct a small DataFrame preview for the selected layout."""

    read_kwargs = dict(
        filepath_or_buffer=file_path,
        header=None,
        names=list(column_names),
        skiprows=list(skiprows),
        nrows=max_rows,
        na_values=["nan", "NaN", "NA"],
        dtype=float,
    )

    if separator:
        try:
            engine = "c" if len(separator) == 1 else "python"
            return pd.read_csv(sep=separator, engine=engine, **read_kwargs)
        except Exception:
            return pd.read_csv(delim_whitespace=True, engine="python", **read_kwargs)

    return pd.read_csv(delim_whitespace=True, engine="python", **read_kwargs)


def _default_column_selection(column_names: Sequence[str], requested: Optional[str], *, fallback: Optional[str] = None) -> str:
    """Return a sensible default column name."""

    if requested and requested in column_names:
        return requested

    if fallback and fallback in column_names:
        return fallback

    if column_names:
        return column_names[0]

    raise ValueError("No columns available for selection.")


def _render_preview_to_text_widget(
    widget: tk.Text,
    preview_rows: Sequence[Sequence[str]],
    *,
    header_row_index: int,
    first_data_row_index: int,
    time_column_index: Optional[int],
    pressure_column_index: Optional[int],
    max_rows: int = 50,
) -> None:
    """Render a preview table into a Tkinter text widget with highlights."""

    if tk is None:
        return

    widget.configure(state="normal")
    widget.delete("1.0", "end")

    if not preview_rows:
        widget.insert("1.0", "No preview data available.")
        widget.configure(state="disabled")
        return

    total_rows = len(preview_rows)
    display_rows = list(preview_rows[:max_rows])
    column_count = max((len(row) for row in display_rows), default=0)
    row_number_width = max(3, len(str(min(total_rows, max_rows)))) + 1

    widths: List[int] = []
    for col_idx in range(column_count):
        col_values = [
            str(row[col_idx]) if col_idx < len(row) and row[col_idx] is not None else ""
            for row in display_rows
        ]
        widths.append(max((len(value) for value in col_values), default=0))

    line_no = 1
    for row_number, row_values in enumerate(display_rows, start=1):
        line_parts: List[str] = [f"{row_number:>{row_number_width}} "]
        cursor = len(line_parts[0])
        positions: List[Tuple[int, int, int]] = []

        for col_idx in range(column_count):
            value = ""
            if col_idx < len(row_values) and row_values[col_idx] is not None:
                value = str(row_values[col_idx])
            padded = value.ljust(widths[col_idx] + 2)
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
    widget.tag_raise("highlight_time")
    widget.tag_raise("highlight_pressure")
    widget.tag_raise("data_start")
    widget.tag_raise("header")


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
        ("Tab (\\t)", "\t"),
        ("Comma (,)", ","),
        ("Semicolon (;)", ";"),
        ("Pipe (|)", "|"),
        ("Space", " "),
        ("Colon (:)", ":"),
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
        if not line:
            return []
        if separator is None:
            return [tok for tok in re.split(r"\s+", line.strip()) if tok]
        if separator == " ":
            stripped = line.strip()
            if not stripped:
                return []
            return [tok for tok in re.split(r"\s+", stripped) if tok]
        try:
            return next(csv.reader([line], delimiter=separator))
        except Exception:
            return line.split(separator)

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
    if default_pressure_index == default_time_index:
        if len(preview.column_names) >= 2:
            for idx in range(1, len(preview.column_names) + 1):
                if idx != default_time_index:
                    default_pressure_index = idx
                    break
        if default_pressure_index == default_time_index:
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

    ttk.Label(
        main_frame,
        text="Data preview (first 50 rows)",
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
        if column_count == 0:
            column_count = len(preview.column_names) or 1
        options: List[Tuple[str, int]] = []
        for idx in range(column_count):
            header_label = ""
            if idx < len(header_values):
                raw_value = header_values[idx]
                header_label = str(raw_value).strip() if raw_value is not None else ""
            if not header_label:
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

    def _update_preview_widget() -> None:
        rows = _current_rows()
        header_index = _get_header_row()
        data_index = _get_first_data_row()
        time_index = _get_time_index()
        pressure_index = _get_pressure_index()
        _render_preview_to_text_widget(
            preview_text,
            rows,
            header_row_index=header_index,
            first_data_row_index=data_index,
            time_column_index=(time_index - 1) if time_index else None,
            pressure_column_index=(pressure_index - 1) if pressure_index else None,
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

    def _derive_column_names(header_tokens: Sequence[str], column_count: int) -> List[str]:
        names: List[str] = []
        seen: Dict[str, int] = {}
        for idx in range(column_count):
            raw_value = header_tokens[idx] if idx < len(header_tokens) else ""
            candidate = str(raw_value).strip() if raw_value is not None else ""
            if not candidate:
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

        if time_index is None or pressure_index is None:
            error_var.set("Select both time and pressure columns.")
            return
        if time_index == pressure_index:
            error_var.set("Time and pressure selections must differ.")
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
        if column_count <= 0:
            column_count = max(time_index, pressure_index)

        column_names = _derive_column_names(header_tokens, column_count)
        if time_index > len(column_names) or pressure_index > len(column_names):
            error_var.set("Selected columns exceed detected column count.")
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
        )

    return None


def _prompt_fast_parser_fallback(error_message: str) -> bool:
    """Ask the user whether to proceed with the slower parser after failure."""

    message = (
        "The fast parser was unable to load the selected columns.\n\n"
        f"Reason:\n{error_message.strip()}\n\n"
        "Would you like to continue with the slower parser?"
    )

    if tk is not None and messagebox is not None:
        dialog_root: Optional[tk.Tk] = None
        try:
            dialog_root = tk.Tk()
            dialog_root.withdraw()
            result = messagebox.askyesno(
                "Fast import failed",
                message,
                parent=dialog_root,
            )
            return bool(result)
        except Exception:
            pass
        finally:
            if dialog_root is not None:
                dialog_root.destroy()

    print(
        "Fast parser failed. Falling back to the slower parser. "
        f"Reason: {error_message.strip()}"
    )
    return True


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

    return _select_columns_via_cli(column_names, default_time=time_default, default_pressure=pressure_default)


def _load_selected_columns_fast(
    file_path: Path,
    *,
    column_indices: Sequence[int],
    column_names: Sequence[str],
    first_data_row: int,
    separator: Optional[str],
) -> DataFrame:
    """Attempt to load numeric columns using NumPy's fast text parsers."""

    if not column_indices:
        raise FastParserError("No column indices were provided for fast parsing.")

    delimiter: Optional[str]
    if separator is None or separator.isspace():
        delimiter = None
    else:
        if len(separator) != 1:
            raise FastParserError(
                "Fast parser only supports single-character delimiters "
                f"(received '{separator}')."
            )
        delimiter = separator

    skiprows = max(0, first_data_row - 1)
    usecols = tuple(column_indices)

    errors: List[str] = []
    data: Optional[np.ndarray]

    try:
        data = np.loadtxt(
            file_path,
            delimiter=delimiter,
            comments="#",
            usecols=usecols,
            skiprows=skiprows,
            dtype=np.float32,
            ndmin=2,
            encoding="utf-8",
            invalid_raise=False,
        )
    except Exception as exc:
        errors.append(f"np.loadtxt failed: {exc}")
        data = None

    if data is None:
        try:
            data = np.genfromtxt(
                file_path,
                delimiter=delimiter,
                comments="#",
                usecols=usecols,
                skip_header=skiprows,
                dtype=np.float32,
                invalid_raise=False,
                filling_values=np.nan,
                encoding="utf-8",
            )
        except Exception as exc:
            errors.append(f"np.genfromtxt failed: {exc}")
            raise FastParserError("\n".join(errors)) from exc

        if isinstance(data, np.ma.MaskedArray):
            data = data.filled(np.nan)

        data = np.asarray(data, dtype=np.float32)

    if data.size == 0:
        return pd.DataFrame(columns=[column_names[idx] for idx in usecols])

    if data.ndim == 1:
        data = data.reshape(-1, len(usecols))

    frame = pd.DataFrame(data, columns=[column_names[idx] for idx in usecols])
    if not frame.empty:
        frame = frame.dropna(how="all")
    return frame


def load_bp_file(
    file_path: Path,
    *,
    preview: Optional[BPFilePreview] = None,
    time_column: Optional[str] = None,
    pressure_column: Optional[str] = None,
    separator: Optional[str] = None,
    import_settings: Optional[ImportDialogResult] = None,
) -> Tuple[DataFrame, Dict[str, str]]:
    """Load the selected columns from a waveform export."""

    if preview is None:
        preview = _scan_bp_file(file_path, separator_override=separator)
    elif separator:
        preview.detected_separator = separator

    if time_column is None or time_column not in preview.column_names or pressure_column is None or pressure_column not in preview.column_names:
        time_column, pressure_column = select_time_pressure_columns(
            preview,
            requested_time=time_column,
            requested_pressure=pressure_column,
        )

    desired = [time_column, pressure_column]
    # ``dict.fromkeys`` preserves order while removing duplicates.
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
    resolved_separator = separator if separator is not None else detected_separator

    frame: Optional[DataFrame] = None
    fast_failure_reason: Optional[str] = None
    if import_settings is not None:
        index_lookup = {
            import_settings.time_column: import_settings.time_column_index - 1,
            import_settings.pressure_column: import_settings.pressure_column_index - 1,
        }
        fast_usecols: List[int] = []
        for column_name in usecols:
            idx = index_lookup.get(column_name)
            if idx is None:
                fast_usecols = []
                break
            fast_usecols.append(idx)

        if fast_usecols:
            try:
                frame = _load_selected_columns_fast(
                    file_path,
                    column_indices=fast_usecols,
                    column_names=preview.column_names,
                    first_data_row=import_settings.first_data_row,
                    separator=resolved_separator,
                )
            except FastParserError as exc:
                fast_failure_reason = exc.message

    if frame is not None:
        return frame, preview.metadata

    if fast_failure_reason:
        if not _prompt_fast_parser_fallback(fast_failure_reason):
            raise RuntimeError(
                "Import cancelled after fast parser failure."
            )

    if resolved_separator:
        try:
            if len(resolved_separator) == 1:
                frame = pd.read_csv(
                    sep=resolved_separator,
                    engine="c",
                    **read_kwargs,
                )
            else:
                frame = pd.read_csv(
                    sep=resolved_separator,
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


def parse_interval(metadata: Dict[str, str], frame: DataFrame) -> float:
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    """Detect systolic peaks using adaptive upstroke and prominence checks.

    The Lab chart module applies a two-step process: first it emphasises the
    rapid systolic upstroke and then it validates peaks based on their
    prominence relative to neighbouring troughs.  To emulate that behaviour we
    detrend the signal, analyse the velocity profile, and adaptively gate the
    candidate peaks before evaluating their prominences.  Peak candidates are
    generated in a vectorised fashion (using :func:`scipy.signal.find_peaks`
    when available) to reduce Python-level looping while preserving the legacy
    gating heuristics.
    """

    if signal.size == 0:
        return (
            np.array([], dtype=int),
            np.array([], dtype=float),
            np.array([], dtype=int),
            [],
        )

    detrended = detrend_signal(signal, fs, window_seconds=0.6)
    velocity = np.gradient(detrended)
    velocity = smooth_signal(velocity, window_seconds=0.03, fs=fs)
    positive_velocity = np.clip(velocity, a_min=0.0, a_max=None)

    if not np.any(positive_velocity > 0):
        return (
            np.array([], dtype=int),
            np.array([], dtype=float),
            np.array([], dtype=int),
            [],
        )

    positive_samples = positive_velocity[positive_velocity > 0]
    baseline = float(np.median(positive_samples))
    mad = float(np.median(np.abs(positive_samples - baseline)))
    if mad == 0.0:
        mad = float(np.std(positive_samples))
    if mad == 0.0:
        mad = 1.0

    global_threshold = baseline + 2.5 * mad
    if global_threshold <= 0:
        global_threshold = baseline * 1.5 if baseline > 0 else 0.5

    velocity_scale = float(np.nanmedian(np.abs(velocity))) if velocity.size else 0.0
    plateau_eps = max(0.02, 0.3 * velocity_scale)
    plateau_samples = max(1, int(round(0.5 * fs)))
    calibration_segments: List[Tuple[int, int]] = []
    if plateau_samples > 1:
        mask = np.abs(velocity) <= plateau_eps
        idx = 0
        while idx < mask.size:
            if mask[idx]:
                start = idx
                while idx < mask.size and mask[idx]:
                    idx += 1
                if idx - start >= plateau_samples:
                    calibration_segments.append((start, idx))
            else:
                idx += 1

    rolling_window_seconds = 4.0
    default_window = max(3, int(round(rolling_window_seconds * fs)))
    if default_window % 2 == 0:
        default_window += 1

    local_medians = np.full(signal.size, np.nan, dtype=float)
    local_mads = np.full(signal.size, np.nan, dtype=float)

    segment_boundaries: List[Tuple[int, int]] = []
    cursor = 0
    for start, end in calibration_segments:
        if start > cursor:
            segment_boundaries.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < signal.size:
        segment_boundaries.append((cursor, signal.size))

    stride = max(1, int(round(fs / 125.0)))

    for seg_start, seg_end in segment_boundaries:
        seg_len = seg_end - seg_start
        if seg_len < 3:
            continue
        window = min(default_window, seg_len if seg_len % 2 == 1 else seg_len - 1)
        if window < 3:
            continue
        segment = positive_velocity[seg_start:seg_end]
        if stride == 1:
            seg_med, seg_mad = _rolling_median_and_mad(segment, window)
        else:
            reduced_series = segment[::stride]
            if reduced_series.size < 3:
                seg_med, seg_mad = _rolling_median_and_mad(segment, window)
            else:
                reduced_window = max(3, int(round(window / stride)))
                if reduced_window % 2 == 0:
                    reduced_window += 1
                if reduced_window > reduced_series.size:
                    reduced_window = reduced_series.size
                    if reduced_window % 2 == 0:
                        reduced_window -= 1
                if reduced_window < 3:
                    seg_med, seg_mad = _rolling_median_and_mad(segment, window)
                else:
                    reduced_med, reduced_mad = _rolling_median_and_mad(
                        reduced_series, reduced_window
                    )
                    seg_med = np.repeat(reduced_med, stride)[:seg_len]
                    seg_mad = np.repeat(reduced_mad, stride)[:seg_len]
        local_medians[seg_start:seg_end] = seg_med
        local_mads[seg_start:seg_end] = seg_mad

    thresholds = np.full(signal.size, global_threshold, dtype=float)
    valid_local = np.isfinite(local_medians)
    adjusted_mads = np.where(
        valid_local & np.isfinite(local_mads) & (local_mads > 1e-6),
        local_mads,
        mad,
    )
    local_thresholds = local_medians + 2.5 * adjusted_mads
    local_thresholds = np.where(
        local_thresholds <= 0,
        np.where(local_medians > 0, local_medians * 1.5, global_threshold),
        local_thresholds,
    )
    np.copyto(thresholds, local_thresholds, where=valid_local)

    thresholds = np.clip(thresholds, 0.05, None)

    above_threshold = positive_velocity >= thresholds
    crossings = np.flatnonzero(~above_threshold[:-1] & above_threshold[1:])
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

    downstroke_grace = max(1, int(round(0.02 * fs)))
    sustained_negative = max(1, int(round(0.02 * fs)))
    notch_guard = max(1, int(round(0.05 * fs)))

    if sp_signal is not None:
        candidate_peaks, _ = sp_signal.find_peaks(
            signal,
            distance=min_distance,
            prominence=min_prominence,
        )
        candidate_peaks = np.asarray(candidate_peaks, dtype=int)
    else:
        candidate_peaks, _ = find_prominent_peaks(
            signal,
            fs=fs,
            min_rr=min_rr,
            max_rr=max_rr,
            min_prominence=min_prominence,
        )

    if candidate_peaks.size == 0:
        return (
            np.array([], dtype=int),
            np.array([], dtype=float),
            np.array([], dtype=int),
            calibration_segments,
        )

    onset_indices = np.searchsorted(candidate_onsets, candidate_peaks, side="right") - 1
    valid_mask = onset_indices >= 0
    if valid_mask.any():
        onset_samples = candidate_onsets[onset_indices[valid_mask]]
        within_window = candidate_peaks[valid_mask] - onset_samples <= search_horizon
        valid_mask_indices = np.flatnonzero(valid_mask)
        keep = valid_mask_indices[within_window]
    else:
        keep = np.array([], dtype=int)

    if keep.size == 0:
        return (
            np.array([], dtype=int),
            np.array([], dtype=float),
            np.array([], dtype=int),
            calibration_segments,
        )

    candidate_peaks = candidate_peaks[keep]
    onset_samples = candidate_onsets[onset_indices[keep]]

    order = np.lexsort((-signal[candidate_peaks], onset_samples))
    sorted_onsets = onset_samples[order]
    _, unique_indices = np.unique(sorted_onsets, return_index=True)
    chosen = order[unique_indices]
    candidate_peaks = candidate_peaks[chosen]
    onset_samples = onset_samples[chosen]

    order = np.argsort(candidate_peaks, kind="mergesort")
    candidate_peaks = candidate_peaks[order]

    precomputed_prominences: Optional[np.ndarray]
    if candidate_peaks.size and sp_signal is not None:
        try:
            wlen = int(min(signal.size, 2 * max_distance))
            wlen = max(wlen, 3)
            if wlen % 2 == 0:
                wlen = max(3, wlen - 1)
            precomputed_prominences, _, _ = sp_signal.peak_prominences(
                signal,
                candidate_peaks,
                wlen=wlen,
            )
        except Exception:
            precomputed_prominences = None
    else:
        precomputed_prominences = None

    negative_mask = np.isfinite(velocity) & (velocity <= 0)
    if negative_mask.size >= sustained_negative:
        downstroke_kernel = np.ones(sustained_negative, dtype=int)
        downstroke_counts = np.convolve(
            negative_mask.astype(int), downstroke_kernel, mode="valid"
        )
    else:
        downstroke_counts = np.array([], dtype=int)

    def _estimate_prominence(peak_idx: int) -> Optional[float]:
        left = signal[max(0, peak_idx - max_distance) : peak_idx + 1]
        right = signal[peak_idx : min(signal.size, peak_idx + max_distance)]

        left = left[np.isfinite(left)]
        right = right[np.isfinite(right)]
        if left.size == 0 or right.size == 0:
            return None

        left_min = float(np.min(left))
        right_min = float(np.min(right))
        return float(signal[peak_idx] - max(left_min, right_min))

    def _estimate_notch(peak_idx: int) -> int:
        notch_idx = -1
        notch_start = peak_idx + 1
        notch_end = min(signal.size, peak_idx + int(round(0.45 * fs)))
        if notch_end > notch_start + 1:
            velocity_slice = velocity[notch_start:notch_end]
            zero_crossings = np.flatnonzero(
                (velocity_slice[:-1] < 0) & (velocity_slice[1:] >= 0)
            )
            if zero_crossings.size:
                zero_idx = notch_start + zero_crossings[0] + 1
                window_radius = max(1, int(round(0.02 * fs)))
                local_start = max(notch_start, zero_idx - window_radius)
                local_end = min(signal.size, zero_idx + window_radius)
                if local_end > local_start:
                    local_segment = signal[local_start:local_end]
                    if np.any(np.isfinite(local_segment)):
                        notch_rel = int(np.nanargmin(local_segment))
                        notch_idx = local_start + notch_rel
        return notch_idx

    def _legacy_has_sustained_downstroke(peak_idx: int) -> bool:
        window_end = min(signal.size, peak_idx + sustained_negative + downstroke_grace + 1)
        if window_end <= peak_idx:
            return False
        velocity_window = velocity[peak_idx:window_end]
        finite_velocity = velocity_window[np.isfinite(velocity_window)]
        if finite_velocity.size == 0:
            return False
        negative_mask = velocity_window <= 0
        if negative_mask.size < sustained_negative:
            return False
        kernel = np.ones(sustained_negative, dtype=int)
        sustained = np.convolve(negative_mask.astype(int), kernel, mode="valid")
        return bool(np.any(sustained >= sustained_negative))

    def _has_sustained_downstroke(peak_idx: int) -> bool:
        if downstroke_counts.size:
            window_end = min(
                signal.size, peak_idx + sustained_negative + downstroke_grace + 1
            )
            if window_end - peak_idx < sustained_negative:
                return False
            max_start = downstroke_counts.size - 1
            start = peak_idx
            if start > max_start:
                return False
            stop = min(window_end - sustained_negative, max_start)
            if stop < start:
                return False
            return bool(
                np.any(downstroke_counts[start : stop + 1] >= sustained_negative)
            )
        return _legacy_has_sustained_downstroke(peak_idx)

    peaks: List[int] = []
    prominences: List[float] = []
    notches: List[int] = []
    suppress_until = -1

    for idx, peak_idx in enumerate(candidate_peaks):
        if precomputed_prominences is not None:
            prominence = float(precomputed_prominences[idx])
            if not np.isfinite(prominence):
                continue
        else:
            prominence = _estimate_prominence(peak_idx)
            if prominence is None:
                continue
        if prominence < min_prominence:
            continue
        if not _has_sustained_downstroke(peak_idx):
            continue

        if peaks and peak_idx - peaks[-1] < min_distance:
            if signal[peak_idx] > signal[peaks[-1]]:
                peaks[-1] = peak_idx
                prominences[-1] = prominence
                notches[-1] = _estimate_notch(peak_idx)
                suppress_until = (
                    notches[-1] if notches[-1] >= 0 else peak_idx
                ) + notch_guard
            continue

        if peak_idx <= suppress_until:
            if peaks and signal[peak_idx] > signal[peaks[-1]]:
                peaks[-1] = peak_idx
                prominences[-1] = prominence
                notches[-1] = _estimate_notch(peak_idx)
                suppress_until = (
                    notches[-1] if notches[-1] >= 0 else peaks[-1]
                ) + notch_guard
            continue

        notch_idx = _estimate_notch(peak_idx)
        peaks.append(peak_idx)
        prominences.append(prominence)
        notches.append(notch_idx)
        suppress_until = (notch_idx if notch_idx >= 0 else peak_idx) + notch_guard

    if peaks:
        peak_diffs = np.diff(peaks) / fs if len(peaks) > 1 else np.array([], dtype=float)
        global_interval = float(np.median(peak_diffs)) if peak_diffs.size else math.nan
        cleaned_peaks: List[int] = [peaks[0]]
        cleaned_prom: List[float] = [prominences[0]]
        cleaned_notches: List[int] = [notches[0]]
        interval_history: List[float] = []
        for idx in range(1, len(peaks)):
            interval = float((peaks[idx] - cleaned_peaks[-1]) / fs)
            if interval_history:
                local_median = float(np.median(interval_history[-3:]))
            else:
                local_median = global_interval if not math.isnan(global_interval) else interval
            short_interval = local_median > 0 and interval < 0.4 * local_median
            weak_prominence = prominences[idx] <= 0.6 * cleaned_prom[-1]
            if short_interval or weak_prominence:
                continue
            cleaned_peaks.append(peaks[idx])
            cleaned_prom.append(prominences[idx])
            cleaned_notches.append(notches[idx])
            interval_history.append(interval)

        peaks = cleaned_peaks
        prominences = cleaned_prom
        notches = cleaned_notches

    if len(peaks) < 3:
        fallback_peaks, fallback_prom = find_prominent_peaks(
            signal,
            fs=fs,
            min_rr=min_rr,
            max_rr=max_rr,
            min_prominence=min_prominence,
        )
        if fallback_peaks.size:
            fallback_notches = np.full_like(fallback_peaks, -1, dtype=int)
            return fallback_peaks, fallback_prom, fallback_notches, calibration_segments

    return (
        np.asarray(peaks, dtype=int),
        np.asarray(prominences, dtype=float),
        np.asarray(notches, dtype=int),
        calibration_segments,
    )


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
    (
        systolic_indices,
        prominences,
        notch_indices,
        calibration_segments,
    ) = detect_systolic_peaks(
        smoothed,
        fs=fs,
        min_rr=min_rr,
        max_rr=max_rr,
        prominence_floor=config.min_prominence,
        prominence_noise_factor=config.prominence_noise_factor,
    )

    beats: List[Beat] = []
    prev_dia_sample: Optional[int] = None

    reset_boundaries = [end for _, end in calibration_segments]
    post_calibration_targets: set[int] = set()
    for _, end in calibration_segments:
        for candidate in systolic_indices:
            if candidate >= end:
                post_calibration_targets.add(candidate)
                break

    for idx, sys_idx in enumerate(systolic_indices):
        prev_sys = systolic_indices[idx - 1] if idx > 0 else None
        next_sys = systolic_indices[idx + 1] if idx + 1 < len(systolic_indices) else None

        notch_idx = notch_indices[idx] if idx < len(notch_indices) else -1

        if prev_sys is not None:
            search_start = prev_sys + max(1, int(round(0.05 * fs)))
        else:
            search_start = max(0, sys_idx - int(round(max_rr * fs)))

        last_reset = max(
            (boundary for boundary in reset_boundaries if boundary <= sys_idx),
            default=-1,
        )
        if last_reset >= 0:
            search_start = max(search_start, int(last_reset))

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
        notch_time = float(time[notch_idx]) if 0 <= notch_idx < len(time) else math.nan
        map_time = float((systolic_time + diastolic_time) / 2.0)
        area_map = math.nan
        integration_end = dia_idx
        if 0 <= notch_idx <= dia_idx:
            integration_end = notch_idx
        if prev_dia_sample is not None and prev_dia_sample < integration_end:
            start_idx = prev_dia_sample
            end_idx = integration_end + 1
            if end_idx - start_idx > 2:
                segment_time = time[start_idx:end_idx]
                segment_pressure = pressure_filled[start_idx:end_idx]
                area = float(np.trapezoid(segment_pressure, segment_time))
                duration = float(segment_time[-1] - segment_time[0])
                if duration > 0:
                    area_map = area / duration
                    map_time = float((segment_time[0] + segment_time[-1]) / 2.0)

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
        is_post_calibration = sys_idx in post_calibration_targets
        if is_post_calibration:
            reasons.append("post calibration gap")

        beats.append(
            Beat(
                systolic_time=systolic_time,
                systolic_pressure=systolic_pressure,
                diastolic_time=diastolic_time,
                diastolic_pressure=diastolic_pressure,
                notch_time=notch_time,
                map_time=map_time,
                map_pressure=map_pressure,
                rr_interval=rr_interval,
                systolic_prominence=float(prominences[idx]),
                is_artifact=False,
                artifact_reasons=reasons,
                is_post_calibration_recovery=is_post_calibration,
            )
        )

        prev_dia_sample = dia_idx

    apply_artifact_rules(beats, config=config)
    return beats


def _rolling_median_and_mad(values: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return rolling median and MAD for ``values`` using an odd-sized window."""

    if window % 2 == 0:
        raise ValueError("window must be odd to compute centred statistics")

    values = np.asarray(values, dtype=float)
    medians = np.full(values.shape, np.nan, dtype=float)
    mads = np.full(values.shape, np.nan, dtype=float)

    if values.size == 0:
        return medians, mads

    if bn is not None:
        medians_bn = bn.move_median(values, window=window, center=True, min_count=3)
        medians[:] = medians_bn

        deviations = np.abs(values - medians_bn)
        mad_bn = bn.move_median(deviations, window=window, center=True, min_count=3)
        mad_bn = np.asarray(mad_bn, dtype=float) * 1.4826
        fallback_mask = (mad_bn < 1e-3) | ~np.isfinite(mad_bn)
        if np.any(fallback_mask):
            std_bn = bn.move_std(
                values, window=window, center=True, ddof=0, min_count=3
            )
            std_bn = np.asarray(std_bn, dtype=float)
            mad_bn = mad_bn.copy()
            mad_bn[fallback_mask] = std_bn[fallback_mask]

        mads[:] = mad_bn
        return medians, mads

    if sliding_window_view is not None:
        half_window = window // 2
        padded = np.pad(values, (half_window, half_window), mode="constant", constant_values=np.nan)
        windows = sliding_window_view(padded, window_shape=window)
        finite_counts = np.sum(np.isfinite(windows), axis=1)
        valid_mask = finite_counts >= 3

        if not np.any(valid_mask):
            return medians, mads

        valid_windows = windows[valid_mask]
        medians_valid = np.nanmedian(valid_windows, axis=1)
        medians[valid_mask] = medians_valid

        deviations = np.abs(valid_windows - medians_valid[:, None])
        mad_valid = 1.4826 * np.nanmedian(deviations, axis=1)
        fallback_mask = (mad_valid < 1e-3) | ~np.isfinite(mad_valid)
        if np.any(fallback_mask):
            mad_valid[fallback_mask] = np.nanstd(valid_windows[fallback_mask], axis=1, ddof=0)

        mads[valid_mask] = mad_valid
        return medians, mads

    # Fallback path when sliding_window_view is unavailable (older NumPy).
    series = pd.Series(values, dtype=float)
    rolling = series.rolling(window, center=True, min_periods=3)
    medians_series = rolling.median()
    medians = medians_series.to_numpy(dtype=float)

    deviations = (series - medians_series).abs()
    mad_series = deviations.rolling(window, center=True, min_periods=3).median() * 1.4826
    mads = mad_series.to_numpy(dtype=float)

    fallback_mask = (mads < 1e-3) | ~np.isfinite(mads)
    if np.any(fallback_mask):
        std_series = rolling.std(ddof=0)
        std_values = std_series.to_numpy(dtype=float)
        mads[fallback_mask] = std_values[fallback_mask]

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
        if (
            beat.is_post_calibration_recovery
            and beat.is_artifact
            and not severe_reasons
        ):
            beat.is_artifact = False
            combined_reasons = [
                reason
                for reason in combined_reasons
                if reason not in soft_reasons
            ]
        beat.artifact_reasons = combined_reasons if beat.is_artifact else []


def plot_waveform(
    time: np.ndarray,
    pressure: np.ndarray,
    beats: Sequence[Beat],
    *,
    show: bool = True,
    save_path: Optional[Path] = None,
    max_points: Optional[int] = 50000,
) -> None:
    """Plot the waveform with annotated beat landmarks."""

    if plt is None:  # pragma: no cover - handled in main
        raise RuntimeError("matplotlib is required for plotting")

    plt.figure(figsize=(12, 6))
    time_values = np.asarray(time, dtype=float)
    pressure_values = np.asarray(pressure, dtype=float)

    if max_points is not None and max_points > 0 and time_values.size > max_points:
        stride = int(math.ceil(time_values.size / float(max_points)))
        stride = max(1, stride)
        downsampled_time = time_values[::stride]
        downsampled_pressure = pressure_values[::stride]
        if downsampled_time.size == 0 or downsampled_time[-1] != time_values[-1]:
            downsampled_time = np.append(downsampled_time, time_values[-1])
            downsampled_pressure = np.append(downsampled_pressure, pressure_values[-1])
        plot_time = downsampled_time
        plot_pressure = downsampled_pressure
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


def summarise_beats(beats: Sequence[Beat]) -> DataFrame:
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


def print_column_overview(frame: DataFrame) -> None:
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
    parser.add_argument(
        "--separator",
        help="Override the column separator if auto-detection fails (e.g., ',' or \\t).",
    )
    parser.add_argument(
        "--plot-max-points",
        type=int,
        default=50000,
        help="Maximum waveform samples to render (set to 0 to disable downsampling).",
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

    if args.plot_max_points is not None and args.plot_max_points < 0:
        parser.error("--plot-max-points must be zero or a positive integer")

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

    time_default = _default_column_selection(preview.column_names, args.time_column, fallback="Time")
    remaining_columns = [name for name in preview.column_names if name != time_default]
    pressure_default = _default_column_selection(
        remaining_columns if remaining_columns else preview.column_names,
        args.column if args.column != time_default else None,
        fallback="reBAP",
    )

    selected_time_column: Optional[str] = None
    selected_pressure_column: Optional[str] = None
    dialog_shown = False
    dialog_result: Optional[ImportDialogResult] = None

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

    print(f"Loaded file preview: {file_path}")
    print("\nColumn preview (first 50 rows loaded for preview):")
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
    if preview_frame is not None:
        print_column_overview(preview_frame)
    else:
        print("  Preview will be available after import.")

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
            import_settings=dialog_result,
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
        plot_waveform(
            time,
            pressure,
            beats,
            show=show_plot,
            save_path=args.savefig,
            max_points=None if args.plot_max_points == 0 else args.plot_max_points,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
