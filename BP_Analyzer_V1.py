"""BP Analyzer V1
===================

A command-line utility for streaming ingestion of delimited blood pressure files.

Features
--------
* Streaming parser with configurable delimiter, NA values, and first data row.
* Terminal preview that lets users map source columns to the expected schema.
* Persistence of the selected columns to a Parquet file plus a JSON manifest
  capturing the parsing configuration and resulting schema.

The module can be run as a script. Example::

    python BP_Analyzer_V1.py --file path/to/data.csv

"""
from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, Iterator, List, Optional

import tkinter as tk
from tkinter import filedialog, ttk

import itertools
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


@dataclass
class ParserConfig:
    """Configuration parameters that drive parsing."""

    file_path: str
    delimiter: str = ","
    na_values: List[str] = None
    first_data_row: int = 1
    header_row: Optional[int] = None
    chunk_size: int = 5000
    encoding: str = "utf-8"

    def __post_init__(self) -> None:
        if self.na_values is None:
            self.na_values = ["", "NA", "NaN"]
        if self.first_data_row < 1:
            raise ValueError("first_data_row must be >= 1")
        if self.header_row is not None and self.header_row < 1:
            raise ValueError("header_row must be >= 1 when provided")
        if self.header_row is not None and self.header_row >= self.first_data_row:
            # Ensure the header is not interpreted as data.
            print(
                "[Info] header_row occurs at or after first_data_row. "
                "Adjusting first_data_row to start after the header."
            )
            self.first_data_row = self.header_row + 1


def interpret_delimiter(raw: str) -> str:
    """Convert user provided delimiter strings such as "\\t" into literals."""

    escape_map = {"\\t": "\t", "\\n": "\n", "\\r": "\r"}
    alias_map = {
        "tab": "\t",
        "space": " ",
        "comma": ",",
        "semicolon": ";",
        "pipe": "|",
        "colon": ":",
    }
    if raw in escape_map:
        return escape_map[raw]

    normalized = raw.lower()
    if normalized in alias_map:
        return alias_map[normalized]

    return raw


COMMON_DELIMITERS = [",", "\t", ";", "|", ":", " "]


def _tokenise_sample_line(line: str, delimiter: str) -> List[str]:
    """Split a sample line using the candidate delimiter."""

    if not line:
        return []

    if delimiter == " ":
        stripped = line.strip()
        if not stripped:
            return []
        return [token for token in stripped.split() if token]

    try:
        return next(csv.reader([line], delimiter=delimiter))
    except csv.Error:
        return line.split(delimiter)


def _has_consistent_split(lines: List[str], delimiter: str) -> bool:
    """Return True when all lines produce the same non-trivial column count."""

    lengths: List[int] = []
    saw_multiple_columns = False
    for line in lines:
        tokens = _tokenise_sample_line(line, delimiter)
        if not tokens:
            continue
        lengths.append(len(tokens))
        if len(tokens) > 1:
            saw_multiple_columns = True

    if not lengths or not saw_multiple_columns:
        return False

    return len(set(lengths)) == 1


def detect_delimiter(
    file_path: str,
    encoding: str,
    sample_size: int = 4096,
    fallback: str = ",",
) -> tuple[str, str, bool]:
    """Attempt to infer a delimiter from a text sample."""

    sample_line = ""
    detected = fallback
    detected_from_sniffer = False

    candidate_lines: List[str] = []

    try:
        with open(file_path, "r", encoding=encoding, newline="") as handle:
            sample = handle.read(sample_size)
            if not sample:
                return detected, sample_line, detected_from_sniffer
            if "\n" not in sample:
                sample += handle.readline()

            raw_lines = sample.splitlines()
            while len(raw_lines) < 5:
                extra_line = handle.readline()
                if not extra_line:
                    break
                raw_lines.append(extra_line.rstrip("\r\n"))

            candidate_lines = [line for line in raw_lines if line.strip()]
            sample_line = candidate_lines[0] if candidate_lines else ""

            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(sample, delimiters=COMMON_DELIMITERS)
            if _has_consistent_split(candidate_lines, dialect.delimiter):
                detected = dialect.delimiter
                detected_from_sniffer = True
            else:
                # Let the heuristics below choose a more reliable delimiter.
                detected_from_sniffer = False
    except (csv.Error, OSError, UnicodeDecodeError):
        pass

    if not candidate_lines:
        return detected, sample_line, detected_from_sniffer

    sniffer_candidate = detected if detected_from_sniffer else None

    if not detected_from_sniffer or detected == fallback:
        for delimiter in [detected] + COMMON_DELIMITERS:
            if delimiter and _has_consistent_split(candidate_lines, delimiter):
                detected = delimiter
                detected_from_sniffer = detected_from_sniffer and delimiter == sniffer_candidate
                break

    return detected, sample_line, detected_from_sniffer


def _format_delimiter_for_display(delimiter: str) -> str:
    """Return a human-readable label for a delimiter."""

    labels = {
        "\t": "\\t (tab)",
        " ": "' ' (space)",
        ",": "',' (comma)",
        ";": "';' (semicolon)",
        "|": "'|' (pipe)",
        ":": "':' (colon)",
    }
    return labels.get(delimiter, repr(delimiter))


def prompt_for_delimiter(
    detected: str,
    sample_line: str,
    detected_from_sniffer: bool,
) -> str:
    """Ask the user to accept or override an inferred delimiter."""

    print("\nDelimiter selection")
    if sample_line:
        print("Sample row:")
        print(sample_line)

    description = _format_delimiter_for_display(detected)
    if detected_from_sniffer:
        message = f"Detected delimiter appears to be {description}."
    else:
        message = f"Could not automatically determine delimiter; defaulting to {description}."

    response = input(
        message
        + " Press Enter to accept or type a custom delimiter (use \\t for tab, 'space' for a space, etc.): "
    ).strip()

    if not response:
        return detected

    return interpret_delimiter(response)


def _tokenise_preview_line(line: str, delimiter: str) -> List[str]:
    """Split a preview line using the provided delimiter."""

    if not line:
        return []

    if delimiter == " ":
        stripped = line.strip()
        if not stripped:
            return []
        return [token for token in stripped.split() if token]

    try:
        return next(csv.reader([line], delimiter=delimiter))
    except csv.Error:
        return line.split(delimiter)


def _load_preview_lines(file_path: str, encoding: str, limit: int = 50) -> List[str]:
    """Read a handful of lines from the file for preview purposes."""

    lines: List[str] = []
    try:
        with open(file_path, "r", encoding=encoding, errors="ignore") as handle:
            for _ in range(limit):
                line = handle.readline()
                if not line:
                    break
                lines.append(line.rstrip("\r\n"))
    except OSError:
        return []
    return lines


def _render_preview_table(widget: tk.Text, rows: List[List[str]], max_rows: int = 25) -> None:
    """Render a simple fixed-width preview of the parsed rows."""

    widget.configure(state=tk.NORMAL)
    widget.delete("1.0", tk.END)

    if not rows:
        widget.insert(tk.END, "No preview available for the selected delimiter.")
        widget.configure(state=tk.DISABLED)
        return

    display_rows = rows[:max_rows]
    column_count = max((len(row) for row in display_rows), default=0)

    if column_count == 0:
        widget.insert(tk.END, "No preview available for the selected delimiter.")
        widget.configure(state=tk.DISABLED)
        return

    widths: List[int] = []
    for column in range(column_count):
        widths.append(
            max(
                [len(str(row[column])) if column < len(row) else 0 for row in display_rows]
                + [len(str(column + 1))]
            )
        )

    header = "    " + "  ".join(
        str(index + 1).ljust(widths[index]) for index in range(column_count)
    )
    widget.insert(tk.END, header + "\n")

    for row_index, row in enumerate(display_rows, start=1):
        values: List[str] = []
        for column in range(column_count):
            value = str(row[column]) if column < len(row) and row[column] is not None else ""
            values.append(value.ljust(widths[column]))
        line = f"{row_index:>4} " + "  ".join(values)
        widget.insert(tk.END, line + "\n")

    widget.configure(state=tk.DISABLED)


class DelimiterPreviewDialog:
    """Interactive delimiter picker that mirrors the working preview design."""
    _OPTION_MAP = [
        ("Auto-detect", None),
        ("Tab (\\t)", "\t"),
        ("Comma (,)", ","),
        ("Semicolon (;)", ";"),
        ("Pipe (|)", "|"),
        ("Space", " "),
        ("Colon (:)", ":"),
    ]

    def __init__(
        self,
        file_path: str,
        encoding: str,
        detected: str,
        detected_from_sniffer: bool,
        fallback: str = ",",
    ) -> None:
        self._file_path = file_path
        self._encoding = encoding
        self._auto_detected = detected or fallback
        self._detected_from_sniffer = detected_from_sniffer
        self._default_result = self._auto_detected
        self._result: Optional[str] = self._auto_detected

        self._raw_lines = _load_preview_lines(file_path, encoding)

        self.root = tk.Tk()
        self.root.title("Select Delimiter")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close_attempt)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self._selection = tk.StringVar(value=self._initial_selection_label())
        self._custom_value = tk.StringVar(value="")
        self._status_message = tk.StringVar(value="")
        self._column_count = tk.StringVar(value="Columns detected: 0")

        self._build_widgets()
        self._update_preview()

    def _initial_selection_label(self) -> str:
        for label, value in self._OPTION_MAP:
            if value and value == self._auto_detected:
                return label
        return "Auto-detect"

    def _build_widgets(self) -> None:
        frame = ttk.Frame(self.root, padding=12)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.columnconfigure(0, weight=1)

        ttk.Label(
            frame,
            text="Choose the delimiter that matches your file",
            font=("TkDefaultFont", 11, "bold"),
        ).grid(row=0, column=0, sticky="w")

        description = self._build_description()
        if description:
            ttk.Label(frame, text=description, wraplength=460, justify=tk.LEFT).grid(
                row=1, column=0, sticky="w", pady=(6, 10)
            )

        selection_row = ttk.Frame(frame)
        selection_row.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        selection_row.columnconfigure(1, weight=1)

        ttk.Label(selection_row, text="Delimiter:").grid(row=0, column=0, sticky="w")
        option_labels = [label for label, _ in self._OPTION_MAP] + ["Custom"]
        self._combobox = ttk.Combobox(
            selection_row,
            values=option_labels,
            state="readonly",
            textvariable=self._selection,
            width=18,
        )
        self._combobox.grid(row=0, column=1, sticky="w", padx=(8, 12))
        self._combobox.bind("<<ComboboxSelected>>", lambda _event: self._on_selection_change())

        ttk.Label(selection_row, text="Custom:").grid(row=0, column=2, sticky="w")
        self._custom_entry = ttk.Entry(
            selection_row,
            textvariable=self._custom_value,
            width=12,
            state="disabled",
        )
        self._custom_entry.grid(row=0, column=3, sticky="w", padx=(8, 0))
        self._custom_value.trace_add("write", lambda *_: self._update_preview())

        preview_frame = ttk.LabelFrame(frame, text="Preview (first 25 rows)")
        preview_frame.grid(row=3, column=0, sticky="nsew")
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)

        self._preview_text = tk.Text(
            preview_frame,
            height=12,
            wrap="none",
            font=("Courier New", 10),
        )
        self._preview_text.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(
            preview_frame, orient="vertical", command=self._preview_text.yview
        )
        scrollbar.grid(row=0, column=1, sticky="ns")
        self._preview_text.configure(yscrollcommand=scrollbar.set)

        ttk.Label(frame, textvariable=self._column_count).grid(
            row=4, column=0, sticky="w", pady=(8, 0)
        )
        ttk.Label(frame, textvariable=self._status_message, foreground="#b22222").grid(
            row=5, column=0, sticky="w", pady=(4, 0)
        )

        buttons = ttk.Frame(frame)
        buttons.grid(row=6, column=0, sticky="e", pady=(12, 0))
        ttk.Button(buttons, text="Cancel", command=self._on_close_attempt).pack(
            side=tk.RIGHT
        )
        ttk.Button(buttons, text="Confirm", command=self._confirm).pack(
            side=tk.RIGHT, padx=(0, 8)
        )

    def _build_description(self) -> str:
        message = _format_delimiter_for_display(self._auto_detected)
        if self._detected_from_sniffer:
            return f"Automatically detected delimiter: {message}."
        return (
            "Unable to confidently detect the delimiter. "
            f"Defaulting to {message}. Choose a different option if needed."
        )

    def _on_selection_change(self) -> None:
        if self._selection.get() == "Custom":
            self._custom_entry.configure(state="normal")
            self._custom_entry.focus_set()
        else:
            self._custom_entry.configure(state="disabled")
        self._update_preview()

    def _resolve_selection(self) -> Optional[str]:
        label = self._selection.get()
        if label == "Custom":
            value = self._custom_value.get().strip()
            if not value:
                return None
            return interpret_delimiter(value)

        for option_label, delimiter in self._OPTION_MAP:
            if option_label == label:
                return self._auto_detected if delimiter is None else delimiter

        return self._auto_detected

    def _update_preview(self) -> None:
        delimiter = self._resolve_selection()
        if delimiter is None:
            self._status_message.set("Enter a custom delimiter to preview the data.")
            rows: List[List[str]] = []
        else:
            self._status_message.set("")
            rows = [_tokenise_preview_line(line, delimiter) for line in self._raw_lines]

        column_count = max((len(row) for row in rows if row), default=0)
        self._column_count.set(f"Columns detected: {column_count}")
        _render_preview_table(self._preview_text, rows)

    def _confirm(self) -> None:
        delimiter = self._resolve_selection()
        if delimiter is None:
            self._status_message.set("Provide a custom delimiter before confirming.")
            return
        self._result = delimiter
        self.root.destroy()

    def _on_close_attempt(self) -> None:
        self._result = self._result or self._default_result
        self.root.destroy()

    def show(self) -> str:
        self.root.mainloop()
        return self._result or self._default_result


def determine_delimiter(args: argparse.Namespace) -> str:
    """Resolve the delimiter from CLI arguments or interactive selection."""

    if args.delimiter:
        return interpret_delimiter(args.delimiter)

    detected, _sample_line, detected_from_sniffer = detect_delimiter(
        args.file, args.encoding
    )
    dialog = DelimiterPreviewDialog(
        args.file, args.encoding, detected, detected_from_sniffer
    )
    return dialog.show()


def read_header_row(config: ParserConfig) -> Optional[List[str]]:
    """Return column names from the configured header row if available."""

    if config.header_row is None:
        return None

    skip_initial_space = config.delimiter == " "

    with open(config.file_path, "r", encoding=config.encoding, newline="") as handle:
        reader = csv.reader(
            handle, delimiter=config.delimiter, skipinitialspace=skip_initial_space
        )
        for index, row in enumerate(reader, start=1):
            if index == config.header_row:
                return [value.strip() for value in row]
    return None


def compute_skiprows(config: ParserConfig) -> List[int]:
    """Compute 0-based row numbers that should be skipped when reading data."""

    if config.first_data_row <= 1:
        return []
    return list(range(config.first_data_row - 1))


def stream_dataframe_chunks(
    config: ParserConfig, column_names: Optional[List[str]]
) -> Iterator[pd.DataFrame]:
    """Yield chunks of the dataset as pandas DataFrames."""

    skiprows = compute_skiprows(config)
    bad_line_stats = {"count": 0, "examples": []}

    def _handle_bad_line(bad_line: List[str]) -> Optional[List[str]]:
        """Skip malformed rows while keeping track of a few examples."""

        bad_line_stats["count"] += 1
        if len(bad_line_stats["examples"]) < 3:
            bad_line_stats["examples"].append(bad_line)
        return None

    skip_initial_space = config.delimiter == " "

    read_kwargs = {
        "sep": config.delimiter,
        "na_values": config.na_values,
        "chunksize": config.chunk_size,
        "skiprows": skiprows,
        "header": None,
        "encoding": config.encoding,
        "engine": "python",
        "on_bad_lines": _handle_bad_line,
    }
    if skip_initial_space:
        read_kwargs["skipinitialspace"] = True
    if column_names:
        read_kwargs["names"] = column_names

    chunk_iterator = pd.read_csv(config.file_path, **read_kwargs)

    inferred_names: Optional[List[str]] = column_names
    for chunk in chunk_iterator:
        if inferred_names is None:
            inferred_names = [f"column_{idx + 1}" for idx in range(len(chunk.columns))]
        chunk.columns = inferred_names
        yield chunk

    if bad_line_stats["count"]:
        example_lines = [
            " | ".join(line) for line in bad_line_stats["examples"] if line
        ]
        preview = "; ".join(example_lines)
        message = f"[Warning] Skipped {bad_line_stats['count']} malformed line(s)."
        if preview:
            message += f" Examples: {preview}"
        print(message)


def display_preview(chunk: pd.DataFrame, max_rows: int = 10) -> None:
    """Pretty-print a sample of the chunk for terminal inspection."""

    preview = chunk.head(max_rows)
    print("\nPreview of the incoming data:")
    print(preview.to_string(index=False))


class ColumnMappingDialog:
    """Tkinter dialog for mapping source columns to target fields."""

    REQUIRED_TARGETS = ["Time", "Blood Pressure"]
    OPTIONAL_TARGETS = ["Comments", "ECG"]

    def __init__(self, column_names: List[str]):
        self.column_names = column_names
        self._mapping: Dict[str, Optional[str]] = {
            target: None for target in self.REQUIRED_TARGETS + self.OPTIONAL_TARGETS
        }
        self._result: Optional[Dict[str, str]] = None

        self.root = tk.Tk()
        self.root.title("Map Columns to Targets")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close_attempt)

        self._warning_var = tk.StringVar(value="")
        self._target_order = self.REQUIRED_TARGETS + self.OPTIONAL_TARGETS

        self._build_widgets()

    def _build_widgets(self) -> None:
        """Create and layout widgets for the dialog."""

        frame = tk.Frame(self.root, padx=10, pady=10)
        frame.pack(fill=tk.BOTH, expand=True)

        lists_frame = tk.Frame(frame)
        lists_frame.pack(fill=tk.BOTH, expand=True)

        source_frame = tk.Frame(lists_frame)
        source_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        tk.Label(source_frame, text="Available Source Columns").pack(anchor=tk.W)
        self.source_listbox = tk.Listbox(
            source_frame,
            exportselection=False,
            height=12,
        )
        self.source_listbox.pack(fill=tk.BOTH, expand=True)
        for name in self.column_names:
            self.source_listbox.insert(tk.END, name)

        controls_frame = tk.Frame(lists_frame)
        controls_frame.pack(side=tk.LEFT, fill=tk.Y)

        tk.Button(
            controls_frame,
            text="Assign →",
            command=self._assign_selected,
            width=12,
        ).pack(pady=(30, 5))
        tk.Button(
            controls_frame,
            text="Clear Assignment",
            command=self._clear_selected,
            width=12,
        ).pack()

        targets_frame = tk.Frame(lists_frame)
        targets_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0))

        tk.Label(targets_frame, text="Target Assignments").pack(anchor=tk.W)
        self.target_listbox = tk.Listbox(
            targets_frame,
            exportselection=False,
            height=12,
        )
        self.target_listbox.pack(fill=tk.BOTH, expand=True)

        self._update_target_listbox()

        warning_label = tk.Label(
            frame,
            textvariable=self._warning_var,
            fg="#b22222",
            wraplength=360,
            justify=tk.LEFT,
        )
        warning_label.pack(fill=tk.X, pady=(10, 0))

        buttons_frame = tk.Frame(frame)
        buttons_frame.pack(fill=tk.X, pady=(10, 0))

        tk.Button(buttons_frame, text="Confirm", command=self._confirm).pack(
            side=tk.RIGHT
        )

    def _update_target_listbox(self) -> None:
        """Refresh the entries in the target listbox."""

        self.target_listbox.delete(0, tk.END)
        for target in self._target_order:
            assignment = self._mapping.get(target)
            display_value = assignment if assignment else "—"
            if target in self.REQUIRED_TARGETS:
                label = f"{target} (required): {display_value}"
            else:
                label = f"{target}: {display_value}"
            self.target_listbox.insert(tk.END, label)

        for index, target in enumerate(self._target_order):
            if target in self.REQUIRED_TARGETS:
                self.target_listbox.itemconfig(
                    index,
                    fg="#1f4b99",
                    selectforeground="#ffffff",
                    selectbackground="#1f4b99",
                )

    def _assign_selected(self) -> None:
        source_selection = self.source_listbox.curselection()
        target_selection = self.target_listbox.curselection()

        if not source_selection or not target_selection:
            self._warning_var.set("Select both a source column and a target slot to assign.")
            return

        source_column = self.column_names[source_selection[0]]
        target_field = self._target_order[target_selection[0]]

        if source_column in self._mapping.values() and self._mapping.get(target_field) != source_column:
            self._warning_var.set(
                f"'{source_column}' is already assigned. Clear the existing assignment before reusing it."
            )
            return

        self._mapping[target_field] = source_column
        self._warning_var.set("")
        self._update_target_listbox()

    def _clear_selected(self) -> None:
        target_selection = self.target_listbox.curselection()
        if not target_selection:
            self._warning_var.set("Select a target slot to clear its assignment.")
            return

        target_field = self._target_order[target_selection[0]]
        self._mapping[target_field] = None
        self._warning_var.set("")
        self._update_target_listbox()

    def _confirm(self) -> None:
        missing = [
            target
            for target in self.REQUIRED_TARGETS
            if not self._mapping.get(target)
        ]
        if missing:
            self._warning_var.set(
                "Required targets missing assignments: " + ", ".join(missing)
            )
            return

        self._result = {k: v for k, v in self._mapping.items() if v}
        self.root.destroy()

    def _on_close_attempt(self) -> None:
        missing = [
            target
            for target in self.REQUIRED_TARGETS
            if not self._mapping.get(target)
        ]
        if missing:
            self._warning_var.set(
                "Assign all required targets (Time, Blood Pressure) before closing."
            )
            return
        self._result = {k: v for k, v in self._mapping.items() if v}
        self.root.destroy()

    def show(self) -> Dict[str, str]:
        """Run the dialog and return the selected mapping."""

        self.root.mainloop()
        return self._result or {}


def prompt_output_paths(default_path: str) -> tuple[str, str]:
    """Ask the user where output artifacts should be stored."""

    base_default = os.path.splitext(default_path)[0]
    default_parquet = f"{base_default}_selected.parquet"
    default_manifest = f"{base_default}_manifest.json"

    parquet_path = input(
        f"Enter output Parquet path [{default_parquet}]: "
    ).strip() or default_parquet
    manifest_path = input(
        f"Enter output manifest path [{default_manifest}]: "
    ).strip() or default_manifest
    return parquet_path, manifest_path


def write_selected_columns(
    first_chunk: pd.DataFrame,
    remaining_chunks: Iterable[pd.DataFrame],
    mapping: Dict[str, str],
    parquet_path: str,
) -> Dict[str, Any]:
    """Persist selected columns to Parquet and return simple metrics."""

    schema = None
    row_count = 0
    writer: Optional[pq.ParquetWriter] = None

    all_chunks = itertools.chain([first_chunk], remaining_chunks)
    selected_columns = list(mapping.values())
    rename_map = {source: target for target, source in mapping.items()}

    try:
        for chunk in all_chunks:
            subset = chunk[selected_columns].rename(columns=rename_map)
            table = pa.Table.from_pandas(subset, preserve_index=False)
            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(parquet_path, schema)
            writer.write_table(table)
            row_count += len(subset)
    finally:
        if writer:
            writer.close()

    return {
        "rows_written": row_count,
        "schema": schema,
    }


def build_manifest(
    config: ParserConfig,
    mapping: Dict[str, str],
    parquet_path: str,
    manifest_path: str,
    metrics: Dict[str, Any],
) -> None:
    """Write a JSON manifest describing the generated Parquet file."""

    schema = metrics.get("schema")
    schema_description = []
    if schema:
        schema_description = [
            {"name": field.name, "type": str(field.type)}
            for field in schema
        ]

    manifest = {
        "manifest_version": "1.0",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "source_file": os.path.abspath(config.file_path),
        "parquet_file": os.path.abspath(parquet_path),
        "column_mapping": mapping,
        "rows_written": metrics.get("rows_written", 0),
        "parser_config": {
            "delimiter": config.delimiter,
            "na_values": config.na_values,
            "first_data_row": config.first_data_row,
            "header_row": config.header_row,
            "chunk_size": config.chunk_size,
            "encoding": config.encoding,
        },
        "schema": schema_description,
    }

    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"\nManifest written to {manifest_path}")


def run_interactive_session(config: ParserConfig) -> None:
    """Execute the main interactive parsing workflow."""

    column_names = read_header_row(config)
    chunk_iterator = stream_dataframe_chunks(config, column_names)
    try:
        first_chunk = next(chunk_iterator)
    except StopIteration:
        print("No data found with the provided configuration.")
        return

    # If no header row was provided, use column names inferred from the first chunk.
    column_names = list(first_chunk.columns)

    display_preview(first_chunk)

    dialog = ColumnMappingDialog(column_names)
    mapping = dialog.show()
    parquet_path, manifest_path = prompt_output_paths(config.file_path)

    # Persist data to Parquet.
    metrics = write_selected_columns(first_chunk, chunk_iterator, mapping, parquet_path)
    print(f"\nParquet written to {parquet_path}")
    print(f"Rows written: {metrics.get('rows_written', 0)}")

    build_manifest(config, mapping, parquet_path, manifest_path, metrics)


def parse_cli_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blood Pressure streaming parser")
    parser.add_argument(
        "--file",
        help="Path to the source data file. If omitted, a file picker dialog will open.",
    )
    parser.add_argument(
        "--delimiter",
        help="Column delimiter. Leave unset to auto-detect with an interactive prompt. Use \\t for tab.",
        default=None,
    )
    parser.add_argument(
        "--na",
        help="Comma-separated list of strings that should be interpreted as NA",
        default="",
    )
    parser.add_argument(
        "--first-data-row",
        type=int,
        default=1,
        help="1-based index for the first row containing data",
    )
    parser.add_argument(
        "--header-row",
        type=int,
        help="1-based index for the header row (optional)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5000,
        help="Number of rows to read per chunk",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="Text encoding used by the input file",
    )
    return parser.parse_args()


def build_config_from_args(args: argparse.Namespace, delimiter: str) -> ParserConfig:
    na_values = [token.strip() for token in args.na.split(",") if token.strip()]

    return ParserConfig(
        file_path=args.file,
        delimiter=delimiter,
        na_values=na_values or None,
        first_data_row=args.first_data_row,
        header_row=args.header_row,
        chunk_size=args.chunk_size,
        encoding=args.encoding,
    )


def main() -> None:
    args = parse_cli_arguments()
    if not args.file:
        root = tk.Tk()
        root.withdraw()
        root.update()
        selected_file = filedialog.askopenfilename(
            title="Select blood pressure data file",
            filetypes=[
                ("Delimited files", "*.csv *.tsv *.txt"),
                ("All files", "*.*"),
            ],
        )
        root.destroy()
        if not selected_file:
            print("No file selected. Exiting.")
            return
        args.file = selected_file
    delimiter = determine_delimiter(args)
    config = build_config_from_args(args, delimiter)
    run_interactive_session(config)


if __name__ == "__main__":
    main()
