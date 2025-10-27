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
from tkinter import filedialog

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

    try:
        with open(file_path, "r", encoding=encoding, newline="") as handle:
            sample = handle.read(sample_size)
            if not sample:
                return detected, sample_line, detected_from_sniffer
            if "\n" not in sample:
                sample += handle.readline()
            sample_line = sample.splitlines()[0] if sample else ""

            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(sample, delimiters=COMMON_DELIMITERS)
            detected = dialect.delimiter
            detected_from_sniffer = True
    except (csv.Error, OSError, UnicodeDecodeError):
        pass

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


class DelimiterSelectionDialog:
    """Tkinter dialog that lets users choose a delimiter interactively."""

    _CUSTOM_SENTINEL = "__custom__"

    def __init__(
        self,
        detected: str,
        sample_line: str,
        detected_from_sniffer: bool,
        fallback: str = ",",
    ) -> None:
        self._default_delimiter = detected or fallback
        self._sample_line = sample_line
        self._detected_from_sniffer = detected_from_sniffer
        self._result: Optional[str] = self._default_delimiter

        self.root = tk.Tk()
        self.root.title("Select Delimiter")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close_attempt)

        self._selection = tk.StringVar(value=self._default_delimiter)
        self._custom_value = tk.StringVar()
        self._status_message = tk.StringVar(value="")
        self._column_count_var = tk.StringVar(value="")

        self._build_widgets()
        self._update_preview()

    # ------------------------------------------------------------------
    def _build_widgets(self) -> None:
        frame = tk.Frame(self.root, padx=10, pady=10)
        frame.pack(fill=tk.BOTH, expand=True)

        title = tk.Label(
            frame,
            text="Choose the delimiter that matches your file",
            font=("TkDefaultFont", 11, "bold"),
        )
        title.pack(anchor=tk.W)

        description = self._build_description()
        if description:
            tk.Label(frame, text=description, wraplength=420, justify=tk.LEFT).pack(
                anchor=tk.W, pady=(4, 8)
            )

        if self._sample_line:
            sample_frame = tk.LabelFrame(frame, text="Sample row from file")
            sample_frame.pack(fill=tk.X, pady=(0, 10))
            sample_text = tk.Text(sample_frame, height=3, wrap=tk.NONE)
            sample_text.insert(tk.END, self._sample_line)
            sample_text.configure(state=tk.DISABLED)
            sample_text.pack(fill=tk.X)

        options_frame = tk.Frame(frame)
        options_frame.pack(fill=tk.BOTH, expand=True)

        tk.Label(options_frame, text="Common delimiters:").pack(anchor=tk.W)

        options_container = tk.Frame(options_frame)
        options_container.pack(fill=tk.X, pady=(2, 6))

        for delimiter in self._candidate_delimiters():
            display = self._format_option_label(delimiter)
            tk.Radiobutton(
                options_container,
                text=display,
                value=delimiter,
                variable=self._selection,
                command=self._update_preview,
                anchor=tk.W,
                justify=tk.LEFT,
                padx=5,
            ).pack(fill=tk.X, anchor=tk.W)

        custom_frame = tk.Frame(frame)
        custom_frame.pack(fill=tk.X, pady=(6, 10))

        tk.Radiobutton(
            custom_frame,
            text="Custom delimiter:",
            value=self._CUSTOM_SENTINEL,
            variable=self._selection,
            command=self._update_preview,
        ).pack(side=tk.LEFT)

        entry = tk.Entry(custom_frame, textvariable=self._custom_value, width=12)
        entry.pack(side=tk.LEFT, padx=(6, 0))
        entry.bind("<FocusIn>", lambda _event: self._selection.set(self._CUSTOM_SENTINEL))
        self._custom_value.trace_add("write", lambda *_args: self._update_preview())

        preview_frame = tk.LabelFrame(frame, text="Preview")
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self._preview_text = tk.Text(preview_frame, height=6, wrap=tk.NONE)
        self._preview_text.configure(state=tk.DISABLED)
        self._preview_text.pack(fill=tk.BOTH, expand=True)

        tk.Label(frame, textvariable=self._column_count_var).pack(anchor=tk.W)
        status = tk.Label(frame, textvariable=self._status_message, fg="#b22222")
        status.pack(fill=tk.X, pady=(4, 0))

        buttons = tk.Frame(frame)
        buttons.pack(fill=tk.X, pady=(10, 0))

        tk.Button(buttons, text="Cancel", command=self._on_close_attempt).pack(side=tk.RIGHT)
        tk.Button(buttons, text="Confirm", command=self._confirm).pack(
            side=tk.RIGHT, padx=(0, 6)
        )

    # ------------------------------------------------------------------
    def _build_description(self) -> str:
        if self._detected_from_sniffer:
            message = _format_delimiter_for_display(self._default_delimiter)
            return f"Automatically detected delimiter: {message}."
        if self._default_delimiter:
            message = _format_delimiter_for_display(self._default_delimiter)
            return (
                "Unable to confidently detect the delimiter. "
                f"Defaulting to {message}. Choose a different option if needed."
            )
        return ""

    # ------------------------------------------------------------------
    def _candidate_delimiters(self) -> List[str]:
        candidates: List[str] = []
        seen = set()
        for delimiter in [self._default_delimiter] + COMMON_DELIMITERS:
            if delimiter and delimiter not in seen:
                seen.add(delimiter)
                candidates.append(delimiter)
        if not candidates:
            candidates.append(",")
        return candidates

    # ------------------------------------------------------------------
    def _format_option_label(self, delimiter: str) -> str:
        label = _format_delimiter_for_display(delimiter)
        if self._sample_line:
            columns = len(self._sample_line.split(delimiter))
            return f"{label} — {columns} column(s) in sample"
        return label

    # ------------------------------------------------------------------
    def _resolve_selection(self) -> Optional[str]:
        selected = self._selection.get()
        if selected == self._CUSTOM_SENTINEL:
            value = self._custom_value.get().strip()
            if not value:
                return None
            return interpret_delimiter(value)
        return selected

    # ------------------------------------------------------------------
    def _update_preview(self) -> None:
        delimiter = self._resolve_selection()
        if delimiter is None:
            self._status_message.set("Enter a custom delimiter to preview the data.")
            columns = []
        else:
            self._status_message.set("")
            columns = (
                self._sample_line.split(delimiter) if self._sample_line and delimiter else []
            )

        self._column_count_var.set(
            f"Columns detected: {len(columns)}" if columns else "Columns detected: 0"
        )

        self._preview_text.configure(state=tk.NORMAL)
        self._preview_text.delete("1.0", tk.END)
        if columns:
            for index, value in enumerate(columns, start=1):
                self._preview_text.insert(tk.END, f"{index}. {value}\n")
        else:
            self._preview_text.insert(
                tk.END,
                "No preview available. Adjust the delimiter or ensure the sample row is not empty.",
            )
        self._preview_text.configure(state=tk.DISABLED)

    # ------------------------------------------------------------------
    def _confirm(self) -> None:
        delimiter = self._resolve_selection()
        if delimiter is None:
            self._status_message.set("Provide a custom delimiter before confirming.")
            return
        self._result = delimiter
        self.root.destroy()

    # ------------------------------------------------------------------
    def _on_close_attempt(self) -> None:
        self._result = self._resolve_selection() or self._default_delimiter
        self.root.destroy()

    # ------------------------------------------------------------------
    def show(self) -> str:
        self.root.mainloop()
        return self._result or self._default_delimiter


def determine_delimiter(args: argparse.Namespace) -> str:
    """Resolve the delimiter from CLI arguments or interactive selection."""

    if args.delimiter:
        return interpret_delimiter(args.delimiter)

    detected, sample_line, detected_from_sniffer = detect_delimiter(
        args.file, args.encoding
    )
    dialog = DelimiterSelectionDialog(detected, sample_line, detected_from_sniffer)
    return dialog.show()


def read_header_row(config: ParserConfig) -> Optional[List[str]]:
    """Return column names from the configured header row if available."""

    if config.header_row is None:
        return None

    with open(config.file_path, "r", encoding=config.encoding, newline="") as handle:
        reader = csv.reader(handle, delimiter=config.delimiter)
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
    read_kwargs = {
        "sep": config.delimiter,
        "na_values": config.na_values,
        "chunksize": config.chunk_size,
        "skiprows": skiprows,
        "header": None,
        "encoding": config.encoding,
        "engine": "python",
    }
    if column_names:
        read_kwargs["names"] = column_names

    chunk_iterator = pd.read_csv(config.file_path, **read_kwargs)

    inferred_names: Optional[List[str]] = column_names
    for chunk in chunk_iterator:
        if inferred_names is None:
            inferred_names = [f"column_{idx + 1}" for idx in range(len(chunk.columns))]
        chunk.columns = inferred_names
        yield chunk


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
