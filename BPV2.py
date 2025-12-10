"""Minimal blood pressure viewer.

The script loads a delimited file, prompts for the header and data rows,
lets the user choose the time and pressure columns, then plots the raw
trace without any downstream analysis.
"""

from __future__ import annotations

import csv
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, ttk, messagebox

import matplotlib.pyplot as plt
import pandas as pd


class ImportSelection:
    def __init__(
        self,
        separator: str,
        header_row: int,
        first_data_row: int,
        time_column: int,
        pressure_column: int,
    ) -> None:
        self.separator = separator
        self.header_row = header_row
        self.first_data_row = first_data_row
        self.time_column = time_column
        self.pressure_column = pressure_column


def _detect_separator(lines: list[str]) -> str:
    sample = "\n".join(lines[:20])
    try:
        dialect = csv.Sniffer().sniff(sample)
        return dialect.delimiter  # type: ignore[return-value]
    except Exception:
        return ","


def _split_preview_lines(lines: list[str], separator: str) -> list[list[str]]:
    reader = csv.reader(lines, delimiter=separator)
    return [list(row) for row in reader]


def _render_preview(
    widget: tk.Text,
    rows: list[list[str]],
    *,
    header_row_index: int,
    time_column_index: int | None,
    pressure_column_index: int | None,
    max_rows: int = 200,
) -> None:
    widget.configure(state="normal")
    widget.delete("1.0", tk.END)

    # Column index header
    if rows:
        column_count = max(len(row) for row in rows[:max_rows])
    else:
        column_count = 0
    index_line = [f"[{idx + 1}]" for idx in range(column_count)]
    widget.insert(tk.END, "\t".join(index_line) + "\n", ("column_index",))

    for row_idx, row in enumerate(rows[:max_rows], start=1):
        tags: list[str] = []
        if row_idx == header_row_index:
            tags.append("header")
        for col_idx, value in enumerate(row):
            cell_tags = list(tags)
            if time_column_index is not None and col_idx == time_column_index:
                cell_tags.append("highlight_time")
            if pressure_column_index is not None and col_idx == pressure_column_index:
                cell_tags.append("highlight_pressure")
            widget.insert(tk.END, str(value), tuple(cell_tags))
            if col_idx < len(row) - 1:
                widget.insert(tk.END, "\t")
        widget.insert(tk.END, "\n")

    widget.configure(state="disabled")


def _build_column_options(rows: list[list[str]], header_row: int) -> list[tuple[str, int]]:
    if rows and 1 <= header_row <= len(rows):
        header_values = rows[header_row - 1]
    else:
        header_values = []

    column_count = max((len(row) for row in rows[:50]), default=len(header_values) or 1)
    options: list[tuple[str, int]] = []
    for idx in range(column_count):
        label = header_values[idx].strip() if idx < len(header_values) else f"Column {idx + 1}"
        options.append((f"{idx + 1}: {label or 'Column'}", idx + 1))
    return options


def launch_import_dialog(file_path: Path, raw_lines: list[str]) -> ImportSelection | None:
    separator = _detect_separator(raw_lines)
    rows = _split_preview_lines(raw_lines, separator)

    root = tk.Tk()
    root.title("Import options")
    root.geometry("900x640")

    main_frame = ttk.Frame(root, padding=12)
    main_frame.grid(row=0, column=0, sticky="nsew")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    main_frame.columnconfigure(0, weight=1)
    main_frame.rowconfigure(2, weight=1)

    header_var = tk.IntVar(value=1)
    first_data_var = tk.IntVar(value=2)

    ttk.Label(main_frame, text=f"File: {file_path.name}").grid(row=0, column=0, sticky="w", pady=(0, 8))

    controls = ttk.Frame(main_frame)
    controls.grid(row=1, column=0, sticky="ew")
    controls.columnconfigure(4, weight=1)

    ttk.Label(controls, text="Delimiter:").grid(row=0, column=0, sticky="w")
    delimiter_var = tk.StringVar(value=separator)
    delimiter_entry = ttk.Entry(controls, textvariable=delimiter_var, width=6)
    delimiter_entry.grid(row=0, column=1, sticky="w", padx=(6, 18))

    ttk.Label(controls, text="Header row:").grid(row=0, column=2, sticky="w")
    header_spin = tk.Spinbox(controls, from_=1, to=10000, textvariable=header_var, width=6)
    header_spin.grid(row=0, column=3, sticky="w", padx=(6, 18))

    ttk.Label(controls, text="First data row:").grid(row=0, column=4, sticky="w")
    first_data_spin = tk.Spinbox(controls, from_=1, to=10000, textvariable=first_data_var, width=6)
    first_data_spin.grid(row=0, column=5, sticky="w", padx=(6, 18))

    time_var = tk.StringVar()
    pressure_var = tk.StringVar()

    ttk.Label(controls, text="Time column:").grid(row=1, column=0, sticky="w", pady=(10, 0))
    time_combo = ttk.Combobox(controls, state="readonly", textvariable=time_var, width=24)
    time_combo.grid(row=1, column=1, columnspan=2, sticky="w", padx=(6, 18), pady=(10, 0))

    ttk.Label(controls, text="Pressure column:").grid(row=1, column=3, sticky="w", pady=(10, 0))
    pressure_combo = ttk.Combobox(controls, state="readonly", textvariable=pressure_var, width=24)
    pressure_combo.grid(row=1, column=4, columnspan=2, sticky="w", padx=(6, 0), pady=(10, 0))

    preview_frame = ttk.Frame(main_frame, relief=tk.SOLID, borderwidth=1)
    preview_frame.grid(row=2, column=0, sticky="nsew", pady=(12, 0))
    preview_frame.columnconfigure(0, weight=1)
    preview_frame.rowconfigure(0, weight=1)

    preview_text = tk.Text(preview_frame, wrap="none", font=("Courier New", 10))
    preview_text.grid(row=0, column=0, sticky="nsew")
    y_scroll = ttk.Scrollbar(preview_frame, orient="vertical", command=preview_text.yview)
    y_scroll.grid(row=0, column=1, sticky="ns")
    x_scroll = ttk.Scrollbar(preview_frame, orient="horizontal", command=preview_text.xview)
    x_scroll.grid(row=1, column=0, sticky="ew")
    preview_text.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)

    preview_text.tag_configure("column_index", background="#e5ecff", font=("Courier New", 10, "bold"))
    preview_text.tag_configure("header", background="#d9d9d9", font=("Courier New", 10, "bold"))
    preview_text.tag_configure("highlight_time", background="#dbeafe")
    preview_text.tag_configure("highlight_pressure", background="#fdeac5")

    selection: ImportSelection | None = None

    def _parse_combo_value(value: str) -> int | None:
        try:
            return int(value.split(":", 1)[0])
        except Exception:
            return None

    def _update_options(*_: object) -> None:
        nonlocal rows
        separator_value = delimiter_var.get() or ","
        rows = _split_preview_lines(raw_lines, separator_value)
        header_index = max(1, int(header_var.get() or 1))
        options = _build_column_options(rows, header_index)
        labels = [label for label, _ in options]
        time_combo.configure(values=labels)
        pressure_combo.configure(values=labels)
        if labels:
            if not time_var.get():
                time_var.set(labels[0])
            if not pressure_var.get():
                pressure_var.set(labels[-1])
        _render_preview(
            preview_text,
            rows,
            header_row_index=header_index,
            time_column_index=_parse_combo_value(time_var.get()) - 1 if time_var.get() else None,
            pressure_column_index=_parse_combo_value(pressure_var.get()) - 1 if pressure_var.get() else None,
        )

    def _confirm() -> None:
        nonlocal selection
        time_index = _parse_combo_value(time_var.get())
        pressure_index = _parse_combo_value(pressure_var.get())
        if time_index is None or pressure_index is None:
            messagebox.showerror("Missing selection", "Please choose both time and pressure columns.")
            return
        selection = ImportSelection(
            separator=delimiter_var.get() or ",",
            header_row=max(1, int(header_var.get() or 1)),
            first_data_row=max(1, int(first_data_var.get() or 1)),
            time_column=time_index,
            pressure_column=pressure_index,
        )
        root.destroy()

    def _cancel() -> None:
        root.destroy()

    button_frame = ttk.Frame(main_frame)
    button_frame.grid(row=3, column=0, sticky="e", pady=12)
    ttk.Button(button_frame, text="Cancel", command=_cancel).grid(row=0, column=0, padx=(0, 8))
    ttk.Button(button_frame, text="Import", command=_confirm).grid(row=0, column=1)

    header_var.trace_add("write", _update_options)
    delimiter_var.trace_add("write", _update_options)
    time_var.trace_add("write", _update_options)
    pressure_var.trace_add("write", _update_options)

    _update_options()
    root.mainloop()
    return selection


def _read_raw_lines(path: Path, limit: int = 200) -> list[str]:
    lines: list[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for _ in range(limit):
            line = handle.readline()
            if not line:
                break
            lines.append(line.rstrip("\n"))
    return lines


def plot_selected_columns(file_path: Path, selection: ImportSelection) -> None:
    def skip_logic(index: int) -> bool:
        zero_based = index
        header_idx = selection.header_row - 1
        first_data_idx = selection.first_data_row - 1
        return zero_based < first_data_idx and zero_based != header_idx

    df = pd.read_csv(
        file_path,
        delimiter=selection.separator,
        header=selection.header_row - 1,
        skiprows=lambda x: skip_logic(x),
    )

    time_series = df.iloc[:, selection.time_column - 1]
    pressure_series = df.iloc[:, selection.pressure_column - 1]

plt.figure(figsize=(10, 4))
    plt.plot(time_series, pressure_series, marker="o", linestyle="-", linewidth=1)
    plt.title("Raw pressure trace")
    plt.xlabel(time_series.name or "Time")
    plt.ylabel(pressure_series.name or "Pressure")
    plt.tight_layout()
    plt.show()


def main() -> None:
    root = tk.Tk()
    root.withdraw()
    file_path_str = filedialog.askopenfilename(title="Select data file")
    root.destroy()
    if not file_path_str:
        return

    file_path = Path(file_path_str)
    raw_lines = _read_raw_lines(file_path)
    if not raw_lines:
        messagebox.showerror("Empty file", "No data found in the selected file.")
        return

    selection = launch_import_dialog(file_path, raw_lines)
    if selection is None:
        return

    plot_selected_columns(file_path, selection)


if __name__ == "__main__":
    main()