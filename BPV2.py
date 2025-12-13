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
        analysis_downsample: int,
        plot_downsample: int,
    ) -> None:
        self.separator = separator
        self.header_row = header_row
        self.first_data_row = first_data_row
        self.time_column = time_column
        self.pressure_column = pressure_column
        self.analysis_downsample = analysis_downsample
        self.plot_downsample = plot_downsample


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
    first_data_row_index: int,
    time_column_index: int | None,
    pressure_column_index: int | None,
    max_rows: int = 200,
) -> None:
    widget.configure(state="normal")
    widget.delete("1.0", tk.END)

    if not rows:
        widget.insert("end", "No preview data available.\n")
        widget.configure(state="disabled")
        return

    header_row_index = max(1, header_row_index)
    first_data_row_index = max(1, first_data_row_index)

    widths: list[int] = []
    sample_rows = list(rows[:max_rows])
    for row in sample_rows:
        for idx, value in enumerate(row):
            text_value = str(value) if value is not None else ""
            if idx >= len(widths):
                widths.append(len(text_value))
            else:
                widths[idx] = max(widths[idx], len(text_value))

    line_no = 1
    for row_number, row_values in enumerate(sample_rows, start=1):
        line_parts: list[str] = []
        positions: list[tuple[int, int, int]] = []
        cursor = 0
        for col_idx, width in enumerate(widths):
            value = str(row_values[col_idx]) if col_idx < len(row_values) and row_values[col_idx] is not None else ""
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
    delimiter_options: list[tuple[str, str | None]] = [
        ("Auto-detect", None),
        ("Tab (\t)", "\t"),
        ("Comma (,)", ","),
        ("Semicolon (;)", ";"),
        ("Pipe (|)", "|"),
        ("Space", " "),
        ("Colon (:)", ":"),
    ]

    downsample_options: list[tuple[str, int]] = [
        ("Full resolution (1×)", 1),
        ("Every 2nd sample (2×)", 2),
        ("Every 4th sample (4×)", 4),
        ("Every 5th sample (5×)", 5),
        ("Every 10th sample (10×)", 10),
        ("Every 20th sample (20×)", 20),
        ("Every 50th sample (50×)", 50),
        ("Every 100th sample (100×)", 100),
    ]

    def _label_for_separator(value: str | None) -> str:
        for label, candidate in delimiter_options:
            if value == candidate:
                return label
        return "Auto-detect"

    def _resolve_separator_from_label(label: str) -> str | None:
        for option_label, value in delimiter_options:
            if option_label == label:
                return value
        return None

    def _label_for_downsample(value: int) -> str:
        for label, candidate in downsample_options:
            if value == candidate:
                return label
        return downsample_options[0][0]

    def _resolve_downsample_from_label(label: str) -> int:
        for option_label, value in downsample_options:
            if option_label == label:
                return value
        return downsample_options[0][1]

    detected_separator = _detect_separator(raw_lines)

    def _effective_separator(value: str | None) -> str:
        if value is not None:
            return value
        return detected_separator

    def _split_preview_lines_with_selection(label: str) -> list[list[str]]:
        return _split_preview_lines(raw_lines, _effective_separator(_resolve_separator_from_label(label)))

    root = tk.Tk()
    root.title("Import options")
    root.geometry("960x660")

    main_frame = ttk.Frame(root, padding=16)
    main_frame.grid(row=0, column=0, sticky="nsew")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    main_frame.columnconfigure(0, weight=1)
    main_frame.rowconfigure(3, weight=1)

    header_var = tk.IntVar(value=1)
    first_data_var = tk.IntVar(value=2)

    ttk.Label(main_frame, text=f"File: {file_path.name}").grid(row=0, column=0, sticky="w", pady=(0, 8))

    controls = ttk.Frame(main_frame)
    controls.grid(row=1, column=0, sticky="ew")
    controls.columnconfigure(6, weight=1)

    ttk.Label(controls, text="Delimiter:").grid(row=0, column=0, sticky="w")
    delimiter_var = tk.StringVar(value=_label_for_separator(detected_separator))
    delimiter_combo = ttk.Combobox(
        controls,
        state="readonly",
        values=[label for label, _ in delimiter_options],
        textvariable=delimiter_var,
        width=18,
    )
    delimiter_combo.grid(row=0, column=1, sticky="w", padx=(8, 24))

    ttk.Label(controls, text="Header row:").grid(row=0, column=2, sticky="w")
    header_spin = tk.Spinbox(controls, from_=1, to=10000, textvariable=header_var, width=8)
    header_spin.grid(row=0, column=3, sticky="w", padx=(8, 24))

    ttk.Label(controls, text="First data row:").grid(row=0, column=4, sticky="w")
    first_data_spin = tk.Spinbox(controls, from_=1, to=10000, textvariable=first_data_var, width=8)
    first_data_spin.grid(row=0, column=5, sticky="w")

    time_var = tk.StringVar()
    pressure_var = tk.StringVar()

    ttk.Label(controls, text="Time column:").grid(row=1, column=0, sticky="w", pady=(12, 0))
    time_combo = ttk.Combobox(controls, state="readonly", textvariable=time_var, width=24)
    time_combo.grid(row=1, column=1, sticky="w", padx=(8, 24), pady=(12, 0))

    ttk.Label(controls, text="Pressure column:").grid(row=1, column=2, sticky="w", pady=(12, 0))
    pressure_combo = ttk.Combobox(controls, state="readonly", textvariable=pressure_var, width=24)
    pressure_combo.grid(row=1, column=3, sticky="w", padx=(8, 24), pady=(12, 0))

    ttk.Label(controls, text="Analysis downsampling:").grid(row=2, column=0, sticky="w", pady=(12, 0))
    analysis_downsample_var = tk.StringVar(value=_label_for_downsample(1))
    analysis_downsample_combo = ttk.Combobox(
        controls,
        state="readonly",
        values=[label for label, _ in downsample_options],
        textvariable=analysis_downsample_var,
        width=28,
    )
    analysis_downsample_combo.grid(row=2, column=1, sticky="w", padx=(8, 24), pady=(12, 0))

    ttk.Label(controls, text="Plot downsampling:").grid(row=2, column=2, sticky="w", pady=(12, 0))
    plot_downsample_var = tk.StringVar(value=_label_for_downsample(10))
    plot_downsample_combo = ttk.Combobox(
        controls,
        state="readonly",
        values=[label for label, _ in downsample_options],
        textvariable=plot_downsample_var,
        width=28,
    )
    plot_downsample_combo.grid(row=2, column=3, sticky="w", padx=(8, 24), pady=(12, 0))

    ttk.Label(
        main_frame,
        text="Data preview (first 200 rows)",
        font=("TkDefaultFont", 10, "bold"),
    ).grid(row=2, column=0, sticky="w", pady=(16, 4))

    preview_frame = ttk.Frame(main_frame, relief=tk.SOLID, borderwidth=1)
    preview_frame.grid(row=3, column=0, sticky="nsew")
    preview_frame.columnconfigure(0, weight=1)
    preview_frame.rowconfigure(0, weight=1)

    preview_text = tk.Text(preview_frame, wrap="none", font=("Courier New", 10), height=20)
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

    rows = _split_preview_lines_with_selection(delimiter_var.get())
    selection: ImportSelection | None = None

    def _parse_combo_value(value: str) -> int | None:
        try:
            return int(value.split(":", 1)[0])
        except Exception:
            return None

    def _update_options(*_: object) -> None:
        nonlocal rows
        separator_label = delimiter_var.get()
        rows = _split_preview_lines_with_selection(separator_label)
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
            first_data_row_index=max(1, int(first_data_var.get() or 1)),
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
            separator=_effective_separator(_resolve_separator_from_label(delimiter_var.get())),
            header_row=max(1, int(header_var.get() or 1)),
            first_data_row=max(1, int(first_data_var.get() or 1)),
            time_column=time_index,
            pressure_column=pressure_index,
            analysis_downsample=_resolve_downsample_from_label(analysis_downsample_var.get()),
            plot_downsample=_resolve_downsample_from_label(plot_downsample_var.get()),
        )
        root.destroy()

    def _cancel() -> None:
        root.destroy()

    button_frame = ttk.Frame(main_frame)
    button_frame.grid(row=4, column=0, sticky="e", pady=12)
    ttk.Button(button_frame, text="Cancel", command=_cancel).grid(row=0, column=0, padx=(0, 8))
    ttk.Button(button_frame, text="Import", command=_confirm).grid(row=0, column=1)

    header_var.trace_add("write", _update_options)
    first_data_var.trace_add("write", _update_options)
    delimiter_var.trace_add("write", _update_options)
    time_var.trace_add("write", _update_options)
    pressure_var.trace_add("write", _update_options)
    analysis_downsample_var.trace_add("write", lambda *_: None)
    plot_downsample_var.trace_add("write", lambda *_: None)

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
    header_idx = selection.header_row - 1
    first_data_idx = selection.first_data_row - 1
    skip_rows = [idx for idx in range(first_data_idx) if idx != header_idx]

    selected_columns = [selection.time_column - 1, selection.pressure_column - 1]

    def _load_dataframe(engine: str, on_bad_lines: str | None = None) -> pd.DataFrame:
        return pd.read_csv(
            file_path,
            delimiter=selection.separator,
            header=header_idx,
            usecols=selected_columns,
            skiprows=skip_rows,
            engine=engine,
            on_bad_lines=on_bad_lines,
        )

    try:
        try:
            df = _load_dataframe(engine="c")
        except pd.errors.ParserError:
            # Fall back to the Python engine to tolerate ragged rows.
            df = _load_dataframe(engine="python", on_bad_lines="skip")
    except Exception as exc:
        messagebox.showerror(
            "Import error",
            "Failed to parse the selected file.\n"
            f"Details: {exc}",
        )
        return

    if df.shape[1] < 2:
        messagebox.showerror(
            "Import error",
            "Unable to read both selected columns. Please check your selections and try again.",
        )
        return

    time_series = df.iloc[:, 0]
    pressure_series = df.iloc[:, 1]

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
    file_path_str = filedialog.askopenfilename(
        title="Select continuous BP export",
        filetypes=[("Text files", "*.txt")],
    )
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