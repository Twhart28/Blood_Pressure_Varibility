#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import tempfile
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path

# ---- deps -------------------------------------------------------
try:
    import pyarrow as pa
    import pyarrow.csv as pv
except Exception:
    sys.stderr.write("This script requires PyArrow. Try: pip install pyarrow\n")
    raise

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional


# ---- Header scan ------------------------------------------------
def find_data_start_and_titles(path: Path):
    """
    Find the first data row (after 'Range=') and build canonical column names:
    ['Index'] + channel_titles + ['Comment']
    """
    data_start = 0
    channel_titles = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if line.startswith("ChannelTitle="):
                parts = line.rstrip("\n").split("\t")
                channel_titles = parts[1:] if len(parts) > 1 else []
            if line.startswith("Range="):
                data_start = i + 1
    names = ["Index"] + channel_titles + ["Comment"]
    return data_start, names


# ---- Fast pre-sanitizer -----------------------------------------
def sanitize_to_temp(path: Path, data_start: int, expected_cols: int, delimiter="\t"):
    """
    Create a sanitized temp file with exactly `expected_cols` per data row:
    - Lines before data_start are copied as-is (header)
    - For data rows:
        * If len(parts) > expected_cols: overflow is joined into the last column
        * If len(parts) < expected_cols: pad with empty strings
    Returns the temp file Path.
    """
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tsv")
    tmp_path = Path(tmp.name)
    tmp.close()

    with path.open("r", encoding="utf-8", errors="ignore") as src, \
         tmp_path.open("w", encoding="utf-8", errors="ignore") as dst:
        for i, line in enumerate(src):
            # Keep header untouched
            if i < data_start:
                dst.write(line)
                continue

            # Normalize line endings
            line = line.rstrip("\n").rstrip("\r")

            # Skip true empty lines in data section
            if not line:
                continue

            parts = line.split(delimiter)

            if len(parts) > expected_cols:
                # join overflow into the last column (Comment)
                head = parts[:expected_cols-1]
                tail = delimiter.join(parts[expected_cols-1:])
                parts = head + [tail]
            elif len(parts) < expected_cols:
                parts = parts + [""] * (expected_cols - len(parts))

            dst.write(delimiter.join(parts) + "\n")

    return tmp_path


def interpret_delimiter(value: str) -> str:
    """Turn user-entered delimiter text into the literal character(s)."""
    text = (value or "").strip()
    if not text:
        return "\t"

    lowered = text.lower()
    if lowered in {"\\t", "\t", "tab"}:
        return "\t"
    if lowered in {"\\n", "\n", "newline"}:
        return "\n"
    if lowered in {"\\r", "\r"}:
        return "\r"

    # Try to interpret escaped sequences (e.g., "\\x1f")
    try:
        return bytes(text, "utf-8").decode("unicode_escape")
    except Exception:
        return text


def read_header_columns(path: Path, header_row: int, delimiter: str) -> List[str]:
    """Read column names from the (1-based) header_row of the file."""
    target_idx = header_row - 1
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for i, line in enumerate(fh):
            if i == target_idx:
                line = line.rstrip("\n").rstrip("\r")
                return [part.strip() for part in line.split(delimiter)]
    raise ValueError(f"Header row {header_row} not found in file")


def parse_manual_columns(text: str, delimiter: str) -> List[str]:
    """Parse a comma- or delimiter-separated list of column names."""
    if not text:
        return []
    cleaned = text.strip()
    if not cleaned:
        return []

    if delimiter and delimiter in cleaned and delimiter != "\t":
        parts = cleaned.split(delimiter)
    else:
        parts = cleaned.split(",")
    return [part.strip() for part in parts if part.strip()]


# ---- PyArrow read -----------------------------------------------
def read_with_pyarrow(path, data_start, names, selected_cols, include_comment,
                      row_start=None, row_end=None, delimiter="\t", comment_name: Optional[str] = None):
    read_options = pv.ReadOptions(
        skip_rows=data_start,
        autogenerate_column_names=True,
        use_threads=True
    )
    parse_options = pv.ParseOptions(delimiter=delimiter)
    convert_options = pv.ConvertOptions(
        strings_can_be_null=True,
        null_values=["NaN"]
    )

    table = pv.read_csv(
        str(path),
        read_options=read_options,
        parse_options=parse_options,
        convert_options=convert_options
    )

    # Rename columns to our canonical list (trim/pad names safely)
    current_n = table.num_columns
    wanted_n = len(names)
    if current_n >= wanted_n:
        table = table.rename_columns(names + [f"EXTRA_{i}" for i in range(current_n - wanted_n)])
    else:
        padded = names + [f"EXTRA_{i}" for i in range(wanted_n - current_n)]
        table = table.rename_columns(padded[:current_n])

    # Select desired columns
    usecols = list(selected_cols)
    comment_candidates = []
    if comment_name:
        comment_candidates.append(comment_name)
    if not comment_name or comment_name != "Comment":
        comment_candidates.append("Comment")
    if not include_comment:
        for candidate in comment_candidates:
            if candidate in usecols:
                usecols.remove(candidate)
    table = table.select(usecols)

    # Optional row slicing
    if (row_start is not None) or (row_end is not None):
        start = int(row_start or 0)
        if (row_end is not None) and (row_end > start):
            length = int(row_end - start)
            table = table.slice(start, length)
        else:
            table = table.slice(start)

    df = table.to_pandas(types_mapper=pd.ArrowDtype)
    return df


# ---- UI ---------------------------------------------------------
class LoaderUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Flexible Delimited Loader (PyArrow + Comment-safe)")
        self.geometry("860x640")

        self.path_var = tk.StringVar()
        self.include_comment_var = tk.BooleanVar(value=False)
        self.row_start_var = tk.StringVar(value="")
        self.row_end_var = tk.StringVar(value="")
        self.delimiter_var = tk.StringVar(value="\\t")
        self.header_row_var = tk.StringVar(value="")
        self.data_start_var = tk.StringVar(value="")
        self.manual_cols_var = tk.StringVar(value="")
        self.comment_name_var = tk.StringVar(value="Comment")
        self.index_name_var = tk.StringVar(value="Index")

        self.names = None
        self.data_start = None
        self.current_delimiter = "\t"

        # File picker
        frm_file = ttk.Frame(self)
        frm_file.pack(fill="x", padx=10, pady=10)
        ttk.Label(frm_file, text="File:").pack(side="left")
        ttk.Entry(frm_file, textvariable=self.path_var).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(frm_file, text="Browse...", command=self.pick_file).pack(side="left")

        # Format settings
        frm_format = ttk.LabelFrame(self, text="Format settings")
        frm_format.pack(fill="x", padx=10, pady=(0, 10))
        for col in (1, 3):
            frm_format.columnconfigure(col, weight=1)

        ttk.Label(frm_format, text="Delimiter:").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        ttk.Entry(frm_format, width=12, textvariable=self.delimiter_var).grid(row=0, column=1, sticky="ew", padx=4, pady=4)
        ttk.Label(frm_format, text="Header row (1-based, optional):").grid(row=0, column=2, sticky="w", padx=4, pady=4)
        ttk.Entry(frm_format, width=12, textvariable=self.header_row_var).grid(row=0, column=3, sticky="ew", padx=4, pady=4)

        ttk.Label(frm_format, text="First data row (1-based, optional):").grid(row=1, column=0, sticky="w", padx=4, pady=4)
        ttk.Entry(frm_format, width=12, textvariable=self.data_start_var).grid(row=1, column=1, sticky="ew", padx=4, pady=4)
        ttk.Label(frm_format, text="Manual column names (comma-separated, optional):").grid(row=1, column=2, sticky="w", padx=4, pady=4)
        ttk.Entry(frm_format, textvariable=self.manual_cols_var).grid(row=1, column=3, sticky="ew", padx=4, pady=4)

        ttk.Label(frm_format, text="Index column name:").grid(row=2, column=0, sticky="w", padx=4, pady=4)
        ttk.Entry(frm_format, width=18, textvariable=self.index_name_var).grid(row=2, column=1, sticky="ew", padx=4, pady=4)
        ttk.Label(frm_format, text="Comment column name:").grid(row=2, column=2, sticky="w", padx=4, pady=4)
        ttk.Entry(frm_format, width=18, textvariable=self.comment_name_var).grid(row=2, column=3, sticky="ew", padx=4, pady=4)

        ttk.Label(frm_format, text="Provide either manual column names or the header row number.").grid(row=3, column=0, columnspan=4, sticky="w", padx=4, pady=(0, 4))

        frm_format_buttons = ttk.Frame(frm_format)
        frm_format_buttons.grid(row=4, column=0, columnspan=4, sticky="e", padx=4, pady=4)
        ttk.Button(frm_format_buttons, text="Auto-detect (original format)", command=self.autodetect_format).pack(side="right")
        ttk.Button(frm_format_buttons, text="Apply format", command=self.apply_format_settings).pack(side="right", padx=(0, 8))

        # Columns list
        frm_cols = ttk.LabelFrame(self, text="Select columns to load (Ctrl/Cmd + click for multi-select)")
        frm_cols.pack(fill="both", expand=True, padx=10, pady=10)

        self.lst_cols = tk.Listbox(frm_cols, selectmode="extended", height=14)
        self.lst_cols.pack(side="left", fill="both", expand=True, padx=(8,4), pady=8)
        self.scr_cols = ttk.Scrollbar(frm_cols, orient="vertical", command=self.lst_cols.yview)
        self.scr_cols.pack(side="left", fill="y", pady=8)
        self.lst_cols.config(yscrollcommand=self.scr_cols.set)

        # Options
        frm_opts = ttk.Frame(self)
        frm_opts.pack(fill="x", padx=10, pady=4)
        ttk.Checkbutton(frm_opts, text="Include trailing Comment column",
                        variable=self.include_comment_var).pack(side="left", padx=(0,20))
        ttk.Label(frm_opts, text="Row start (optional):").pack(side="left")
        ttk.Entry(frm_opts, width=10, textvariable=self.row_start_var).pack(side="left", padx=(4,20))
        ttk.Label(frm_opts, text="Row end (optional):").pack(side="left")
        ttk.Entry(frm_opts, width=10, textvariable=self.row_end_var).pack(side="left", padx=4)

        # Buttons
        frm_btns = ttk.Frame(self)
        frm_btns.pack(fill="x", padx=10, pady=10)
        ttk.Button(frm_btns, text="Load & Plot", command=self.load_and_plot).pack(side="right")
        ttk.Button(frm_btns, text="Quit", command=self.destroy).pack(side="right", padx=6)

        # Status
        self.status_var = tk.StringVar(value="Pick a file to begin.")
        ttk.Label(self, textvariable=self.status_var, anchor="w").pack(fill="x", padx=10, pady=(0,10))

    def pick_file(self):
        path = filedialog.askopenfilename(
            title="Select data file",
            filetypes=[("Delimited text files", "*.txt *.tsv *.csv"), ("All files", "*.*")]
        )
        if not path:
            return
        self.path_var.set(path)
        self.names = None
        self.data_start = None
        self.lst_cols.delete(0, "end")
        self.status_var.set("File selected. Configure the format settings and click Apply.")
        # Try original auto-detect silently for convenience
        self.autodetect_format(show_errors=False)

    def _parse_optional_line(self, value: str, label: str) -> Optional[int]:
        text = (value or "").strip()
        if not text:
            return None
        try:
            number = int(text)
        except ValueError:
            messagebox.showerror("Invalid input", f"{label} must be an integer (got '{value}').")
            raise ValueError from None
        if number < 1:
            messagebox.showerror("Invalid input", f"{label} must be greater than or equal to 1.")
            raise ValueError
        return number

    def _parse_non_negative(self, value: str, label: str) -> Optional[int]:
        text = (value or "").strip()
        if not text:
            return None
        try:
            number = int(text)
        except ValueError:
            messagebox.showerror("Invalid input", f"{label} must be an integer (got '{value}').")
            raise ValueError from None
        if number < 0:
            messagebox.showerror("Invalid input", f"{label} must be zero or a positive integer.")
            raise ValueError
        return number

    def populate_columns(self, names: List[str], preselect: Optional[List[str]] = None) -> None:
        self.lst_cols.delete(0, "end")
        if not names:
            return
        for name in names:
            self.lst_cols.insert("end", name)

        if preselect is None:
            preselect = []
            index_name = (self.index_name_var.get() or "").strip()
            if index_name and index_name in names:
                preselect.append(index_name)
            for candidate in ["Heart Rate", "HR", "SBP", "DBP", "MAP"]:
                if candidate in names and candidate not in preselect:
                    preselect.append(candidate)

        selected_names = set(preselect)
        for i, name in enumerate(names):
            if name in selected_names:
                self.lst_cols.selection_set(i)

    def apply_format_settings(self):
        path_str = self.path_var.get().strip()
        if not path_str:
            messagebox.showwarning("No file", "Please choose a file first.")
            return
        path = Path(path_str)
        if not path.exists():
            messagebox.showerror("File missing", "The selected file no longer exists.")
            return

        delimiter = interpret_delimiter(self.delimiter_var.get())

        try:
            header_row = self._parse_optional_line(self.header_row_var.get(), "Header row")
        except ValueError:
            return

        try:
            first_data_row = self._parse_optional_line(self.data_start_var.get(), "First data row")
        except ValueError:
            return

        manual_columns = parse_manual_columns(self.manual_cols_var.get(), delimiter)
        if not manual_columns and header_row is None:
            messagebox.showerror(
                "Missing format information",
                "Provide manual column names or specify the header row number, or use Auto-detect."
            )
            return

        if manual_columns:
            names = manual_columns
        else:
            try:
                names = read_header_columns(path, header_row, delimiter)
            except Exception as exc:
                messagebox.showerror("Header read error", str(exc))
                return

        if not names:
            messagebox.showerror("No columns", "Could not determine any column names with the provided settings.")
            return

        if first_data_row is None:
            if header_row is not None:
                first_data_row = header_row + 1
            else:
                first_data_row = 1

        skip_rows = max(0, first_data_row - 1)

        self.names = names
        self.data_start = skip_rows
        self.current_delimiter = delimiter

        # Normalize stored inputs
        self.data_start_var.set(str(skip_rows + 1))
        if header_row is not None:
            self.header_row_var.set(str(header_row))
        if manual_columns:
            self.manual_cols_var.set(", ".join(manual_columns))

        if not self.index_name_var.get().strip():
            self.index_name_var.set("Index")
        if not self.comment_name_var.get().strip():
            self.comment_name_var.set("Comment")

        self.populate_columns(names)
        self.status_var.set(
            f"Applied format. {len(names)} columns detected. First data row: {skip_rows + 1}."
        )

    def autodetect_format(self, show_errors: bool = True):
        path_str = self.path_var.get().strip()
        if not path_str:
            if show_errors:
                messagebox.showwarning("No file", "Please choose a file first.")
            return

        path = Path(path_str)
        if not path.exists():
            if show_errors:
                messagebox.showerror("File missing", "The selected file no longer exists.")
            return

        try:
            data_start, names = find_data_start_and_titles(path)
        except Exception as exc:
            if show_errors:
                messagebox.showerror("Auto-detect failed", str(exc))
            self.status_var.set("Auto-detect failed. Please configure the format manually and click Apply.")
            return

        if not names:
            if show_errors:
                messagebox.showerror("Auto-detect failed", "No column names were discovered.")
            self.status_var.set("Auto-detect could not determine column names. Configure manually.")
            return

        self.names = names
        self.data_start = data_start
        self.current_delimiter = "\t"
        self.delimiter_var.set("\\t")
        self.data_start_var.set(str(data_start + 1))
        self.header_row_var.set("")
        self.manual_cols_var.set(", ".join(names))
        self.index_name_var.set("Index")
        self.comment_name_var.set("Comment")
        self.populate_columns(names)
        self.status_var.set(
            f"Auto-detected format. First data row: {data_start + 1}. Review settings and adjust if needed."
        )

    def load_and_plot(self):
        path = self.path_var.get().strip()
        if not path:
            messagebox.showwarning("No file", "Please choose a file first.")
            return
        if not self.names or self.data_start is None:
            messagebox.showwarning(
                "Format not applied",
                "Apply the format settings before loading the data."
            )
            return

        sel_indices = self.lst_cols.curselection()
        if not sel_indices:
            messagebox.showwarning("No columns", "Select at least one column (e.g., Index + a signal).")
            return
        selected_cols = [self.names[i] for i in sel_indices]

        comment_name = (self.comment_name_var.get() or "").strip()
        include_comment = bool(self.include_comment_var.get())
        if not include_comment and comment_name:
            selected_cols = [col for col in selected_cols if col != comment_name]
            if not selected_cols:
                messagebox.showwarning(
                    "No columns",
                    "All selected columns were excluded because the comment column is disabled."
                )
                return

        # Row range
        try:
            row_start = self._parse_non_negative(self.row_start_var.get(), "Row start")
            row_end = self._parse_non_negative(self.row_end_var.get(), "Row end")
        except ValueError:
            return

        index_name = (self.index_name_var.get() or "").strip()

        # If including Comment, sanitize first so column counts are consistent
        src_path = Path(path)
        read_path = src_path
        tmp_for_cleanup = None

        try:
            if include_comment and comment_name and comment_name in self.names:
                expected_cols = len(self.names)
                self.status_var.set("Sanitizing comment column...")
                self.update_idletasks()
                tmp_for_cleanup = sanitize_to_temp(
                    src_path,
                    self.data_start,
                    expected_cols,
                    delimiter=self.current_delimiter
                )
                read_path = tmp_for_cleanup

            self.status_var.set("Reading with PyArrow...")
            self.update_idletasks()

            df = read_with_pyarrow(
                read_path,
                self.data_start,
                self.names,
                selected_cols,
                include_comment,
                row_start=row_start,
                row_end=row_end,
                delimiter=self.current_delimiter,
                comment_name=comment_name or None
            )

            # Clean up temp file
            if tmp_for_cleanup is not None and tmp_for_cleanup.exists():
                try:
                    tmp_for_cleanup.unlink()
                except Exception:
                    pass

            # Plot
            exclusions = {index_name}
            if comment_name:
                exclusions.add(comment_name)
            exclusions = {col for col in exclusions if col}
            numeric_cols = [c for c in df.columns if c not in exclusions]
            if not numeric_cols:
                messagebox.showinfo("Loaded", f"Loaded {len(df)} rows, but no numeric columns selected to plot.")
                return

            if index_name and index_name in df.columns:
                x = pd.to_numeric(df[index_name], errors="coerce").to_numpy()
            else:
                x = np.arange(len(df), dtype=np.int32)

            step = max(1, len(x) // 100_000)
            x_ds = x[::step]

            plt.figure()
            plotted_any = False
            for col in df.columns:
                if col == index_name or (comment_name and col == comment_name):
                    continue
                y = pd.to_numeric(df[col], errors="coerce").to_numpy()
                plt.plot(x_ds, y[::step], label=col)
                plotted_any = True

            if not plotted_any:
                messagebox.showinfo("Loaded", f"Loaded {len(df)} rows, but nothing numeric to plot.")
                return

            plt.title(Path(path).name)
            plt.xlabel(index_name if index_name else "Index (samples)")
            plt.ylabel("Value")
            plt.legend()
            plt.tight_layout()
            plt.show()

            self.status_var.set(f"Loaded {len(df)} rows. Plotted {len(x_ds)} points per selected signal.")

        except Exception as e:
            messagebox.showerror("Read error", str(e))
            self.status_var.set("Read failed.")
        finally:
            # Ensure temp file is removed on any error path
            if 'tmp_for_cleanup' in locals() and tmp_for_cleanup is not None and tmp_for_cleanup.exists():
                try:
                    tmp_for_cleanup.unlink()
                except Exception:
                    pass


if __name__ == "__main__":
    app = LoaderUI()
    app.mainloop()
